"""Mass-distance-time diagram, three-part front kinematics and drag from CME reconstructions.

Compares a SuNeRF reconstruction with a ground-truth density series (HAO or
PSI cubes) inside a cone around the CME direction, or evaluates the model alone
on a regular grid when no cubes are given.  No pre-event background is
subtracted; see :mod:`sunerf.evaluation.cme.mass_transport`.

Example (HAO 2-view reconstruction, frames 10-40 of the density series)::

    python -m sunerf.evaluation.cme.mass_transport_evaluation \
        --sunerf_path /path/save_state.snf --data_path "/path/density_cube/*.sav" \
        --dataset hao --date0 2010-04-03T09:04:00 --frames 10:40 \
        --target_latitude 0 --target_longitude 135
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from sunerf.evaluation.cme.density_series import load_series, synthetic_grid
from sunerf.evaluation.cme.export_density import model_query_points
from sunerf.evaluation.cme.mass_transport import (
    ambient_ahead_of_front, cone_mask, cone_summary, finite_difference_speed, fit_dbm, fit_speed,
    mass_per_radius, physical_drag_parameter, structure_maps,
)
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.evaluation.util import none_or_float

TRACKS = ('front_apex', 'front_median', 'sheath_median', 'cavity_median', 'ejecta_median')
LABELS = {'true': 'Ground truth', 'pred': 'SuNeRF'}
COLORS = {'true': 'k', 'pred': 'C1'}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--out_path', help='Output directory (default: <save state dir>/mass_transport)')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--trusted', action='store_true', help='Load a pickled (non weights-only) save state')
    # ground truth
    parser.add_argument('--data_path', help='Glob of ground-truth density cubes; omit for model-only evaluation')
    parser.add_argument('--dataset', choices=['hao', 'psi'], default='hao')
    parser.add_argument('--date0', default='2010-04-03T09:04:00.000', help='HAO simulation start time')
    parser.add_argument('--dump_times', help='PSI dump_times.json (default: 5 min cadence from 2021-10-28T15:30)')
    parser.add_argument('--frames', default=None, help='Slice of the sorted cube files, e.g. 10:40 or ::2')
    parser.add_argument('--stride', type=int, default=1, help='Angular subsampling of the ground-truth grid')
    # model-only grid
    parser.add_argument('--start_time', help='Model-only evaluation: first time (ISO)')
    parser.add_argument('--end_time', help='Model-only evaluation: last time (ISO)')
    parser.add_argument('--cadence', type=float, default=1.0, help='Model-only evaluation: cadence in hours')
    parser.add_argument('--n_radius', type=int, default=200)
    parser.add_argument('--angular_step', type=float, default=2.0, help='Model-only grid step in degrees')
    # region
    parser.add_argument('--min_radius', type=float, default=30.0)
    parser.add_argument('--max_radius', type=float, default=120.0)
    parser.add_argument('--min_latitude', type=none_or_float, default=-60.0)
    parser.add_argument('--max_latitude', type=none_or_float, default=60.0)
    parser.add_argument('--min_longitude', type=none_or_float, default=70.0)
    parser.add_argument('--max_longitude', type=none_or_float, default=200.0)
    parser.add_argument('--target_latitude', type=float, default=0.0, help='CME direction latitude [deg]')
    parser.add_argument('--target_longitude', type=float, default=135.0, help='CME direction longitude [deg]')
    parser.add_argument('--cone_half_angle', type=float, default=30.0, help='Cone around the CME direction [deg]')
    # structure detection
    parser.add_argument('--dr', type=float, default=0.5, help='Radial resampling step for ray profiles [R_sun]')
    parser.add_argument('--smooth', type=float, default=1.0, help='Smoothing of log(r^2 rho) [R_sun]')
    parser.add_argument('--min_prominence', type=float, default=0.15, help='Peak prominence [dex]')
    parser.add_argument('--min_gradient', type=float, default=0.05, help='Front steepness [dex / R_sun]')
    parser.add_argument('--r_search_min', type=float, default=None, help='Ignore structure below this radius')
    parser.add_argument('--ambient_offset', type=float, nargs=2, default=(1.0, 4.0),
                        help='Shell ahead of the front for the ambient density and wind [R_sun]')
    parser.add_argument('--drag_coefficient', type=float, default=1.0)
    return parser


def iterate_frames(args):
    if args.data_path:
        return load_series(args.data_path, args.dataset, frames=args.frames, date0=parse(args.date0),
                           dump_times_path=args.dump_times)
    if args.start_time is None or args.end_time is None:
        raise ValueError('Model-only evaluation needs --start_time and --end_time.')
    start, end = parse(args.start_time), parse(args.end_time)
    n = int((end - start).total_seconds() / (args.cadence * 3600)) + 1
    times = [start + timedelta(hours=args.cadence * i) for i in range(n)]
    return synthetic_grid(times, args.min_radius, args.max_radius, args.n_radius, args.min_latitude,
                          args.max_latitude, args.min_longitude, args.max_longitude, args.angular_step)


def evaluate_frame(rho, v_radial, frame, cone, args):
    """Mass distribution, structure maps and cone summary of one density cube."""
    detection = dict(dr=args.dr, smooth=args.smooth, min_prominence=args.min_prominence,
                     min_gradient=args.min_gradient, r_search_min=args.r_search_min)
    maps = structure_maps(rho, frame.radius, angular_mask=cone, **detection)
    summary = cone_summary(maps, frame.latitude, frame.longitude, cone)
    summary['ambient_density'] = ambient_ahead_of_front(rho, frame.radius, maps['front'], cone,
                                                        offset=args.ambient_offset)
    summary['ambient_wind'] = (ambient_ahead_of_front(v_radial, frame.radius, maps['front'], cone,
                                                      offset=args.ambient_offset)
                               if v_radial is not None else np.nan)
    summary['cone_mass'] = float(np.trapezoid(mass_per_radius(rho, frame.radius, frame.latitude, frame.longitude,
                                                              cone), frame.radius))
    return mass_per_radius(rho, frame.radius, frame.latitude, frame.longitude, cone), maps, summary


def kinematics(times_s, records, drag_coefficient):
    """Speeds, drag fits and the physical drag parameter of one density series."""
    table = pd.DataFrame(records)
    result = {'table': table, 'speed_fit': {}, 'speed': {}}
    for track in TRACKS:
        result['speed_fit'][track] = fit_speed(times_s, table[track].values)
        result['speed'][track] = finite_difference_speed(times_s, table[track].values)
    front = table['front_apex'].values
    result['dbm_free'] = fit_dbm(times_s, front)
    wind = np.nanmedian(table['ambient_wind'].values) if np.isfinite(table['ambient_wind'].values).any() else None
    result['dbm_fixed_wind'] = fit_dbm(times_s, front, w_kms=wind) if wind is not None else None
    cross_section = table['front_solid_angle'].values * front ** 2
    result['gamma_physical'] = physical_drag_parameter(table['ambient_density'].values, cross_section,
                                                       table['excess_mass'].values, drag_coefficient)
    return result


def plot_mass_distance_time(times, radius, mass_r, results, out_file):
    names = [n for n in ('true', 'pred') if n in mass_r]
    n_panels = len(names) + (1 if len(names) == 2 else 0)
    fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True, constrained_layout=True)
    axs = np.atleast_1d(axs)
    finite = np.concatenate([mass_r[n][np.isfinite(mass_r[n])] for n in names])
    norm = LogNorm(vmin=np.percentile(finite[finite > 0], 1), vmax=finite.max())
    for ax, name in zip(axs, names):
        im = ax.pcolormesh(times, radius, mass_r[name].T, norm=norm, cmap='inferno', shading='nearest')
        ax.set_title(LABELS[name])
        ax.set_xlabel('Time [UTC]')
        for other in names:
            table = results[other]['table']
            ax.plot(times, table['front_apex'], color=COLORS[other], ls='-', lw=1,
                    label=f'{LABELS[other]} front')
            ax.plot(times, table['ejecta_median'], color=COLORS[other], ls='--', lw=1,
                    label=f'{LABELS[other]} ejecta')
    fig.colorbar(im, ax=list(axs[:len(names)]), label=r'$dM/dr$ [electrons R$_\odot^{-1}$]')
    axs[0].set_ylabel(r'Radius [R$_\odot$]')
    axs[0].legend(fontsize=7, loc='upper left')
    if len(names) == 2:
        ratio = mass_r['pred'] / mass_r['true']
        im = axs[-1].pcolormesh(times, radius, ratio.T, norm=LogNorm(vmin=0.5, vmax=2), cmap='RdBu_r',
                                shading='nearest')
        axs[-1].set_title('SuNeRF / ground truth')
        axs[-1].set_xlabel('Time [UTC]')
        fig.colorbar(im, ax=axs[-1], label='mass ratio')
    for ax in axs:
        ax.tick_params(axis='x', labelrotation=30)
    fig.savefig(out_file, dpi=300, transparent=True)
    plt.close(fig)


def plot_kinematics(times, results, out_file):
    tracks = [('front_apex', 'Front (apex)'), ('front_median', 'Front (median)'), ('cavity_median', 'Cavity'),
              ('ejecta_median', 'Ejecta')]
    fig, axs = plt.subplots(2, len(tracks), figsize=(4 * len(tracks), 6.5), sharex=True)
    for col, (track, title) in enumerate(tracks):
        for name, result in results.items():
            table = result['table']
            axs[0, col].plot(times, table[track], '-o', ms=3, color=COLORS[name],
                             label=f"{LABELS[name]} ({result['speed_fit'][track]:.0f} km/s)")
            axs[1, col].plot(times, result['speed'][track], '-o', ms=3, color=COLORS[name], label=LABELS[name])
        axs[0, col].set_title(title)
        axs[0, col].legend(fontsize=8)
        axs[1, col].set_xlabel('Time [UTC]')
    axs[0, 0].set_ylabel(r'Radius [R$_\odot$]')
    axs[1, 0].set_ylabel('Speed [km/s]')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, transparent=True)
    plt.close(fig)


def plot_drag(times, results, out_file):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for name, result in results.items():
        table = result['table']
        axs[0].plot(times, table['front_apex'], 'o', ms=3, color=COLORS[name], label=f'{LABELS[name]} front')
        fit = result['dbm_free']
        axs[0].plot(times, fit['radius_fit'], '-', color=COLORS[name],
                    label=fr"DBM: $\gamma$={fit['gamma']:.2e} km$^{{-1}}$, w={fit['w']:.0f} km/s")
        if result['dbm_fixed_wind'] is not None:
            fit = result['dbm_fixed_wind']
            axs[0].plot(times, fit['radius_fit'], ':', color=COLORS[name],
                        label=fr"DBM, w fixed {fit['w']:.0f} km/s: $\gamma$={fit['gamma']:.2e} km$^{{-1}}$")
        axs[1].plot(times, result['gamma_physical'], '-o', ms=3, color=COLORS[name],
                    label=fr"{LABELS[name]}: $\rho_{{sw}} A / M$")
        axs[2].plot(times, table['ambient_density'], '-o', ms=3, color=COLORS[name], label=LABELS[name])
    axs[0].set_ylabel(r'Front radius [R$_\odot$]')
    axs[0].legend(fontsize=7)
    axs[1].set_yscale('log')
    axs[1].set_ylabel(r'$\gamma$ [km$^{-1}$]')
    axs[1].legend(fontsize=8)
    axs[2].set_yscale('log')
    axs[2].set_ylabel(r'Ambient $n_e$ ahead of front [cm$^{-3}$]')
    axs[2].legend(fontsize=8)
    if 'pred' in results:
        ax = axs[2].twinx()
        ax.plot(times, results['pred']['table']['ambient_wind'], '-s', ms=3, color='C0')
        ax.set_ylabel('SuNeRF ambient wind [km/s]', color='C0')
    for ax in axs:
        ax.set_xlabel('Time [UTC]')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=300, transparent=True)
    plt.close(fig)


def write_summary(results, out_path):
    summary = {}
    for name, result in results.items():
        table = result['table']
        table.to_csv(os.path.join(out_path, f'structure_{name}.csv'), index=False)
        gamma_phys = result['gamma_physical']
        entry = {
            'speed_fit_kms': {k: float(v) for k, v in result['speed_fit'].items()},
            'dbm_free': {k: float(v) for k, v in result['dbm_free'].items() if k != 'radius_fit'},
            'gamma_physical_median_per_km': float(np.nanmedian(gamma_phys)) if np.isfinite(gamma_phys).any() else None,
            'excess_mass_median_electrons': float(np.nanmedian(table['excess_mass'])),
            'cone_mass_median_electrons': float(np.nanmedian(table['cone_mass'])),
            'detection_fraction_mean': float(np.nanmean(table['detection_fraction'])),
        }
        if result['dbm_fixed_wind'] is not None:
            entry['dbm_fixed_wind'] = {k: float(v) for k, v in result['dbm_fixed_wind'].items() if k != 'radius_fit'}
        summary[name] = entry
    if {'true', 'pred'} <= set(results):
        t, p = results['true']['table'], results['pred']['table']
        summary['comparison'] = {
            track: {'mae_rsun': float(np.nanmean(np.abs(t[track] - p[track])))} for track in TRACKS
        }
        summary['comparison']['excess_mass_relative_error'] = float(
            np.nanmean(np.abs(p['excess_mass'] - t['excess_mass']) / t['excess_mass']))
        summary['comparison']['cone_mass_relative_error'] = float(
            np.nanmean(np.abs(p['cone_mass'] - t['cone_mass']) / t['cone_mass']))
    with open(os.path.join(out_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else str(x))
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    out_path = args.out_path or os.path.join(os.path.dirname(args.sunerf_path), 'mass_transport')
    os.makedirs(out_path, exist_ok=True)

    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device, trusted=args.trusted)
    times, radius = [], None
    mass_r = {'true': [], 'pred': []}
    records = {'true': [], 'pred': []}
    front_maps = {'true': [], 'pred': []}

    for frame in tqdm(iterate_frames(args), desc='frames'):
        frame = frame.crop(args.min_radius, args.max_radius, args.min_latitude, args.max_latitude,
                           args.min_longitude, args.max_longitude, stride=args.stride)
        if radius is None:
            radius = frame.radius
            cone = cone_mask(frame.latitude, frame.longitude, np.deg2rad(args.target_latitude),
                             np.deg2rad(args.target_longitude), np.deg2rad(args.cone_half_angle))
            if cone.sum() == 0:
                raise ValueError('The cone contains no grid rays; check --target_* and the sector limits.')
        times.append(frame.time)

        # ground-truth grids are Carrington; query the model in its own reference frame
        query_points = model_query_points(loader, frame)
        output = loader.load_coords(query_points, batch_size=args.batch_size)  # rho in cm^-3, v in km/s
        rho_pred = output['rho'][..., 0]
        if len(times) == 1 and np.isfinite(frame.rho).any():
            ratio = np.nanmedian(rho_pred) / np.nanmedian(frame.rho)
            print(f'Median density ratio SuNeRF / ground truth in the first frame: {ratio:.3g}')
            if not 0.1 < ratio < 10:
                print('WARNING: density of model and ground truth disagree; check the units and the region.')
        r_hat = query_points[..., :3] / np.linalg.norm(query_points[..., :3], axis=-1, keepdims=True)
        v_radial = (output['v'] * r_hat).sum(-1)

        mass, maps, summary = evaluate_frame(rho_pred, v_radial, frame, cone, args)
        mass_r['pred'].append(mass)
        records['pred'].append(summary)
        front_maps['pred'].append(maps['front'])
        if np.isfinite(frame.rho).any():
            mass, maps, summary = evaluate_frame(frame.rho, None, frame, cone, args)
            mass_r['true'].append(mass)
            records['true'].append(summary)
            front_maps['true'].append(maps['front'])

    times = np.array(times)
    times_s = np.array([(t - times[0]).total_seconds() for t in times])
    mass_r = {k: np.array(v) for k, v in mass_r.items() if v}
    results = {k: kinematics(times_s, v, args.drag_coefficient) for k, v in records.items() if v}

    plot_mass_distance_time(times, radius, mass_r, results, os.path.join(out_path, 'mass_distance_time.png'))
    plot_kinematics(times, results, os.path.join(out_path, 'front_kinematics.png'))
    plot_drag(times, results, os.path.join(out_path, 'drag.png'))
    summary = write_summary(results, out_path)
    np.savez(os.path.join(out_path, 'mass_transport.npz'), times=times, radius=radius,
             **{f'mass_per_radius_{k}': v for k, v in mass_r.items()},
             **{f'front_map_{k}': np.array(v) for k, v in front_maps.items() if v},
             **{f'gamma_physical_{k}': v['gamma_physical'] for k, v in results.items()})
    print(json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
