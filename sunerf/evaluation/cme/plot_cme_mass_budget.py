"""Kinematics and mass budget of the dense core of the primary CME.

Reads the density of the spherical frames of ``export_density_grid``.  Along the CME axis (the rays
within ``AXIS_HALF_ANGLE`` of ``--latitude``, ``--longitude``, inertial HCI frame) the core, the
leading edge and the rear are tracked with ``core_tracks`` of ``plot_cme_propagation`` against the
ambient corona (the first frame, before the CME, corotated with the Sun to every time).  The tracks
of the averaged profile are the main result; the 16-84 percentiles of the tracks of the single rays
give their spread.  The figure holds

* the distance-time map of the compensated density ``r^2 n_e`` along the axis with the tracks;
* the speed and the acceleration of the leading edge, the core and the rear (quadratic fits in a
  sliding window of ``KINEMATICS_WINDOW_HOURS``);
* the mass between the rear (at least ``TRACK_MIN_RADIUS``) and the leading edge within the sector
  of ``--half_angle`` against the leading edge, and the ambient mass of the same volume.  The same
  volume mirrored to the northern
  latitude ``-latitude`` gives the noise floor: the band of its difference between the
  reconstruction and the ambient around the ambient mass.

No background is subtracted from the maps or the masses.  Output: ``--out_path``.

Example::

    python -m sunerf.evaluation.cme.plot_cme_mass_budget --export_path /path/export \
        --out_path /path/cme_mass_budget.jpg --latitude -30 --longitude 5
"""
import argparse
import os
import warnings

import numpy as np
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from sunerf.evaluation.cme.grid_series import GRAMS_PER_ELECTRON, cell_volume, load_series
from sunerf.evaluation.cme.mass_transport import local_kinematics
from sunerf.evaluation.cme.plot_cme_propagation import TRACK_MIN_RADIUS, ambient_on_rays, core_tracks, sector

# half angle of the cone around the CME axis of the tracking [deg]
AXIS_HALF_ANGLE = 10.0
# window of the quadratic fits of the speed and acceleration [h]
KINEMATICS_WINDOW_HOURS = 6.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--out_path', required=True, help='Output figure (jpg)')
    parser.add_argument('--latitude', type=float, required=True, help='CME direction, latitude [deg]')
    parser.add_argument('--longitude', type=float, required=True, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--half_angle', type=float, default=30.0,
                        help='Half width of the mass sector in latitude and longitude [deg]')
    args = parser.parse_args(argv)

    series = load_series(args.export_path, keys=('density',))
    radius, latitude, longitude, times = series['radius'], series['latitude'], series['longitude'], series['times']
    seconds = (times - times[0]).astype(float)
    hours = seconds / 3600.0
    n_lon = longitude.size

    def rays(direction, half_angle):
        in_latitude, in_longitude = sector(latitude, longitude, np.deg2rad(direction), np.deg2rad(half_angle))
        selected = np.flatnonzero((in_latitude[:, None] & in_longitude[None, :]).ravel())
        return selected // n_lon, selected % n_lon

    axis_rays = rays((args.latitude, args.longitude), AXIS_HALF_ANGLE)
    sectors = {'cme': rays((args.latitude, args.longitude), args.half_angle),
               'control': rays((-args.latitude, args.longitude), args.half_angle)}

    # density and the ambient (first frame corotated to every time) on the axis rays and in the sectors
    density = np.nan_to_num(series.pop('density'), copy=False)
    axis_ambient, *ambients = ambient_on_rays(density[0], longitude, seconds, [axis_rays, *sectors.values()])
    sector_ambient = dict(zip(sectors, ambients))  # [time, ray, r]
    axis_density = density[:, axis_rays[0], axis_rays[1]]

    # tracks of the averaged profile and of the single rays
    tracks = core_tracks(axis_density.mean(axis=1), axis_ambient.mean(axis=1), radius, seconds)
    ray_tracks = np.stack([core_tracks(axis_density[:, ray], axis_ambient[:, ray], radius, seconds)
                           for ray in range(axis_density.shape[1])], axis=1)  # [time, ray, 3]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        spread = np.nanpercentile(ray_tracks, (16, 84), axis=1)  # [2, time, 3]
    rear, core, front = tracks.T
    detected = np.isfinite(front)
    print(f'Frames with a core on the axis: {detected.sum()} of {len(times)}')

    # kinematics of the rear, the core and the leading edge: sliding quadratic fits
    window = max(KINEMATICS_WINDOW_HOURS, 5.0 * np.median(np.diff(hours))) * 3600.0
    kinematics = [local_kinematics(seconds, track, window) for track in tracks.T]  # (speed, acceleration)

    # masses between the rear and the leading edge in the CME sector and the mirrored control sector
    volume = cell_volume(radius, latitude, longitude) * GRAMS_PER_ELECTRON
    inner = np.maximum(rear, TRACK_MIN_RADIUS)
    between = (radius[None, :] >= inner[:, None]) & (radius[None, :] <= front[:, None])  # [time, r]
    masses = {}
    for key, selection in sectors.items():
        cells = volume[selection]
        total = (density[:, selection[0], selection[1]] * cells[None]).sum(axis=1)  # [time, r]
        ambient = (sector_ambient[key] * cells[None]).sum(axis=1)
        masses[key] = (np.where(between, total, 0.0).sum(axis=1), np.where(between, ambient, 0.0).sum(axis=1))
    noise = np.abs(masses['control'][0] - masses['control'][1])

    fig, axes = plt.subplots(4, 1, figsize=(9, 14), constrained_layout=True,
                             gridspec_kw={'height_ratios': (1.3, 0.7, 0.7, 1.0)})
    ax = axes[0]
    compensated = axis_density.mean(axis=1) * radius[None] ** 2
    positive = compensated[compensated > 0]
    im = ax.pcolormesh(hours, radius, compensated.T, cmap='viridis',
                       norm=LogNorm(*np.percentile(positive, (5.0, 95.0))), shading='nearest')
    outline = [patheffects.withStroke(linewidth=2.5, foreground='black')]
    for column, color, label in ((2, 'red', 'Leading edge (half maximum)'), (1, 'white', 'Core'),
                                 (0, 'white', 'Rear (half maximum)')):
        ax.fill_between(hours, spread[0, :, column], spread[1, :, column], color=color, alpha=0.25, lw=0)
        ax.plot(hours, tracks[:, column], color=color, lw=1.2, ls='--' if column == 0 else '-', label=label,
                path_effects=outline)
    ax.axhline(TRACK_MIN_RADIUS, color='white', lw=0.8, ls=':')
    fig.colorbar(im, ax=ax, label=r'$r^2 n_e$ [cm$^{-3}$ R$_\odot^2$]', pad=0.01)
    ax.set_ylabel(r'Radius [R$_\odot$]')
    ax.set_title(f'Along the CME axis (±{AXIS_HALF_ANGLE:.0f}°): compensated density')
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(labelbottom=False)

    for row, label in ((0, r'Speed [km s$^{-1}$]'), (1, r'Acceleration [m s$^{-2}$]')):
        ax = axes[row + 1]
        for column, color, style, name in ((2, 'red', '-', 'Leading edge'), (1, 'black', '-', 'Core'),
                                           (0, 'tab:blue', '--', 'Rear')):
            ax.plot(hours, kinematics[column][row], color=color, lw=1.4, ls=style, label=name)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.sharex(axes[0])
    axes[1].legend(loc='lower right', fontsize=8)
    axes[1].tick_params(labelbottom=False)
    axes[2].axhline(0, color='0.5', lw=0.8)
    axes[2].set_xlabel(f'Hours since {times[0]}'.replace('T', ' '))

    ax = axes[3]
    total, ambient = masses['cme']
    ax.fill_between(front[detected], (ambient - noise)[detected], (ambient + noise)[detected], color='tab:blue',
                    alpha=0.2, lw=0, label=f'Noise floor (same volume at latitude {-args.latitude:.0f}°)')
    ax.plot(front[detected], total[detected], color='black', lw=1.6, label='Total mass (reconstruction)')
    ax.plot(front[detected], ambient[detected], color='tab:blue', lw=1.6, ls='--',
            label=f'Ambient mass (first frame {str(times[0])[:16].replace("T", " ")}, corotated)')
    ax.set_xlabel(r'Leading edge on the axis [R$_\odot$]')
    ax.set_ylabel('Mass [g]')
    ax.set_title(f'Mass between the rear (at least {TRACK_MIN_RADIUS:.0f} R$_\\odot$) and the leading edge, '
                 f'sector of ±{args.half_angle:.0f}°')
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    fig.suptitle(f'CME direction: latitude {args.latitude:.0f}°, HCI longitude {args.longitude:.0f}°', fontsize=12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fig.savefig(args.out_path, dpi=200)
    plt.close(fig)
    print(f'Figure written to {args.out_path}')


if __name__ == '__main__':
    main()
