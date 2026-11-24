"""Trace the half-height CME leading edge about the moving centroid used in the shell cuts."""
import argparse
import csv
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from sunerf.evaluation.cme.export_density_grid import corotation_angle, rotate_background
from sunerf.evaluation.cme.plot_cme_cuts import cme_shell, frame_centroid, load_centroid_reference, unit
from sunerf.evaluation.cme.plot_cme_mass_budget import AXIS_HALF_ANGLE, _axis_track_statistics
from sunerf.evaluation.cme.plot_cme_propagation import sector


def patch_indices(lat, lon, center):
    lat_sel, lon_sel = sector(lat, lon, np.radians(center), np.radians(AXIS_HALF_ANGLE))
    return np.where(lat_sel[:, None] & lon_sel)


def front_track(profiles, radius, seconds):
    """Use the actual number of angular rays in each patch for the majority requirement."""
    counts = np.array([len(values) for values, _ in profiles])
    shape = (len(seconds), max(1, counts.max()), len(radius))
    density = np.full(shape, np.nan)
    ambient = np.full(shape, np.nan)
    for t, (values, reference) in enumerate(profiles):
        density[t, :counts[t]] = values
        ambient[t, :counts[t]] = reference
    tracks, spread, _ = _axis_track_statistics(density, ambient, radius, seconds, counts)
    return tracks[:, 2], spread[:, :, 2], counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export_path', required=True, type=Path)
    parser.add_argument('--out_path', required=True, type=Path, help='Figure path. The CSV uses the same stem.')
    parser.add_argument('--latitude', type=float, required=True, help='Centroid seed latitude, as in the cuts')
    parser.add_argument('--longitude', type=float, required=True, help='Centroid seed HCI longitude, as in the cuts')
    parser.add_argument('--centroid_reference', help='Optional pre-event NPZ used by plot_cme_cuts')
    parser.add_argument('--fixed_axis', nargs=2, type=float, metavar=('LAT', 'LON'), default=(-30, 5),
                        help='Fixed HCI direction for the comparison (default: -30 5)')
    args = parser.parse_args(argv)
    files = sorted(args.export_path.glob('rho*.npz'))
    if not files:
        parser.error(f'No exported frames in {args.export_path}')
    with np.load(files[0]) as frame:
        first = frame['density']
        radius, latitude, longitude = (frame[key] for key in ('radius', 'latitude', 'longitude'))
        start = frame['time'].astype('datetime64[us]').item()
    reference = load_centroid_reference(args.centroid_reference, radius, latitude, longitude)
    lat, lon = np.radians(latitude), np.radians(longitude)
    seed = args.latitude, args.longitude
    fixed_rays = patch_indices(lat, lon, args.fixed_axis)
    fixed, moving, times = [], [], []
    centers = np.full((len(files), 2), np.nan)
    shells = np.full(len(files), np.nan)
    empty = np.empty((0, len(radius)))

    for t, path in enumerate(files):
        with np.load(path) as frame:
            time = frame['time'].astype('datetime64[s]')[()]
            times.append(time)
            density = frame['density']
            center = None
            if t:
                shell = cme_shell(density, radius, latitude, longitude, unit(*seed))
                center = frame_centroid(frame, density, shell, latitude, longitude, seed, reference)
                shells[t] = radius[shell]
        angle = corotation_angle(time.astype('datetime64[us]').item(), start)
        # Both edge fits use the same corotated first-frame baseline as the fixed-axis budget.
        ambient = rotate_background(first, lon, angle)
        fixed.append((density[fixed_rays], ambient[fixed_rays]))
        if center is None:
            moving.append((empty, empty))
        else:
            centers[t] = center
            rays = patch_indices(lat, lon, center)
            moving.append((density[rays], ambient[rays]))

    seconds = (np.array(times) - times[0]) / np.timedelta64(1, 's')
    front, spread, counts = front_track(moving, radius, seconds)
    fixed_front, _, _ = front_track(fixed, radius, seconds)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_utc', 'centroid_lat_deg', 'centroid_lon_deg', 'shell_rsun', 'ray_count',
                         'front_rsun', 'front_p16_rsun', 'front_p84_rsun', 'fixed_front_rsun'])
        writer.writerows(zip(times, *centers.T, shells, counts, front, *spread, fixed_front))

    hours = seconds / 3600.
    fig, ax = plt.subplots(figsize=(7.4, 4.5), layout='constrained')
    ax.fill_between(hours, *spread, color='#187099', alpha=.20, lw=0, label='16th–84th percentile across rays')
    ax.plot(hours, front, '.-', color='#165d80', ms=4, label='Centroid-guided front')
    ax.plot(hours, fixed_front, '--', color='#865234', lw=1.3, label='Fixed-axis front')
    ax.set(xlabel=f'Hours since {str(times[0]).replace("T", " ")} UTC',
           ylabel=r'Leading-edge radius [$R_\odot$]', title='CME leading edge along the shell centroid')
    ax.spines[['right', 'top']].set_visible(False)
    ax.grid(alpha=.17)
    ax.legend(frameon=False)
    fig.savefig(args.out_path, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
