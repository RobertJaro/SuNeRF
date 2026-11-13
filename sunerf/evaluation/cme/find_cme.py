"""Identify the CMEs of a density export and their propagation directions.

The export of ``export_density_grid`` holds the density and the quasi-static background of every
frame.  A CME is a connected region in time and space (time, radius, latitude, longitude) in
which the compensated excess ``r^2 (n_e - n_background)`` exceeds ``--threshold`` times the
typical compensated background of its radial shell.  Regions that are connected in space but
separated in time, or the other way round, are different CMEs.

For every CME the excess mass, the mass-weighted direction (per frame and overall), the
mass-weighted radius, the speed of that radius and the angular half width are written to::

    <export>/cme_catalog.txt    one line per CME, sorted by the peak excess mass (first: primary CME)
    <export>/cme_catalog.npz    tracks of every CME: times, mass [g], radius [R_sun],
                                latitude and longitude [rad, HCI] per frame
    <export>/cme_catalog.jpg    sky map and distance-time diagram of the excess mass with the CMEs

Example::

    python -m sunerf.evaluation.cme.find_cme --export_path /path/export
"""
import argparse
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage

from sunerf.evaluation.cme.density_series import list_series_files
from sunerf.evaluation.cme.export_layout import background_file, coordinates_file, density_file

CM_PER_RSUN = 6.957e10
KM_PER_RSUN = 6.957e5
# mass per electron of a plasma with 10 % helium, the convention of coronagraph CME masses
GRAMS_PER_ELECTRON = 1.97e-24
# resolution of the identification: the export is subsampled to about these steps
RADIAL_STEP = 1.0
ANGULAR_STEP = np.deg2rad(4.0)
# a CME lasts at least this number of frames and holds at least this fraction of the largest mass
MIN_FRAMES = 3
MIN_MASS_FRACTION = 0.01


def load_excess(export_path):
    """Excess density [cm^-3] of all frames on the subsampled grid, with the grid and the times."""
    names = [Path(file).stem for file in list_series_files(density_file(export_path, '*'))]
    coordinates = np.load(coordinates_file(export_path, names[0]))
    radius, latitude, longitude = coordinates['radius'], coordinates['latitude'], coordinates['longitude']
    step = (slice(None, None, max(1, round(RADIAL_STEP / (radius[1] - radius[0])))),
            slice(None, None, max(1, round(ANGULAR_STEP / (latitude[1] - latitude[0])))),
            slice(None, None, max(1, round(ANGULAR_STEP / (longitude[1] - longitude[0])))))
    excess, background, times = [], [], []
    for name in names:
        rho = np.load(density_file(export_path, name), mmap_mode='r')[step]
        frame_background = np.load(background_file(export_path, name), mmap_mode='r')[step]
        excess.append(rho - frame_background)
        background.append(frame_background)
        times.append(np.load(coordinates_file(export_path, name))['time'])
    return (np.stack(excess), np.stack(background), radius[step[0]], latitude[step[1]], longitude[step[2]],
            np.array(times, dtype='datetime64[s]'))


def label_regions(mask):
    """Connected regions of ``mask[time, radius, latitude, longitude]``, periodic in longitude."""
    labels, n_labels = ndimage.label(mask)
    # regions that touch across the longitude seam are the same region
    parent = np.arange(n_labels + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first, last = labels[..., 0], labels[..., -1]
    for a, b in np.unique(np.stack([first[(first > 0) & (last > 0)], last[(first > 0) & (last > 0)]], -1), axis=0):
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n_labels + 1)])[labels]


def describe_region(region, mass, radius, latitude, longitude, times):
    """Track and summary of one region; ``mass`` is the excess mass of every cell [g]."""
    frames = np.flatnonzero(region.any(axis=(1, 2, 3)))
    r, lat, lon = np.meshgrid(radius, latitude, longitude, indexing='ij')
    unit = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    track = {key: np.full(len(times), np.nan) for key in ('mass', 'radius', 'latitude', 'longitude')}
    moment = np.zeros(3)
    for t in frames:
        cell_mass = np.where(region[t], mass[t], 0.0)
        vector = np.sum(unit * cell_mass, axis=(1, 2, 3))
        moment += vector
        track['mass'][t] = cell_mass.sum()
        track['radius'][t] = np.sum(r * cell_mass) / cell_mass.sum()
        track['latitude'][t] = np.arctan2(vector[2], np.hypot(vector[0], vector[1]))
        track['longitude'][t] = np.arctan2(vector[1], vector[0]) % (2 * np.pi)
    direction = moment / np.linalg.norm(moment)
    # mass-weighted rms angle to the direction
    total_mass = np.where(region, mass, 0.0).sum(axis=0)
    angle = np.arccos(np.clip(np.tensordot(direction, unit, axes=1), -1, 1))
    seconds = (times[frames] - times[frames[0]]).astype(float)
    speed = np.polyfit(seconds, track['radius'][frames], 1)[0] * KM_PER_RSUN if len(frames) > 1 else np.nan
    return {
        'region': region, 'track': track, 'frames': frames, 'peak_mass': np.nanmax(track['mass']),
        'latitude': np.arctan2(direction[2], np.hypot(direction[0], direction[1])),
        'longitude': np.arctan2(direction[1], direction[0]) % (2 * np.pi),
        'half_width': np.sqrt(np.sum(total_mass * angle ** 2) / total_mass.sum()),
        'speed': speed, 'radius_range': (np.nanmin(track['radius']), np.nanmax(track['radius'])),
    }


def find_cmes(excess, background, radius, latitude, longitude, times, threshold):
    """CMEs sorted by their peak excess mass (with their region masks), and the excess mass of every cell [g]."""
    r2 = radius[None, :, None, None] ** 2
    shell_background = np.median(background * r2, axis=(0, 2, 3))
    relative = excess * r2 / shell_background[None, :, None, None]
    volume = (radius ** 2 * np.gradient(radius))[:, None, None] * np.cos(latitude)[None, :, None] \
             * np.gradient(latitude)[None, :, None] * np.gradient(longitude)[None, None, :] * CM_PER_RSUN ** 3
    mass = excess * volume[None] * GRAMS_PER_ELECTRON

    labels = label_regions(relative > threshold)
    cmes = []
    for label in np.unique(labels[labels > 0]):
        region = labels == label
        if region.any(axis=(1, 2, 3)).sum() >= MIN_FRAMES:
            cmes.append(describe_region(region, mass, radius, latitude, longitude, times))
    if not cmes:
        raise ValueError(f'No region exceeds the threshold {threshold} for {MIN_FRAMES} frames.')
    cmes.sort(key=lambda cme: -cme['peak_mass'])
    return [cme for cme in cmes if cme['peak_mass'] >= MIN_MASS_FRACTION * cmes[0]['peak_mass']], mass


def write_catalog(export_path, cmes, times, threshold):
    lines = [
        f'# CMEs of the density export (find_cme, threshold {threshold}), sorted by the peak excess mass.',
        '# latitude, longitude: mass-weighted direction in the inertial (HCI) frame [deg];',
        '# half_width: mass-weighted rms angle to the direction [deg]; speed: of the mass-weighted radius [km/s];',
        '# r_start, r_end: mass-weighted radius in the first and the last frame [R_sun].',
        '# id start_utc end_utc latitude longitude half_width peak_mass_g speed r_start r_end',
    ]
    tracks = {'times': times}
    for index, cme in enumerate(cmes):
        frames, track = cme['frames'], cme['track']
        lines.append(f'{index} {times[frames[0]]} {times[frames[-1]]} {np.rad2deg(cme["latitude"]):.1f} '
                     f'{np.rad2deg(cme["longitude"]):.1f} {np.rad2deg(cme["half_width"]):.1f} '
                     f'{cme["peak_mass"]:.3e} {cme["speed"]:.0f} {track["radius"][frames[0]]:.1f} '
                     f'{track["radius"][frames[-1]]:.1f}')
        for key, values in track.items():
            tracks[f'cme{index}_{key}'] = values
    with open(os.path.join(export_path, 'cme_catalog.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    np.savez(os.path.join(export_path, 'cme_catalog.npz'), **tracks)
    print('\n'.join(lines))


def plot_catalog(export_path, cmes, mass, radius, latitude, longitude, times):
    """Sky map (maximum over time of the excess mass per direction) and distance-time diagram."""
    fig, (ax_sky, ax_time) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    positive = np.clip(mass, 0, None)
    sky = positive.sum(axis=1).max(axis=0)
    norm = LogNorm(vmin=sky.max() * 1e-3, vmax=sky.max())
    im = ax_sky.pcolormesh(np.rad2deg(longitude), np.rad2deg(latitude), sky, norm=norm, cmap='magma', shading='auto')
    fig.colorbar(im, ax=ax_sky, label='Excess mass per cell column [g]')
    ax_sky.set_xlabel('HCI longitude [deg]')
    ax_sky.set_ylabel('Latitude [deg]')
    ax_sky.set_title('Maximum over time')

    hours = (times - times[0]).astype(float) / 3600.0
    per_radius = positive.sum(axis=(2, 3)) / np.gradient(radius)[None]
    norm = LogNorm(vmin=per_radius.max() * 1e-3, vmax=per_radius.max())
    im = ax_time.pcolormesh(hours, radius, per_radius.T, norm=norm, cmap='magma', shading='auto')
    fig.colorbar(im, ax=ax_time, label=r'Excess mass [g R$_\odot^{-1}$]')
    ax_time.set_xlabel(f'Hours since {times[0]}')
    ax_time.set_ylabel(r'Radius [R$_\odot$]')
    ax_time.set_title('All directions')

    for index, cme in enumerate(cmes):
        color = f'C{index % 10}'
        ax_sky.plot(np.rad2deg(cme['longitude']), np.rad2deg(cme['latitude']), marker='x', color=color)
        ax_sky.annotate(str(index), (np.rad2deg(cme['longitude']), np.rad2deg(cme['latitude'])),
                        xytext=(4, 4), textcoords='offset points', color=color)
        ax_time.plot(hours, cme['track']['radius'], color=color, lw=1.5, label=f'CME {index}')
    ax_time.legend(loc='upper left', fontsize=8)
    fig.savefig(os.path.join(export_path, 'cme_catalog.jpg'), dpi=200)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--threshold', type=float, default=10.0,
                        help='Compensated excess in units of the typical compensated background of the shell')
    args = parser.parse_args(argv)

    excess, background, radius, latitude, longitude, times = load_excess(args.export_path)
    cmes, mass = find_cmes(excess, background, radius, latitude, longitude, times, args.threshold)
    write_catalog(args.export_path, cmes, times, args.threshold)
    plot_catalog(args.export_path, cmes, mass, radius, latitude, longitude, times)


if __name__ == '__main__':
    main()
