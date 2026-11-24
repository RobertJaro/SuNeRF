"""Propagation of the primary CME: latitude-radius slices at five longitudes over time.

Reads the spherical frames of ``export_density_grid``.  The CME is defined by its direction
(``--latitude``, ``--longitude`` in the inertial HCI frame) and the sector of ``--half_angle``
around it in latitude.  The figure shows, at ``--n_panels`` times, the latitude-integrated density
of the latitude sector as longitude-radius maps over the full circle (top row, the five slice
longitudes marked by dashed lines)
and latitude-radius slices of the density (not integrated) at five longitudes within
``+-TRACK_HALF_ANGLE`` of the CME longitude.

The tracking of the CME core along a ray (``core_tracks``, used by ``plot_cme_mass_budget`` and
``plot_cme_front``) is defined here: the excess ``r^2 (n_e - n_ambient)`` over the ambient corona
(the first frame corotated with the Sun, ``ambient_on_rays``) defines the core at its maximum,
first detected where the density exceeds ``CORE_CONTRAST`` times the ambient and followed from
there forward and backward in time down to the inner boundary of the grid, and the leading edge and
the rear where the excess falls to half its maximum ahead of and behind the core; from frame to
frame the core and both edges move at most at ``MAX_SPEED_KMS``.  Output: ``--out_path``.

Example::

    python -m sunerf.evaluation.cme.plot_cme_propagation --export_path /path/export \
        --latitude -1 --longitude 268
"""
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

from sunerf.evaluation.cme.export_density_grid import rotate_background
from sunerf.evaluation.cme.grid_series import CM_PER_RSUN, load_series
from sunerf.train.coordinate_transformation import carrington_rotation_rate

# color scales of the maps: percentiles of the shown values
SCALE_PERCENTILES = (5.0, 95.0)
# five tracking longitudes within this half angle of the CME longitude [deg]
TRACK_HALF_ANGLE = 20.0
# the first detection of the CME core and the mass evaluation start at this radius [R_sun]
TRACK_MIN_RADIUS = 10.0
TRACK_OFFSETS = np.linspace(-TRACK_HALF_ANGLE, TRACK_HALF_ANGLE, 5)
# minimum density over the ambient at the core
CORE_CONTRAST = 2.0
# Continuation threshold after the strong initial detection (density / ambient).
FOLLOW_CONTRAST = 1.2
# largest speed of each tracked feature between detections [km/s]
MAX_SPEED_KMS = 1500.0
KM_PER_RSUN = CM_PER_RSUN / 1e5


def ambient_on_rays(first, longitude, seconds, selections):
    """The first frame ``[lat, lon, r]`` corotated to every time ``seconds`` (since the first frame) on the
    rays of every ``(lat_index, lon_index)`` selection: one array ``[time, ray, r]`` per selection."""
    angles = float(carrington_rotation_rate()) * np.asarray(seconds, dtype=float)
    ambient = [[] for _ in selections]
    for angle in angles:
        rotated = rotate_background(first, longitude, angle)
        for values, selection in zip(ambient, selections):
            values.append(rotated[selection])
    return [np.stack(values) for values in ambient]


def half_maximum(values, start, stop):
    """First half-height crossing from the peak toward the boundary, confirmed by two bins."""
    half = 0.5 * values[stop]
    step = 1 if start > stop else -1
    for index in range(stop + step, start, step):
        if not np.isfinite(values[index]):
            break
        if values[index] <= half and values[index + step] <= half and np.isfinite(values[index + step]):
            return index - step * (half - values[index]) / (values[index - step] - values[index])
    return np.nan


def core_tracks(density, ambient, radius, seconds):
    """Track an outward-moving peak and its two independently observed half-height edges.

    Seed above TRACK_MIN_RADIUS at CORE_CONTRAST and follow local peaks at FOLLOW_CONTRAST.
    A single missing core frame can be crossed. Each edge can resume while the core is
    tracked, subject to the speed limit since that edge's own last detection. Missing
    crossings, including profiles truncated at a grid boundary, stay NaN.
    """
    excess = (density - ambient) * radius ** 2
    tracks = np.full((len(excess), 3), np.nan)
    index = np.arange(radius.size)
    # Find peaks on the full profile so a slope truncated by the motion gate cannot become a core.
    peaks = [find_peaks(profile)[0] for profile in excess]

    def locate_core(t, lower, upper, contrast):
        candidates = peaks[t]
        valid = ((radius[candidates] >= lower) & (radius[candidates] <= upper)
                 & (ambient[t, candidates] > 0)
                 & (density[t, candidates] >= contrast * ambient[t, candidates])
                 & np.isfinite(excess[t, candidates]))
        candidates = candidates[valid]
        if candidates.size:
            return candidates[np.argmax(excess[t, candidates])]

    def measure(t, peak, last_edges):
        tracks[t, 1] = radius[peak]
        for column, boundary in ((0, 0), (2, radius.size - 1)):
            crossing = half_maximum(excess[t], boundary, peak)
            if not np.isfinite(crossing):
                continue
            edge = np.interp(crossing, index, radius)
            previous = last_edges[column]
            if previous is not None:
                reach = MAX_SPEED_KMS * abs(seconds[t] - seconds[previous]) / KM_PER_RSUN
                if abs(edge - tracks[previous, column]) > reach:
                    continue
            tracks[t, column] = edge
            last_edges[column] = t

    for seed in range(len(excess)):
        peak = locate_core(seed, TRACK_MIN_RADIUS, radius[-1], CORE_CONTRAST)
        if peak is not None:
            break
    else:
        return tracks

    seed_edges = {0: None, 2: None}
    measure(seed, peak, seed_edges)
    for direction, end in ((1, len(excess)), (-1, -1)):
        previous = seed
        last_edges = seed_edges.copy()
        for t in range(seed + direction, end, direction):
            if abs(t - previous) > 2:
                break
            reach = MAX_SPEED_KMS * (seconds[t] - seconds[previous]) / KM_PER_RSUN
            center = tracks[previous, 1]
            lower, upper = sorted((center, center + reach))
            peak = locate_core(t, lower, upper, FOLLOW_CONTRAST)
            if peak is not None:
                measure(t, peak, last_edges)
                previous = t
    return tracks


def sector(latitude, longitude, direction, half_angle):
    """Latitude and longitude selections of the sector ``|lat - lat0| <= half_angle``, ``|lon - lon0| <= half_angle``."""
    in_latitude = np.abs(latitude - direction[0]) <= half_angle
    in_longitude = np.abs((longitude - direction[1] + np.pi) % (2 * np.pi) - np.pi) <= half_angle
    return in_latitude, in_longitude


def integrate_latitude(density, latitude, radius, in_latitude):
    """Density integrated along the meridional arc of the sector, ``[time, lon, r]`` [cm^-2]."""
    arc = (np.gradient(latitude)[in_latitude, None, None] * radius[None, None, :] * CM_PER_RSUN)
    return (density[:, in_latitude] * arc[None]).sum(axis=1)


def plot_sector(ax, angle, values, radius, center, half_angle, norm, cmap):
    """Polar map over ``angle`` (rad) and radius, restricted to the sector; the full circle if
    ``half_angle`` is None."""
    if half_angle is None:
        # close the map in longitude
        angle = np.append(angle, angle[0] + 2 * np.pi)
        values = np.concatenate([values, values[:1]], axis=0)
        order = np.arange(angle.size)
        # radial labels opposite to the CME
        ax.set_rlabel_position(np.rad2deg(center) + 180)
    else:
        # unwrap the angles around the center so that the sector is contiguous
        angle = center + (angle - center + np.pi) % (2 * np.pi) - np.pi
        order = np.argsort(angle)
        ax.set_thetamin(np.rad2deg(center - half_angle))
        ax.set_thetamax(np.rad2deg(center + half_angle))
    im = ax.pcolormesh(angle[order], radius, values[order].T, norm=norm, cmap=cmap, shading='nearest')
    ax.set_rorigin(0)
    ax.set_rlim(0, radius[-1])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    return im


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--out_path', required=True, help='Output figure (jpg)')
    parser.add_argument('--latitude', type=float, required=True, help='CME direction, latitude [deg]')
    parser.add_argument('--longitude', type=float, required=True, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--half_angle', type=float, default=45.0,
                        help='Half width of the sector in latitude and longitude [deg]')
    parser.add_argument('--n_panels', type=int, default=6, help='Number of sector-map time steps')
    args = parser.parse_args(argv)

    series = load_series(args.export_path, keys=('density',))
    radius, latitude, longitude, times = series['radius'], series['latitude'], series['longitude'], series['times']
    direction = (np.deg2rad(args.latitude), np.deg2rad(args.longitude))
    hours = (times - times[0]).astype(float) / 3600.0
    half_angle = np.deg2rad(args.half_angle)
    in_latitude, _ = sector(latitude, longitude, direction, half_angle)
    density = np.nan_to_num(series.pop('density'), copy=False)
    track_longitudes = direction[1] + np.deg2rad(TRACK_OFFSETS)
    # nearest grid rays of the tracks: the CME latitude at the five longitudes
    lon_index = [int(np.argmin(np.abs((longitude - lon + np.pi) % (2 * np.pi) - np.pi))) for lon in track_longitudes]

    n_rows = len(lon_index) + 1
    fig = plt.figure(figsize=(2.6 * args.n_panels + 0.6, 2.3 * n_rows + 0.8), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, args.n_panels + 1, width_ratios=[1.0] * args.n_panels + [0.06])
    panel_frames = np.unique(np.linspace(0, len(times) - 1, args.n_panels).round().astype(int))

    # top row: longitude-radius maps of the latitude-integrated density of the sector
    integrated = integrate_latitude(density, latitude, radius, in_latitude)  # [time, lon, r]
    norm = LogNorm(*np.percentile(integrated[integrated > 0], SCALE_PERCENTILES))
    for column, t in enumerate(panel_frames):
        ax = fig.add_subplot(gs[0, column], projection='polar')
        im = plot_sector(ax, longitude, np.maximum(integrated[t], norm.vmin), radius, direction[1], None,
                         norm, 'plasma')
        for index in lon_index:
            ax.plot([longitude[index]] * 2, [0, radius[-1]], color='white', linestyle='--', linewidth=0.8)
        ax.set_title(f'{times[t]}'.replace('T', ' ')[:16], fontsize=9)
        if column == 0:
            ax.set_ylabel(f'latitude {np.rad2deg(direction[0]) - args.half_angle:.0f}° to '
                          f'{np.rad2deg(direction[0]) + args.half_angle:.0f}°', fontsize=9, labelpad=28)
    fig.colorbar(im, cax=fig.add_subplot(gs[0, -1]), label=r'$\int n_e\, r\, d\theta$ [cm$^{-2}$]')

    # latitude-radius slices of the density at the five longitudes
    slices = density[:, :, lon_index]  # [time, lat, slice, r]
    shown = slices[:, in_latitude]
    norm = LogNorm(*np.percentile(shown[shown > 0], SCALE_PERCENTILES))
    for row, index in enumerate(lon_index, start=1):
        for column, t in enumerate(panel_frames):
            ax = fig.add_subplot(gs[row, column], projection='polar')
            im = plot_sector(ax, latitude, np.maximum(slices[t, :, row - 1], norm.vmin), radius, direction[0],
                             half_angle, norm, 'plasma')
            if column == 0:
                ax.set_ylabel(f'longitude {np.rad2deg(longitude[index]):.0f}°', fontsize=9, labelpad=28)
    fig.colorbar(im, cax=fig.add_subplot(gs[1:, -1]), label=r'$n_e$ [cm$^{-3}$]')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fig.savefig(args.out_path, dpi=200)
    plt.close(fig)
    print(f'Figure written to {args.out_path}')


if __name__ == '__main__':
    main()
