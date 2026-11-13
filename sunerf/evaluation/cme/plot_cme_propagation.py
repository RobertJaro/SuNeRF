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
frame the core and the leading edge move at most at ``MAX_SPEED_KMS``.  Output: ``--out_path``.

Example::

    python -m sunerf.evaluation.cme.plot_cme_propagation --export_path /path/export \
        --latitude -1 --longitude 268
"""
import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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
# largest speed of the core and the leading edge between two tracked frames [km/s]
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
    """Fractional index between ``start`` and ``stop`` (the maximum) where ``values`` crosses half of
    ``values[stop]``, searched from ``stop`` toward ``start``; behind the maximum ``start`` when it does
    not cross, ahead of it NaN."""
    half = 0.5 * values[stop]
    step = 1 if start > stop else -1
    for index in range(stop + step, start + step, step):
        if values[index] < half:
            return index - step * (half - values[index]) / (values[index - step] - values[index])
    return float(start) if step < 0 else np.nan


def core_tracks(density, ambient, radius, seconds):
    """Rear, core and leading edge [R_sun] of the CME along one ray, ``[time, 3]``, from the density and
    the ambient ``[time, r]`` at the times ``seconds``.

    The core is first detected beyond ``TRACK_MIN_RADIUS`` where the maximum of the excess exceeds
    ``CORE_CONTRAST`` times the ambient; from there it is followed forward (outward) and backward in
    time (inward, down to the inner boundary of the grid) as long as its maximum lies above the
    ambient, inside the grid, and the excess falls to half its maximum ahead of it.  From frame to
    frame the core is searched, and the leading edge accepted, only within the distance covered at
    ``MAX_SPEED_KMS`` since the last tracked frame, so that the track neither switches to a structure
    farther out nor jumps across a secondary maximum; the track ends after two frames without a
    match.  NaN elsewhere."""
    excess = (density - ambient) * radius ** 2
    tracks = np.full((len(excess), 3), np.nan)
    start = int(np.searchsorted(radius, TRACK_MIN_RADIUS))
    index = np.arange(radius.size)

    def locate(t, lower, upper, contrast, previous):
        """Core index of frame ``t`` between ``lower`` and ``upper`` and within reach of the tracked frame
        ``previous`` (None before the first detection), or None; fills the tracks."""
        values = excess[t]
        if previous is not None:
            reach = MAX_SPEED_KMS * abs(seconds[t] - seconds[previous]) / KM_PER_RSUN
            if t > previous:
                upper = min(upper, int(np.searchsorted(radius, tracks[previous, 1] + reach, side='right')) - 1)
            else:
                lower = max(lower, int(np.searchsorted(radius, tracks[previous, 1] - reach)))
        if upper < lower:
            return None
        peak = lower + int(np.argmax(values[lower:upper + 1]))
        if peak == 0 or density[t, peak] < contrast * ambient[t, peak]:
            return None
        front = half_maximum(values, radius.size - 1, peak)
        if not np.isfinite(front):
            return None
        front = np.interp(front, index, radius)
        if previous is not None and abs(front - tracks[previous, 2]) > reach:
            return None
        tracks[t] = [np.interp(half_maximum(values, 0, peak), index, radius), radius[peak], front]
        return peak

    first = previous = None
    lower = start
    for t in range(len(excess)):
        # a single frame may be skipped; the reach grows with the gap and would admit other structures
        if previous is not None and t - previous > 2:
            break
        peak = locate(t, lower, radius.size - 1, CORE_CONTRAST if first is None else 1.0, previous)
        if peak is not None:
            first = t if first is None else first
            previous, lower = t, peak
    if first is not None:
        previous, upper = first, int(np.argmin(np.abs(radius - tracks[first, 1])))
        for t in range(first - 1, -1, -1):
            upper = locate(t, 0, upper, 1.0, previous)
            if upper is None:
                break
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
