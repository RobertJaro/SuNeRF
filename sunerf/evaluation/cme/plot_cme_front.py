"""Spatially resolved leading edge of the primary CME: how the different parts of the front propagate.

Reads the density of the spherical frames of ``export_density_grid``.  On every ray of the sector of
``--half_angle`` around the CME direction (``--latitude``, ``--longitude``, inertial HCI frame) the
leading edge is tracked with ``core_tracks`` of ``plot_cme_propagation`` against the ambient corona
(the first frame, before the CME, corotated with the Sun to every time).  Rays whose track covers
less than ``MIN_TRACK_HOURS`` are left out.  The deviation is the leading edge of every ray minus
the median leading edge of all rays at the same time.  Three figures, named after ``--out_path``
with the suffixes

* ``_maps``: the deformation of the front (the time average of the deviation), and the mean speed
  (linear fit) and the acceleration (constant-acceleration fit) of the leading edge of every ray;
* ``_times``: the deviation at four times between the first and the last detection on the axis;
* ``_tracks``: the deviation over time on the axis and ``FLANK_OFFSET`` to the north, south, east
  and west of it (marked in the four maps), with their mean speed and acceleration.

All where the core lies beyond ``TRACK_MIN_RADIUS`` (the launch phase, while the core is inside it
and single rays can follow structures that are not the CME, is left out); the deviation also before
the leading rays leave the outer edge of the grid (``leaving_frame``; the median would lose them).
Maps are centered on the CME longitude, cropped to the rays with a leading edge; the axis is marked
by a cross.

Example::

    python -m sunerf.evaluation.cme.plot_cme_front --export_path /path/export \
        --out_path /path/cme_front.jpg --latitude -30 --longitude 5
"""
import argparse
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt

from sunerf.evaluation.cme.grid_series import load_series
from sunerf.evaluation.cme.mass_transport import constant_acceleration_track, fit_speed
from sunerf.evaluation.cme.plot_cme_propagation import TRACK_MIN_RADIUS, ambient_on_rays, core_tracks, sector

# shortest leading-edge track of a ray that is shown [h]
MIN_TRACK_HOURS = 6.0
# offset of the flank tracks from the CME axis [deg]
FLANK_OFFSET = 15.0
# fraction of the rays that leave the grid at which the deviation stops (single fast structures leave earlier)
LEAVING_FRACTION = 0.01


def leaving_frame(front, outer_radius):
    """First frame at which ``LEAVING_FRACTION`` of the rays with a leading edge ``front[time, ...]`` have left
    the grid (a track ends and its last step carried on reaches ``outer_radius``); the number of frames if
    fewer leave."""
    rays = front.reshape(len(front), -1)
    exits = []
    for track in rays.T:
        finite = np.flatnonzero(np.isfinite(track))
        if finite.size < 2 or finite[-1] == len(track) - 1 or finite[-2] != finite[-1] - 1:
            continue
        if 2 * track[finite[-1]] - track[finite[-2]] >= outer_radius:
            exits.append(finite[-1] + 1)
    left = np.cumsum(np.bincount(exits, minlength=len(front) + 1))
    reached = np.flatnonzero(left >= LEAVING_FRACTION * np.isfinite(rays).any(axis=0).sum())
    return int(reached[0]) if reached.size else len(front)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--out_path', required=True,
                        help='Output figure (jpg); the three figures carry the suffixes _maps, _times, _tracks')
    parser.add_argument('--latitude', type=float, required=True, help='CME direction, latitude [deg]')
    parser.add_argument('--longitude', type=float, required=True, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--half_angle', type=float, default=40.0,
                        help='Half width of the sector in latitude and longitude [deg]')
    args = parser.parse_args(argv)

    series = load_series(args.export_path, keys=('density',))
    radius, latitude, longitude, times = series['radius'], series['latitude'], series['longitude'], series['times']
    seconds = (times - times[0]).astype(float)
    density = np.nan_to_num(series.pop('density'), copy=False)

    # sector rays on a [lat, lon] grid centered on the CME longitude
    in_latitude, in_longitude = sector(latitude, longitude, np.deg2rad((args.latitude, args.longitude)),
                                       np.deg2rad(args.half_angle))
    lat_index = np.flatnonzero(in_latitude)
    offset = np.rad2deg((longitude - np.deg2rad(args.longitude) + np.pi) % (2 * np.pi) - np.pi)
    lon_index = np.flatnonzero(in_longitude)
    lon_index = lon_index[np.argsort(offset[lon_index])]
    grid = np.meshgrid(lat_index, lon_index, indexing='ij')
    map_latitude, map_offset = np.rad2deg(latitude[lat_index]), offset[lon_index]

    ambient, = ambient_on_rays(density[0], longitude, seconds, [grid])
    values = density[:, grid[0], grid[1]]  # [time, lat, lon, r]
    tracks = np.full(values.shape[:3] + (3,), np.nan)
    for i in range(lat_index.size):
        for j in range(lon_index.size):
            tracks[:, i, j] = core_tracks(values[:, i, j], ambient[:, i, j], radius, seconds)
    core, front = tracks[..., 1], tracks[..., 2]
    finite = np.isfinite(front)
    span = np.where(finite.any(axis=0), (np.where(finite, seconds[:, None, None], -np.inf).max(axis=0)
                                         - np.where(finite, seconds[:, None, None], np.inf).min(axis=0)), 0.0)
    front[:, span < MIN_TRACK_HOURS * 3600.0] = np.nan

    # axis ray and the flank rays
    def nearest(lat, lon_offset):
        return (int(np.argmin(np.abs(map_latitude - lat))), int(np.argmin(np.abs(map_offset - lon_offset))))

    axis = nearest(args.latitude, 0.0)
    flanks = {'Axis': axis,
              f'North (+{FLANK_OFFSET:.0f}°)': nearest(args.latitude + FLANK_OFFSET, 0.0),
              f'South (-{FLANK_OFFSET:.0f}°)': nearest(args.latitude - FLANK_OFFSET, 0.0),
              f'East (-{FLANK_OFFSET:.0f}° lon)': nearest(args.latitude, -FLANK_OFFSET),
              f'West (+{FLANK_OFFSET:.0f}° lon)': nearest(args.latitude, FLANK_OFFSET)}
    # leading edge without the launch phase (core inside TRACK_MIN_RADIUS)
    propagation = np.where(core >= TRACK_MIN_RADIUS, front, np.nan)
    exit_frame = leaving_frame(front, radius[-1])
    within = np.arange(len(times)) < exit_frame
    detected = np.flatnonzero(np.isfinite(propagation[:, axis[0], axis[1]]) & within)
    if detected.size == 0:
        raise ValueError('No leading edge on the CME axis')
    shown = detected[np.linspace(0, detected.size - 1, 4).round().astype(int)]
    if exit_frame < len(times):
        print(f'Leading rays leave the grid at {times[exit_frame]}; the deviation stops before')

    # deviation from the leading edge of all rays (their median, against structures apart from the front)
    # until the leading rays leave the grid, its time average (the deformation), and the kinematics of every ray
    fitted = np.isfinite(propagation).sum(axis=0) >= 5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median_front = np.nanmedian(propagation, axis=(1, 2))
        offsets = np.where(within[:, None, None], propagation - median_front[:, None, None], np.nan)
        deviation = np.nanmean(offsets, axis=0)
    deviation[~(fitted & np.isfinite(offsets).any(axis=0))] = np.nan
    speed = np.full(front.shape[1:], np.nan)
    acceleration = np.full(front.shape[1:], np.nan)
    for i, j in zip(*np.nonzero(fitted)):
        speed[i, j] = fit_speed(seconds, propagation[:, i, j])
        acceleration[i, j] = constant_acceleration_track(seconds, propagation[:, i, j])[2]
    print(f'Rays with a leading edge: {np.isfinite(speed).sum()} of {speed.size}; axis speed '
          f'{speed[axis]:.0f} km/s, acceleration {acceleration[axis]:.1f} m/s^2')

    # maps cropped to the rays with a leading edge
    tracked_rays = np.isfinite(propagation).any(axis=0)
    rows, columns = np.flatnonzero(tracked_rays.any(axis=1)), np.flatnonzero(tracked_rays.any(axis=0))
    map_extent = ((map_offset[columns[0]] - 1.0, map_offset[columns[-1]] + 1.0),
                  (map_latitude[rows[0]] - 1.0, map_latitude[rows[-1]] + 1.0))
    base, extension = os.path.splitext(args.out_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    def map_axes(fig, n_columns):
        """A row of maps with shared axes and one axis label each."""
        axes = fig.subplots(1, n_columns, sharex=True, sharey=True)
        for ax in axes:
            ax.plot(0.0, args.latitude, marker='+', color='black', ms=8)
            ax.set_aspect('equal')
            ax.set_xlim(*map_extent[0])
            ax.set_ylim(*map_extent[1])
        fig.supxlabel(f'Longitude from {args.longitude:g}° (HCI) [deg]', fontsize=9)
        fig.supylabel('Latitude [deg]', fontsize=9)
        return axes

    def save(fig, suffix):
        path = f'{base}_{suffix}{extension}'
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f'Figure written to {path}')

    plt.rcParams.update({'font.size': 8})

    # deformation, speed and acceleration maps; the colorbars on top carry the titles
    # symmetric color ranges; structures apart from the CME front lie beyond them
    deviation_limit = np.nanpercentile(np.abs(deviation), 90)
    limit = np.nanpercentile(np.abs(acceleration), 90)
    fig = plt.figure(figsize=(10, 3.1), layout='constrained')
    for ax, (data, norm, cmap, label) in zip(map_axes(fig, 3), (
            (deviation, (-deviation_limit, deviation_limit), 'RdBu_r',
             r'Deformation: mean deviation [R$_\odot$]'),
            (speed, np.nanpercentile(speed, (5, 95)), 'cividis', r'Mean speed (linear fit) [km s$^{-1}$]'),
            (acceleration, (-limit, limit), 'PiYG', r'Acceleration (constant-acceleration fit) [m s$^{-2}$]'))):
        image = ax.pcolormesh(map_offset, map_latitude, data, cmap=cmap, vmin=norm[0], vmax=norm[1],
                              shading='nearest')
        fig.colorbar(image, ax=ax, location='top', shrink=0.95, pad=0.02, aspect=30, label=label)
    save(fig, 'maps')

    # deviation maps at four times
    # symmetric color range; structures apart from the CME front lie beyond it
    offset_limit = np.nanpercentile(np.abs(offsets[shown]), 90)
    fig = plt.figure(figsize=(10, 2.4), layout='constrained')
    time_axes = map_axes(fig, 4)
    for ax, t in zip(time_axes, shown):
        image = ax.pcolormesh(map_offset, map_latitude, offsets[t], cmap='RdBu_r', vmin=-offset_limit,
                              vmax=offset_limit, shading='nearest')
        ax.set_title(f'{str(times[t])[5:16].replace("T", " ")}, median {median_front[t]:.0f} R$_\\odot$')
        # the axis and flank points of the tracks, in the colors of their lines
        for index, (name, (i, j)) in enumerate(flanks.items()):
            ax.plot(map_offset[j], map_latitude[i], marker='o', mfc='none', mec=f'C{index}', mew=1.5, ms=6)
            ax.annotate(name.split()[0], (map_offset[j], map_latitude[i]), xytext=(4, 4), color=f'C{index}',
                        textcoords='offset points', fontsize=7, fontweight='bold')
    fig.colorbar(image, ax=time_axes, shrink=0.9, pad=0.01, aspect=25,
                 label=r'Leading edge $-$ median [R$_\odot$]')
    save(fig, 'times')

    # deviation of the axis and flank rays over time, over the spread of all rays (16-84 percentiles, the
    # interval of one standard deviation that ignores the structures apart from the front)
    fig = plt.figure(figsize=(10, 3.2), layout='constrained')
    ax = fig.subplots()
    hours = seconds / 3600.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        spread = np.nanpercentile(offsets.reshape(len(offsets), -1), (16, 84), axis=1)
    # from the first to the last frame with a leading edge on the axis (earlier frames hold only single rays
    # apart from the CME)
    tracked = (np.arange(len(times)) >= detected[0]) & (np.arange(len(times)) <= detected[-1])
    spread[:, ~tracked] = np.nan
    ax.fill_between(hours, spread[0], spread[1], color='0.85', lw=0, label='All rays (16-84%)')
    for name, (i, j) in flanks.items():
        kinematics = (f'{speed[i, j]:.0f} km s$^{{-1}}$, {acceleration[i, j]:.1f} m s$^{{-2}}$'
                      if np.isfinite(speed[i, j]) else 'no leading edge')
        ax.plot(hours, offsets[:, i, j], marker='.', ms=4, lw=1.2, label=f'{name}: {kinematics}')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xlabel(f'Hours since {times[0]}'.replace('T', ' '))
    # median leading edge at the times of the bottom axis, from the frames with a leading edge on the axis
    # (earlier frames hold only single rays apart from the CME)
    hour_points = hours[detected]
    radius_points = np.maximum.accumulate(median_front[detected])
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= hour_points[0]) & (ticks <= hour_points[-1])]
    top = ax.secondary_xaxis('top')
    top.set_xticks(ticks, [f'{value:.0f}' for value in np.interp(ticks, hour_points, radius_points)])
    top.set_xlabel(r'Median leading edge [R$_\odot$]')
    ax.set_ylabel(r'Leading edge $-$ median [R$_\odot$]')
    ax.set_xlim(hours[detected[0]] - 0.5, hours[detected[-1]] + 0.5)
    fig.legend(loc='outside right upper', fontsize=7)
    ax.grid(alpha=0.3)
    save(fig, 'tracks')


if __name__ == '__main__':
    main()
