"""Kinematics and mass budget of the dense core of the primary CME.

Reads the density of the spherical frames of ``export_density_grid``.  Along the CME axis (the rays
within ``AXIS_HALF_ANGLE`` of ``--latitude``, ``--longitude``, inertial HCI frame) the core, the
leading edge and the rear are measured against the first frame corotated to each time.
A motion-constrained path through observed peaks of the median radius-compensated excess identifies
the common ridge. The strongest sustained path must include a high-contrast detection beyond
the minimum seed radius. Individual rays supply the core positions and half-height crossings on that
ridge at each frame. By default each position requires a strict majority of the axis rays.
With --edge_fraction the edge is the first crossing of that fraction of the core excess on each flank.
With --above_background the edge is the first crossing of the specified fractional enhancement
over the local background. Edge settings are applied after the core is selected.
With --median_profile the core and both crossings come directly from the median
excess profile, with majority coverage in each radial bin. Shading spans
the 16th to 84th percentiles of the supporting positions.  The figure holds

* the distance-time map of the compensated density ``r^2 n_e`` along the axis with the tracks;
* the speed of the leading edge, the core and the rear (quadratic fits in a
  sliding window of ``KINEMATICS_WINDOW_HOURS``);
* the signed radial offsets of the front and rear from the core, averaged over supporting rays
  with shading at one standard deviation. In median-profile mode each side uses all rays
  with a matched core and edge, provided the median edge exists and at least two rays contribute;
* the mass between the rear (at least ``TRACK_MIN_RADIUS``) and the leading edge within the sector
  of ``--half_angle`` against the leading edge, and the ambient mass of the same volume.  The same
  volume mirrored to the northern
  latitude ``-latitude`` gives the noise floor: the band of its difference between the
  reconstruction and the ambient around the ambient mass.

Kinematics are fitted within uninterrupted tracks. Mass is omitted when either boundary is missing.
The mass curves integrate total and ambient density separately.  Output: ``--out_path``.

Example::

    python -m sunerf.evaluation.cme.plot_cme_mass_budget --export_path /path/export \
        --out_path /path/cme_mass_budget.jpg --latitude -30 --longitude 5
"""
import argparse
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from sunerf.evaluation.cme.grid_series import GRAMS_PER_ELECTRON, cell_volume, load_series
from sunerf.evaluation.cme.mass_transport import local_kinematics
from sunerf.evaluation.cme.plot_cme_propagation import (
    CORE_CONTRAST, FOLLOW_CONTRAST, KM_PER_RSUN, MAX_SPEED_KMS, TRACK_MIN_RADIUS, ambient_on_rays, half_maximum, sector)


# half angle of the cone around the CME axis of the tracking [deg]
AXIS_HALF_ANGLE = 10.0
# window of the quadratic fits of the speed and acceleration [h]
KINEMATICS_WINDOW_HOURS = 6.0


def _axis_ridge(density, ambient, radius, seconds, ray_counts=None, *, seed_min_radius=TRACK_MIN_RADIUS):
    """Link measured peaks of the median excess into the strongest sustained outward ridge."""
    valid = np.isfinite(density) & np.isfinite(ambient) & (ambient > 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        excess = np.nanmedian(np.where(valid, (density - ambient) * radius**2, np.nan), axis=1)
    majority = density.shape[1] // 2 + 1 if ray_counts is None else ray_counts[:, None] // 2 + 1
    follow = (valid & (density >= FOLLOW_CONTRAST * ambient)).sum(axis=1) >= majority
    seed = (valid & (density >= CORE_CONTRAST * ambient)).sum(axis=1) >= majority
    peaks = []
    strength = []
    for t, profile in enumerate(excess):
        candidates = find_peaks(profile)[0]
        candidates = candidates[follow[t, candidates] & (profile[candidates] > 0)]
        peaks.append(candidates)
        if candidates.size:
            reference = np.nanmedian(np.where(valid[t], ambient[t], np.nan)[:, candidates], axis=0)
            contrast = profile[candidates] / (radius[candidates]**2 * reference)
            # Bounded weights reward persistent contrast without letting one bright frame dominate.
            strength.append(1 + np.clip(contrast, 0, CORE_CONTRAST))
        else:
            strength.append(np.empty(0))

    def link(direction):
        scores = [weights.copy() for weights in strength]
        links = [np.full((len(nodes), 2), -1, dtype=int) for nodes in peaks]
        order = range(len(peaks)) if direction == 1 else reversed(range(len(peaks)))
        for t in order:
            for gap in (1, 2):
                other = t - direction * gap
                if not 0 <= other < len(peaks) or not peaks[other].size:
                    continue
                reach = MAX_SPEED_KMS * abs(seconds[t] - seconds[other]) / KM_PER_RSUN
                for j, peak in enumerate(peaks[t]):
                    step = direction * (radius[peak] - radius[peaks[other]])
                    nearby = np.flatnonzero((step >= 0) & (step <= reach))
                    if nearby.size:
                        best = nearby[np.argmax(scores[other][nearby])]
                        score = strength[t][j] + scores[other][best]
                        if score > scores[t][j]:
                            scores[t][j] = score
                            links[t][j] = other, best
        return scores, links

    past, previous = link(1)
    future, following = link(-1)
    best_score = 0
    anchor = None
    for t, nodes in enumerate(peaks):
        for j in np.flatnonzero(seed[t, nodes] & (radius[nodes] >= seed_min_radius)):
            score = past[t][j] + future[t][j] - strength[t][j]
            if score > best_score:
                best_score, anchor = score, (t, j)
    ridge = np.full(len(seconds), -1, dtype=int)
    if anchor is not None:
        for links in (previous, following):
            t, j = anchor
            while t >= 0:
                ridge[t] = peaks[t][j]
                t, j = links[t][j]
    return ridge, excess


def first_crossing(values, end, peak, fraction):
    """Fractional index where a continuous flank first reaches the specified fraction of the peak."""
    if not np.isfinite(values[peak]) or values[peak] <= 0:
        return np.nan
    level = fraction * values[peak]
    step = 1 if end > peak else -1
    for i in range(peak + step, end + step, step):
        if not np.isfinite(values[i]):
            break
        if values[i] <= level:
            previous = values[i - step]
            return i - step + step * (previous - level) / (previous - values[i])
    return np.nan


def _continue_positive_ridge(ridge, profiles, radius, seconds):
    """Continue from the last accepted core using resolved positive peaks on the same outward path."""
    measured = np.flatnonzero(ridge >= 0)
    if not measured.size:
        return
    last = measured[-1]
    for t in range(last + 1, len(ridge)):
        if t - last > 2:
            break
        profile = profiles[t]
        peaks = find_peaks(profile)[0]
        step = radius[peaks] - radius[ridge[last]]
        reachable = (step >= 0) & (step <= MAX_SPEED_KMS * (seconds[t] - seconds[last]) / KM_PER_RSUN)
        # Resolve the peak on at least two bins, independent of the eventual flank threshold.
        adjacent = np.maximum(profile[peaks - 1], profile[peaks + 1])
        peaks = peaks[reachable & (profile[peaks] > 0) & (adjacent > 0.1 * profile[peaks])]
        if peaks.size:
            ridge[t] = peaks[np.argmax(profile[peaks])]
            last = t


def _ray_measurements(density, ambient, radius, guide, axis_profile, crossing=half_maximum,
                      edge_profiles=None):
    """Measure individual peaks and edge crossings on the lobe containing the guide."""
    grid = np.arange(len(radius))
    bounds = [half_maximum(axis_profile, end, guide) for end in (0, len(radius) - 1)]
    lower = int(np.floor(bounds[0])) if np.isfinite(bounds[0]) else 0
    upper = int(np.ceil(bounds[1])) if np.isfinite(bounds[1]) else len(radius) - 1
    positions = np.full((len(density), 3), np.nan)
    excess = np.where(ambient > 0, (density - ambient) * radius**2, np.nan)
    for j, profile in enumerate(excess):
        if not np.isfinite(profile[guide]) or profile[guide] <= 0:
            continue
        peaks = find_peaks(profile)[0]
        peaks = peaks[(peaks >= lower) & (peaks <= upper) & (ambient[j, peaks] > 0)
                      & (density[j, peaks] >= FOLLOW_CONTRAST * ambient[j, peaks])
                      & (profile[guide] >= 0.5 * profile[peaks])]
        for peak in peaks[np.argsort(profile[peaks])[::-1]]:
            between = profile[min(peak, guide):max(peak, guide) + 1]
            below = between < 0.5 * profile[peak]
            if not np.isfinite(between).all() or (below[1:] & below[:-1]).any():
                continue
            rear = half_maximum(profile, 0, peak)
            front = half_maximum(profile, len(radius) - 1, peak)
            left = int(np.ceil(rear)) if np.isfinite(rear) else peak
            right = int(np.floor(front)) if np.isfinite(front) else peak
            # A shoulder blended with a stronger peak has no independent half-height lobe.
            if np.max(profile[left:right + 1]) > profile[peak]:
                continue
            break
        else:
            continue

        # The ray core is fixed even if neither flank has a measurable crossing.
        if crossing is not half_maximum:
            flank = profile if edge_profiles is None else edge_profiles[j]
            rear = crossing(flank, 0, peak)
            front = crossing(flank, len(radius) - 1, peak)
        positions[j] = np.interp([rear, peak, front], grid, radius)
    return positions


def axis_tracks(density, ambient, radius, seconds):
    """Follow a common ridge in time and measure its edges on a majority of the axis rays."""
    tracks, spread, _ = _axis_track_statistics(density, ambient, radius, seconds)
    return tracks, spread


def _axis_track_statistics(density, ambient, radius, seconds, ray_counts=None, *, start_index=0, ray_edges=None,
                           seed_min_radius=TRACK_MIN_RADIUS, median_profile=False, edge_fraction=None,
                           continue_to_index=None, weak_frames=None, above_background=None):
    n_time, n_rays, _ = density.shape
    if n_rays == 0:
        raise ValueError('No rays in the CME axis patch')
    window = slice(start_index, None if continue_to_index is None else continue_to_index + 1)
    counts = None if ray_counts is None else ray_counts[window]
    ridge, excess = _axis_ridge(density[window], ambient[window], radius, seconds[window], counts,
                                seed_min_radius=seed_min_radius)
    if continue_to_index is not None:
        original = ridge.copy()
        valid = np.isfinite(density[window]) & np.isfinite(ambient[window]) & (ambient[window] > 0)
        required = n_rays // 2 + 1 if counts is None else counts[:, None] // 2 + 1
        covered = np.where(valid.sum(axis=1) >= required, excess, np.nan)
        _continue_positive_ridge(ridge, covered, radius, seconds[window])
        if weak_frames is not None:
            weak_frames[window] = (original < 0) & (ridge >= 0)
    tracks = np.full((n_time, 3), np.nan)
    spread = np.full((2, n_time, 3), np.nan)
    # [time, rear/front, mean/std] of paired edge-minus-core measurements.
    offsets = np.full((n_time, 2, 2), np.nan)
    crossing = half_maximum if edge_fraction is None else partial(first_crossing, fraction=edge_fraction)
    if above_background is not None:
        crossing = partial(first_crossing, fraction=0.)
    last = [None, None, None]
    for i in np.flatnonzero(ridge >= 0):
        t = start_index + i
        majority = (n_rays if ray_counts is None else ray_counts[t]) // 2 + 1
        peak = ridge[i]
        profile = excess[i]
        edge_profiles = None
        if above_background is not None:
            valid = np.isfinite(density[t]) & np.isfinite(ambient[t]) & (ambient[t] > 0)
            ratio = np.divide(density[t], ambient[t], out=np.full(density[t].shape, np.nan),
                              where=valid)
            edge_profiles = ratio - (1 + above_background)
        if median_profile:
            # Missing rays must not create a crossing on a sparsely sampled flank.
            covered = np.isfinite(density[t]) & np.isfinite(ambient[t]) & (ambient[t] > 0)
            profile = np.where(covered.sum(axis=0) >= majority, profile, np.nan)
            flank = profile
            if edge_profiles is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    flank = np.nanmedian(edge_profiles, axis=0)
                flank[covered.sum(axis=0) < majority] = np.nan
            rear = crossing(flank, 0, peak)
            front = crossing(flank, len(radius) - 1, peak)
            tracks[t] = np.interp([rear, peak, front], np.arange(len(radius)), radius)
        frame = _ray_measurements(density[t], ambient[t], radius, peak, profile, crossing,
                                  edge_profiles)
        if ray_edges is not None:
            ray_edges[t] = frame
        if median_profile:
            # These are within-ray distances. Agreement on absolute edge radius is
            # only needed below for the envelope of the position measurements.
            for side, column in enumerate((0, 2)):
                distances = frame[:, column] - frame[:, 1]
                distances = distances[np.isfinite(distances)]
                if np.isfinite(tracks[t, column]) and distances.size >= 2:
                    offsets[t, side] = distances.mean(), distances.std()
        for column in (1, 0, 2):
            if median_profile and not np.isfinite(tracks[t, column]):
                continue
            valid = np.isfinite(frame[:, column])
            positions = frame[valid, column]
            if positions.size < majority:
                continue
            if column != 1:
                median = np.median(positions)
                half_width = np.median(np.abs(positions - frame[valid, 1]))
                valid &= np.abs(frame[:, column] - median) <= half_width
                positions = frame[valid, column]
                if positions.size < majority:
                    continue
            lo, position, hi = np.percentile(positions, (16, 50, 84))
            previous = last[column]
            if not median_profile and previous is not None:
                reach = MAX_SPEED_KMS * (seconds[t] - seconds[previous]) / KM_PER_RSUN
                if abs(position - tracks[previous, column]) > reach:
                    if column == 1:
                        break
                    continue
            if not median_profile:
                if (column == 0 and position >= tracks[t, 1]) or (column == 2 and position <= tracks[t, 1]):
                    continue
                tracks[t, column] = position
                last[column] = t
            spread[:, t, column] = lo, hi
            if column != 1 and not median_profile:
                distances = positions - frame[valid, 1]
                offsets[t, column // 2] = distances.mean(), distances.std()
    return tracks, spread, offsets


def compression_track(radius, density, velocity, front, ray_edges):
    """Fit the primary velocity-supported compression near the half-height edge, in solar radii."""
    step = radius[1] - radius[0]
    half = max(1, round(1.5 / step))
    with np.errstate(divide='ignore', invalid='ignore'):
        log_density = np.log(density * radius**2)
    gradient = -np.gradient(gaussian_filter1d(log_density, 1.0 / step, axis=-1), radius, axis=-1)
    track = np.full(len(front), np.nan)
    for t in np.flatnonzero(np.isfinite(front)):
        offsets, positions = [], []
        for ray, (_, core, edge) in enumerate(ray_edges[t]):
            if not np.isfinite(edge):
                continue
            g = gradient[t, ray]
            peaks = find_peaks(g, prominence=.01)[0]
            near = peaks[(radius[peaks] > core) & (radius[peaks] >= edge - 2) & (radius[peaks] <= edge + 6)
                         & (peaks >= half) & (peaks < len(radius) - half) & (g[peaks] > 0)]
            if not near.size:
                continue
            k = near[np.argmax(g[near])]
            log_ratio = log_density[t, ray, k-half] - log_density[t, ray, k+half]
            delta_v = velocity[t, ray, k-half] - velocity[t, ray, k+half]
            if not (log_ratio >= np.log(1.08) and delta_v >= 10.):
                continue
            shift = .5 * (g[k-1] - g[k+1]) / (g[k-1] - 2*g[k] + g[k+1])
            position = radius[k] + shift * step
            offsets.append(position - edge)
            positions.append(position)
        if len(offsets) > density.shape[1] // 2:
            offsets = np.asarray(offsets)
            support = max((np.abs(offsets - offset) <= 2 for offset in offsets), key=np.count_nonzero)
            if support.sum() > density.shape[1] // 2:
                track[t] = np.mean(np.asarray(positions)[support])
    return track


def track_kinematics(seconds, track, window):
    """Fit each continuous run of measurements separately."""
    result = np.full((2, len(track)), np.nan)
    valid = np.flatnonzero(np.isfinite(track))
    for run in np.split(valid, np.flatnonzero(np.diff(valid) > 1) + 1):
        if run.size:
            result[:, run] = local_kinematics(seconds[run], track[run], window)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--out_path', required=True, help='Output figure (jpg)')
    parser.add_argument('--latitude', type=float, required=True, help='CME direction, latitude [deg]')
    parser.add_argument('--longitude', type=float, required=True, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--half_angle', type=float, default=30.0,
                        help='Half width of the mass sector in latitude and longitude [deg]')
    parser.add_argument('--seed_min_radius', type=float, default=TRACK_MIN_RADIUS,
                        help='Minimum core radius for a high-contrast tracking seed [solar radii]')
    parser.add_argument('--fit_max_radius', type=float, help='Outer radius eligible for tracing [solar radii]')
    parser.add_argument('--end_hours', type=float, help='Last hour included in the evaluation and figure')
    parser.add_argument('--fit_start_hours', type=float, default=0.,
                        help='First hour eligible for ridge fitting, relative to the first export frame')
    parser.add_argument('--continue_to_hours', type=float,
                        help='End the fit at this hour and follow resolved positive median peaks after contrast loss')
    parser.add_argument('--show_shock', action='store_true', help='Plot the velocity-supported candidate shock front')
    parser.add_argument('--median_profile', action='store_true',
                        help='Measure the core and edges on the median compensated excess profile')
    edges = parser.add_mutually_exclusive_group()
    edges.add_argument('--above_background', type=float,
                       help='Fractional enhancement over the local ambient density for both edges (e.g. 0.10)')
    edges.add_argument('--edge_fraction', type=float,
                       help='Use the first crossing of this fraction of the core excess on each flank (0 to <1)')
    edges.add_argument('--zero_crossing', dest='edge_fraction', action='store_const', const=0.,
                       help='Equivalent to --edge_fraction 0')
    args = parser.parse_args(argv)
    if args.above_background is not None and not (np.isfinite(args.above_background) and args.above_background >= 0):
        parser.error('--above_background must be finite and nonnegative')
    if args.edge_fraction is not None and not 0 <= args.edge_fraction < 1:
        parser.error('--edge_fraction must be in [0, 1)')

    series = load_series(args.export_path, keys=('density',))
    radius, latitude, longitude, times = series['radius'], series['latitude'], series['longitude'], series['times']
    seconds = (times - times[0]).astype(float)
    hours = seconds / 3600.0
    if args.end_hours is not None:
        if len(hours) < 2 or not hours[1] <= args.end_hours <= hours[-1]:
            parser.error('--end_hours must include at least two frames and be within the export')
        stop = np.searchsorted(hours, args.end_hours, side='right')
        hours, seconds, times = hours[:stop], seconds[:stop], times[:stop]
        series['density'] = series['density'][:stop]
    n_lon = longitude.size

    def rays(direction, half_angle):
        in_latitude, in_longitude = sector(latitude, longitude, np.deg2rad(direction), np.deg2rad(half_angle))
        selected = np.flatnonzero((in_latitude[:, None] & in_longitude[None, :]).ravel())
        return selected // n_lon, selected % n_lon

    axis_rays = rays((args.latitude, args.longitude), AXIS_HALF_ANGLE)
    sectors = {'cme': rays((args.latitude, args.longitude), args.half_angle),
               'control': rays((-args.latitude, args.longitude), args.half_angle)}

    # density and the ambient (first frame corotated to every time) on the axis rays and in the sectors
    density = series.pop('density')
    axis_ambient, *ambients = ambient_on_rays(density[0], longitude, seconds, [axis_rays, *sectors.values()])
    sector_ambient = dict(zip(sectors, ambients))  # [time, ray, r]
    axis_density = density[:, axis_rays[0], axis_rays[1]]

    if not 0 <= args.fit_start_hours <= hours[-1]:
        parser.error('--fit_start_hours is outside the export time range')
    if not 0 < args.seed_min_radius <= radius[-1]:
        parser.error('--seed_min_radius must be positive and within the outer export radius')
    fit_bins = slice(None)
    if args.fit_max_radius is not None:
        if not args.seed_min_radius <= args.fit_max_radius <= radius[-1]:
            parser.error('--fit_max_radius must be between the seed and outer export radii')
        fit_bins = slice(0, np.searchsorted(radius, args.fit_max_radius, side='right'))
    fit_radius = radius[fit_bins]
    start_index = np.searchsorted(hours, args.fit_start_hours)
    end_index = None
    weak_frames = np.zeros(len(hours), dtype=bool)
    if args.continue_to_hours is not None:
        if not args.median_profile or not args.fit_start_hours <= args.continue_to_hours <= hours[-1]:
            parser.error('--continue_to_hours requires a median-profile fit and an end within the fitting interval')
        end_index = np.searchsorted(hours, args.continue_to_hours, side='right') - 1
    ray_edges = np.full((*axis_density.shape[:2], 3), np.nan) if args.show_shock else None
    tracks, spread, offsets = _axis_track_statistics(axis_density[..., fit_bins], axis_ambient[..., fit_bins],
                                                    fit_radius, seconds,
                                                    start_index=start_index, ray_edges=ray_edges,
                                                    seed_min_radius=args.seed_min_radius, median_profile=args.median_profile,
                                                    edge_fraction=args.edge_fraction, continue_to_index=end_index,
                                                    weak_frames=weak_frames, above_background=args.above_background)
    rear, core, front = tracks.T
    bounded = np.isfinite(rear) & np.isfinite(front)
    print(f'Axis detections in {len(times)} frames: rear {np.isfinite(rear).sum()}, '
          f'core {np.isfinite(core).sum()}, front {np.isfinite(front).sum()}, both edges {bounded.sum()}')
    if bounded.any():
        t = np.flatnonzero(bounded)[-1]
        print(f'Last mass measurement at {hours[t]:g} h, leading edge {front[t]:.2f} solar radii')
    if weak_frames.any():
        print('Low-contrast continuation (hour, rear/core/front):',
              np.round(np.column_stack([hours[weak_frames], tracks[weak_frames]]), 2))

    # kinematics of the rear, the core and the leading edge: sliding quadratic fits
    window = max(KINEMATICS_WINDOW_HOURS, 5.0 * np.median(np.diff(hours))) * 3600.0
    speeds = [track_kinematics(seconds, track, window)[0] for track in tracks.T]

    if args.show_shock:
        velocity = []
        for path in sorted(Path(args.export_path).glob('rho*.npz'))[:len(times)]:
            with np.load(path) as frame:
                velocity.append(frame['velocity'][axis_rays])
        shock = compression_track(fit_radius, axis_density[..., fit_bins], np.asarray(velocity)[..., fit_bins],
                                  front, ray_edges)
        shock_speed = track_kinematics(seconds, shock, window)[0]
        for name, track in [('Leading edge', front), ('Candidate shock', shock)]:
            detected = np.flatnonzero(np.isfinite(track))
            if detected.size:
                print(f'{name}: {detected.size} detections, first at {hours[detected[0]]:g} h, '
                      f'last at {hours[detected[-1]]:g} h')
            else:
                print(f'{name}: no detections')

    # masses between the rear and the leading edge in the CME sector and the mirrored control sector
    volume = cell_volume(radius, latitude, longitude) * GRAMS_PER_ELECTRON
    inner = np.maximum(rear, TRACK_MIN_RADIUS)
    between = (radius[None, :] >= inner[:, None]) & (radius[None, :] <= front[:, None])  # [time, r]
    masses = {}
    for key, selection in sectors.items():
        cells = volume[selection]
        total = (density[:, selection[0], selection[1]] * cells[None]).sum(axis=1)  # [time, r]
        ambient = (sector_ambient[key] * cells[None]).sum(axis=1)
        total_mass = np.where(between, total, 0.0).sum(axis=1)
        ambient_mass = np.where(between, ambient, 0.0).sum(axis=1)
        masses[key] = (np.where(bounded, total_mass, np.nan), np.where(bounded, ambient_mass, np.nan))
    noise = np.abs(masses['control'][0] - masses['control'][1])

    fig, axes = plt.subplots(4, 1, figsize=(9, 13.5), constrained_layout=True,
                             gridspec_kw={'height_ratios': (1.3, 0.7, 0.65, 1.0)})
    if weak_frames.any():
        half_step = np.median(np.diff(hours)) / 2
        for panel in axes[:3]:
            panel.axvspan(hours[weak_frames][0] - half_step, hours[weak_frames][-1] + half_step,
                          color='tab:orange', alpha=.13, zorder=1,
                          label='Low-contrast continuation' if panel is axes[0] else None)
    ax = axes[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        compensated = np.nanmedian(axis_density, axis=1) * radius**2
    positive = compensated[compensated > 0]
    im = ax.pcolormesh(hours, radius, compensated.T, shading='nearest', cmap='viridis',
                       norm=LogNorm(*np.percentile(positive, (5., 95.))))
    outline = [patheffects.withStroke(linewidth=2.5, foreground='black')]
    edge_label = 'half maximum' if args.edge_fraction is None else f'{args.edge_fraction:.0%} of peak'
    if args.above_background is not None:
        edge_label = f'{args.above_background:.0%} above background'
    for column, color, label in ((2, 'red', f'Leading edge ({edge_label})'), (1, 'white', 'Core'),
                                 (0, 'white', f'Rear ({edge_label})')):
        ax.fill_between(hours, spread[0, :, column], spread[1, :, column], color=color, alpha=0.25, lw=0)
        ax.plot(hours, tracks[:, column], color=color, lw=1.2, ls='--' if column == 0 else '-', label=label,
                path_effects=outline)
    if args.show_shock:
        ax.plot(hours, shock, color='cyan', ls='--', marker='.', ms=2.5, lw=1.3,
                path_effects=outline, label='Candidate shock (compression)')
    if args.fit_start_hours:
        ax.axvline(args.fit_start_hours, color='0.85', ls=':', lw=1, label=f'Fit start ({args.fit_start_hours:g} h)')
    ax.axhline(args.seed_min_radius, color='white', lw=0.8, ls=':',
               label=f'Minimum seed radius ({args.seed_min_radius:g} R$_\\odot$)')
    if args.fit_max_radius is not None:
        ax.axhline(args.fit_max_radius, color='0.85', lw=0.8, ls='--',
                   label=rf'Tracing limit ({args.fit_max_radius:g} R$_\odot$)')
    fig.colorbar(im, ax=ax, label=r'$r^2 n_e$ [cm$^{-3}$ R$_\odot^2$]', pad=0.01)
    ax.set_ylabel(r'Radius [R$_\odot$]')
    ax.set_title(f'Median compensated density along the CME axis (±{AXIS_HALF_ANGLE:.0f}°)')
    ax.legend(loc='upper left', fontsize=8, title='Shading: 16–84% ray spread', title_fontsize=8)
    ax.tick_params(labelbottom=False)

    ax = axes[1]
    ax.sharex(axes[0])
    for column, color, style, name in ((2, 'red', '-', 'Leading edge'), (1, 'black', '-', 'Core'),
                                       (0, 'tab:blue', '--', 'Rear')):
        ax.plot(hours, speeds[column], color=color, lw=1.4, ls=style, label=name)
    if args.show_shock:
        ax.plot(hours, shock_speed, color='teal', ls='-.', lw=1.3, label='Candidate shock')
    ax.set_ylabel(r'Speed [km s$^{-1}$]')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)
    ax.tick_params(labelbottom=False)

    ax = axes[2]
    ax.sharex(axes[0])
    ax.axhline(0, color='0.5', lw=0.8)
    for side, color, style, name in ((1, 'red', '-', 'Front − core'), (0, 'tab:blue', '--', 'Rear − core')):
        mean, std = offsets[:, side].T
        ax.plot(hours, mean, color=color, ls=style, lw=1.4, marker='.', ms=3, label=name)
        ax.fill_between(hours, mean - std, mean + std, color=color, alpha=0.2, lw=0)
    ax.set_ylabel(r'Offset from core [R$_\odot$]')
    offset_label = 'Mean ± 1 SD (≥2 paired rays)' if args.median_profile else 'Ray mean ± 1 SD'
    ax.legend(title=offset_label, fontsize=8, title_fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlabel(f'Hours since {times[0]}'.replace('T', ' '))

    ax = axes[3]
    total, ambient = masses['cme']
    ax.fill_between(front, ambient - noise, ambient + noise, color='tab:blue',
                    alpha=0.2, lw=0, label=f'Noise floor (same volume at latitude {-args.latitude:.0f}°)')
    ax.plot(front, total, color='black', lw=1.6, label='Total mass (reconstruction)')
    if weak_frames.any():
        ax.plot(front[weak_frames], total[weak_frames], 'o', ms=4, mfc='white', mec='tab:orange',
                label='Low-contrast continuation')
    ax.plot(front, ambient, color='tab:blue', lw=1.6, ls='--',
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
