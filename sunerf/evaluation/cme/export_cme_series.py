"""Reduce a density series to the CME diagnostics of the tracking and mass propagation plots.

The series is either a SuNeRF export (directory written by ``export_density``)
or the PSI cubes (glob of ``rho*.hdf``).  Every frame is evaluated inside a cone
around the CME direction; no pre-event background is subtracted (see
:mod:`sunerf.evaluation.cme.mass_transport`).  The output holds, for every frame,

* ``mass_per_radius[time, radius]``: electrons per R_sun inside the cone,
* ``body_peak_median``: radius of the outermost density enhancement of ``r^2 n_e``
  along the rays of the cone (median) [R_sun],
* ``body_edge_apex``, ``body_edge_median``: outer edge of this enhancement (90th
  percentile and median over the rays) [R_sun],
* ``cavity_median``: minimum between this enhancement and the next one inside [R_sun],
* ``body_edge_p16`` / ``_p84`` (also ``body_peak``, ``cavity``): 16th and 84th
  percentile of the radius over the rays [R_sun],
* ``detection_fraction``, ``cavity_fraction``: fraction of rays with a body / cavity,
* ``beam_density[time, radius]``: compensated density ``r^2 n_e`` along the CME
  direction (median of fixed directions within 5 deg) [cm^-3 R_sun^2], with the body
  edge and peak of this profile (``beam_body_edge``, ``beam_body_peak``),
* ``direction_latitude``, ``direction_longitude``: excess-mass weighted direction [rad].

Example::

    python -m sunerf.evaluation.cme.export_cme_series \
        --density_path /path/export/run --out_file /path/export/run/cme_series.npz
"""
import argparse
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from sunerf.evaluation.cme.density_series import load_export_series, load_series
from sunerf.evaluation.cme.mass_transport import (
    cone_mask, cone_summary, mass_per_radius, ray_structure, solid_angle_weights, structure_maps,
)

RADIUS_GRID = np.arange(3.0, 30.0 + 1e-6, 0.25)
DETECTION = dict(dr=0.25, smooth=0.5, r_search_min=RADIUS_GRID[0])
ANGULAR_STRIDE = 2
BEAM_HALF_ANGLE = np.deg2rad(5.0)
# name in the series: quantity of mass_transport.cone_summary (front: outer edge, sheath: peak of the body)
TRACKS = {'body_edge_apex': 'front_apex', 'body_edge_median': 'front_median', 'body_peak_median': 'sheath_median',
          'cavity_median': 'cavity_median', 'detection_fraction': 'detection_fraction'}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--density_path', required=True,
                        help='SuNeRF export directory or glob of PSI density cubes (rho*.hdf)')
    parser.add_argument('--out_file', required=True, help='Output npz')
    parser.add_argument('--dump_times', help='PSI dump_times.json (default: 5 min cadence from 2021-10-28T15:30)')
    parser.add_argument('--frames', default=None, help='Slice of the sorted cubes, e.g. 12: or ::2')
    parser.add_argument('--target_latitude', type=float, default=-20.0, help='CME direction latitude [deg]')
    parser.add_argument('--target_longitude', type=float, default=245.0,
                        help='CME direction Carrington longitude [deg]')
    parser.add_argument('--cone_half_angle', type=float, default=30.0, help='Cone around the CME direction [deg]')
    return parser


def cone_frame(frame, target_latitude, target_longitude, half_angle):
    """Density, axes and cone mask of the angular sector that holds the cone (radians)."""
    lat_idx = np.arange(0, frame.latitude.size, ANGULAR_STRIDE)
    lon_idx = np.arange(0, frame.longitude.size, ANGULAR_STRIDE)
    cone = cone_mask(frame.latitude[lat_idx], frame.longitude[lon_idx], target_latitude, target_longitude,
                     half_angle)
    rows, columns = cone.any(1), cone.any(0)
    if not rows.any():
        raise ValueError('The cone contains no grid rays; check the CME direction.')
    lat_idx, lon_idx = lat_idx[rows], lon_idx[columns]
    shell = frame.radius >= RADIUS_GRID[0]
    rho = frame.rho[shell][:, lat_idx][:, :, lon_idx]
    # PSI holds single negative cells; they carry no mass and no structure
    rho = np.where(rho > 0, rho, np.nan)
    return rho, frame.radius[shell], frame.latitude[lat_idx], frame.longitude[lon_idx], cone[rows][:, columns]


def mean_direction(maps, latitude, longitude):
    """Excess-mass weighted direction (latitude, longitude) of the rays with a front [rad]."""
    weights = np.nan_to_num(maps['excess_mass_per_sr']) * solid_angle_weights(latitude, longitude)
    if weights.sum() <= 0:
        return np.nan, np.nan
    lat, lon = np.meshgrid(latitude, longitude, indexing='ij')
    x, y, z = (np.sum(weights * c) for c in (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))
    return np.arctan2(z, np.hypot(x, y)), np.arctan2(y, x) % (2 * np.pi)


def beam_directions(target_latitude, target_longitude):
    """Fixed directions (latitude, longitude) of the beam: the target and two rings around it (radians)."""
    latitude, longitude = [target_latitude], [target_longitude]
    for offset, count in ((0.5 * BEAM_HALF_ANGLE, 6), (BEAM_HALF_ANGLE, 12)):
        azimuth = 2 * np.pi * np.arange(count) / count
        latitude.extend(target_latitude + offset * np.cos(azimuth))
        longitude.extend(target_longitude + offset * np.sin(azimuth) / np.cos(target_latitude))
    return np.array(latitude), np.array(longitude) % (2 * np.pi)


def beam_profile(frame, target_latitude, target_longitude):
    """Compensated density ``r^2 n_e`` on the radius grid along the target direction (radians).

    The density is interpolated at fixed directions, so that the profile does
    not depend on the mesh of the frame (the PSI mesh changes within a series).
    """
    shell = frame.radius >= RADIUS_GRID[0]
    rho = frame.rho[shell]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rho = np.where(rho > 0, np.log(rho), np.nan).transpose(1, 2, 0)  # (lat, lon, r)
    # periodic longitude
    longitude = np.concatenate([frame.longitude[-1:] - 2 * np.pi, frame.longitude, frame.longitude[:1] + 2 * np.pi])
    log_rho = np.concatenate([log_rho[:, -1:], log_rho, log_rho[:, :1]], axis=1)
    beam_latitude, beam_longitude = beam_directions(target_latitude, target_longitude)
    points = np.stack([np.clip(beam_latitude, frame.latitude[0], frame.latitude[-1]), beam_longitude], -1)
    rays = RegularGridInterpolator((frame.latitude, longitude), log_rho)(points)  # (direction, r)
    profile = np.exp(np.nanmedian(rays, axis=0)) * frame.radius[shell] ** 2
    return np.interp(RADIUS_GRID, frame.radius[shell], profile)


# name in the series: structure map of mass_transport.structure_maps
SPREADS = {'body_edge': 'front', 'body_peak': 'sheath', 'cavity': 'cavity'}


def ray_spread(values):
    """16th and 84th percentile of the finite values; NaN without detections."""
    values = values[np.isfinite(values)]
    return np.percentile(values, (16, 84)) if values.size else (np.nan, np.nan)


def reduce_frame(frame, target_latitude, target_longitude, half_angle):
    """CME diagnostics of one :class:`DensityFrame` (angles in radians)."""
    beam = beam_profile(frame, target_latitude, target_longitude)
    beam_structure = ray_structure(beam / RADIUS_GRID ** 2, RADIUS_GRID, **DETECTION)
    rho, radius, latitude, longitude, cone = cone_frame(frame, target_latitude, target_longitude, half_angle)
    mass = mass_per_radius(rho, radius, latitude, longitude, cone)
    # the mask covers the cone only up to the angular cells of the mesh; scale to the exact solid angle
    mass = mass * 2 * np.pi * (1 - np.cos(half_angle)) / (solid_angle_weights(latitude, longitude) * cone).sum()
    maps = structure_maps(rho, radius, angular_mask=cone, **DETECTION)
    summary = cone_summary(maps, latitude, longitude, cone)
    direction = mean_direction(maps, latitude, longitude)
    spreads = {name: ray_spread(maps[key][cone]) for name, key in SPREADS.items()}
    return {
        **{f'{name}_p16': spread[0] for name, spread in spreads.items()},
        **{f'{name}_p84': spread[1] for name, spread in spreads.items()},
        'mass_per_radius': np.interp(RADIUS_GRID, radius, mass),
        **{name: summary[key] for name, key in TRACKS.items()},
        'cavity_fraction': np.isfinite(maps['cavity']).sum() / cone.sum(),
        'direction_latitude': direction[0],
        'direction_longitude': direction[1],
        'beam_density': beam,
        'beam_body_edge': beam_structure.front,
        'beam_body_peak': beam_structure.sheath,
    }


def load_frames(density_path, frames=None, dump_times=None):
    if os.path.isdir(density_path):
        return load_export_series(density_path, frames)
    return load_series(density_path, 'psi', frames=frames, dump_times_path=dump_times)


def main(argv=None):
    args = build_parser().parse_args(argv)
    direction = [np.deg2rad(angle) for angle in (args.target_latitude, args.target_longitude, args.cone_half_angle)]

    times, records = [], []
    for frame in tqdm(load_frames(args.density_path, args.frames, args.dump_times), desc='frames'):
        times.append(frame.time)
        records.append(reduce_frame(frame, *direction))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    np.savez(args.out_file, times=np.array(times, dtype='datetime64[s]'), radius=RADIUS_GRID,
             target_latitude=direction[0], target_longitude=direction[1], cone_half_angle=direction[2],
             **{key: np.array([record[key] for record in records]) for key in records[0]})


if __name__ == '__main__':
    main()
