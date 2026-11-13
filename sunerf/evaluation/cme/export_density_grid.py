"""Export the density of a SuNeRF reconstruction of observations on a spherical grid, one file per frame.

The grid covers the full sphere between the radii of ``--radius_range`` in the inertial (HCI) frame
of the model, with ``--radial_step`` and ``--angular_step``::

    <out_path>/rho000000.npz    density, background [cm^-3] and velocity [km/s] as
                                [latitude, longitude, radius], with the axes latitude and
                                longitude [deg, HCI], radius [R_sun] and the time [datetime64[s]]

The background is the quasi-static corona: the median of the reconstructed density over time at
every point of the grid that corotates with the Sun, taken from the exported period extended by
two days on both sides.  A transient crosses a point only for a small part of that period, so the
difference of the density to the background holds the CMEs.  (The static branch of the
split-temporal model is not used: its output alone is not a density of the corona, because the
dynamic branch is free to carry a constant offset.)  ``export_vtk`` renders cubes for ParaView.

Example::

    python -m sunerf.evaluation.cme.export_density_grid \
        --sunerf_path /path/save_state.snf --out_path /path/export \
        --time_range 2025-09-22T00:00:00 2025-09-23T06:00:00
"""
import argparse
import os
from datetime import timedelta

import numpy as np
from dateutil.parser import parse
from tqdm import tqdm

from sunerf.evaluation.cme.export_density import default_batch_size, model_reference_frame
from sunerf.evaluation.cme.export_layout import dump_name
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.train.coordinate_transformation import carrington_rotation_rate

# times of the background median and the extension of their period beyond the exported one
BACKGROUND_FRAMES = 15
BACKGROUND_MARGIN = timedelta(days=2)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--out_path', required=True, help='Output directory')
    parser.add_argument('--time_range', nargs=2, required=True, metavar=('START', 'END'), type=parse,
                        help='First and last exported time')
    parser.add_argument('--cadence_minutes', type=float, default=60.0, help='Time between the frames')
    parser.add_argument('--radius_range', type=float, nargs=2, default=(3.0, 100.0), help='Radial range [R_sun]')
    parser.add_argument('--radial_step', type=float, default=0.5, help='Radial grid spacing [R_sun]')
    parser.add_argument('--angular_step', type=float, default=1.0, help='Spacing in latitude and longitude [deg]')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Query points per forward pass (default: 2^18 per GPU, 2^16 without CUDA)')
    return parser


def frame_times(time_range, cadence_minutes):
    start, end = time_range
    n_frames = int(np.floor((end - start).total_seconds() / 60.0 / cadence_minutes + 1e-6)) + 1
    return [start + timedelta(minutes=i * cadence_minutes) for i in range(n_frames)]


def spherical_grid(radius_range, radial_step, angular_step):
    """Radius [R_sun], latitude and longitude [rad] of the cell centres of the full sphere."""
    radius = np.arange(radius_range[0], radius_range[1] + 1e-6, radial_step)
    latitude = np.deg2rad(np.arange(-90.0 + angular_step / 2, 90.0, angular_step))
    longitude = np.deg2rad(np.arange(0.0, 360.0 - 1e-9, angular_step))
    return radius, latitude, longitude


def spherical_points(radius, latitude, longitude):
    """Cartesian points [lat, lon, r, 3] of a spherical grid (R_sun, rad, rad; HCI)."""
    lat, lon, r = np.meshgrid(latitude, longitude, radius, indexing='ij')
    return np.stack([r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)], -1)


def evaluate(loader, points, time, batch_size):
    """Density [cm^-3] and radial velocity [km/s] at the finite Cartesian points [..., 3] in R_sun;
    NaN points stay NaN."""
    query = np.concatenate([points / loader.Rs_per_ds,
                            np.full(points.shape[:-1] + (1,), loader.normalize_datetime(time))], -1)
    output = loader.load_coords(query.astype(np.float32), batch_size=batch_size)
    r_hat = points / np.linalg.norm(points, axis=-1, keepdims=True)
    return output['rho'][..., 0], np.sum(output['v'] * r_hat, axis=-1)


def corotation_angle(time, reference_time):
    """Rotation of the Sun between ``reference_time`` and ``time`` in the inertial frame [rad]."""
    return float(carrington_rotation_rate()) * (time - reference_time).total_seconds()


def rotate_about_z(points, angle):
    """Points rotated by ``angle`` about the solar rotation axis (z of the HCI frame)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return np.stack([x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle), z], -1)


def corotating_background(loader, points, time_range, batch_size):
    """Median density over time at the points corotating with the Sun (positions at ``time_range[0]``)."""
    start, end = time_range[0] - BACKGROUND_MARGIN, time_range[1] + BACKGROUND_MARGIN
    times = [start + (end - start) * i / (BACKGROUND_FRAMES - 1) for i in range(BACKGROUND_FRAMES)]
    samples = [evaluate(loader, rotate_about_z(points, corotation_angle(time, time_range[0])), time, batch_size)[0]
               for time in tqdm(times, desc='background')]
    return np.median(np.stack(samples), axis=0)


def rotate_background(background, longitude, angle):
    """Corotating background ``[lat, lon, r]`` at the inertial longitudes of the grid after a rotation by ``angle``."""
    position = ((longitude - angle) % (2 * np.pi)) / (longitude[1] - longitude[0])
    lower = np.floor(position).astype(int) % longitude.size
    weight = (position - np.floor(position))[None, :, None]
    return background[:, lower] * (1 - weight) + background[:, (lower + 1) % longitude.size] * weight


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_path, exist_ok=True)

    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device)
    if model_reference_frame(loader) != 'inertial':
        raise ValueError('The grid export expects a model in the inertial frame.')
    batch_size = args.batch_size or default_batch_size(loader)
    radius, latitude, longitude = spherical_grid(args.radius_range, args.radial_step, args.angular_step)
    points = spherical_points(radius, latitude, longitude)
    times = frame_times(args.time_range, args.cadence_minutes)
    print(f'Exporting {len(times)} frames on {latitude.size} x {longitude.size} x {radius.size} (lat, lon, r) '
          f'points ({loader.device}, {batch_size} points per forward pass)')

    corotating = corotating_background(loader, points, args.time_range, batch_size)
    for index, time in enumerate(tqdm(times, desc='frames')):
        rho, velocity = evaluate(loader, points, time, batch_size)
        background = rotate_background(corotating, longitude, corotation_angle(time, args.time_range[0]))
        np.savez(os.path.join(args.out_path, f'{dump_name(index)}.npz'), density=rho.astype(np.float32),
                 background=background.astype(np.float32), velocity=velocity.astype(np.float32),
                 latitude=np.rad2deg(latitude), longitude=np.rad2deg(longitude), radius=radius,
                 time=np.datetime64(time, 's'), frame='inertial')


if __name__ == '__main__':
    main()
