"""Render a SuNeRF reconstruction of observations into VTK files for ParaView.

Independent of the evaluation export: the model is evaluated on a Cartesian cube of
``--pixels_per_solRad`` spanning the outer radius of ``--radius_range`` (0 outside the radial
range, only the points inside are evaluated; the files hold no NaN) at every frame of ``--time_range``.
One legacy binary VTK image-data file per frame, ``<out_path>/rho000000.vtk``, in HCI coordinates
[R_sun], with the fields

* ``ne_r2``: density compensated for the radial fall-off, ``r^2 n_e`` [cm^-3 R_sun^2],
* ``excess_r2``: background-subtracted compensated density ``r^2 (n_e - background)``,
* ``log10_ne``: log10 of the density [cm^-3],
* ``v_radial``: radial velocity [km/s],
* ``radial_distance``: distance from the center of the Sun [R_sun] (to color the CME front).

The background is the reconstruction at ``--background_time`` (before the CME), fixed in the
inertial frame for every frame.  It is written once to ``<out_path>/background.vtk`` (``ne_r2`` and
``log10_ne``).  With ``--latitude`` and ``--longitude`` (the CME direction, HCI) all fields of the
frames are 0 outside ``--half_angle`` in latitude and in longitude around it, so that the frames
hold only the CME.  Every observer of ``--observers`` (ephemeris position at the middle of
``--time_range``) is written to ``<out_path>/observer_<name>.vtk`` as one point at
``ARROW_BASE`` times the outer radius with the unit vector ``direction`` toward the observer.  In
ParaView, a Glyph filter (Arrow, orientation ``direction``, no scale array, scale factor
``(1 - ARROW_BASE)`` times the outer radius, 10 for 100 R_sun) draws the arrow up to the outer radius;
its size and thickness are set there.

Example::

    python -m sunerf.evaluation.cme.export_vtk --sunerf_path /path/save_state.snf --out_path /path/vtk \
        --time_range 2025-09-22T00:00:00 2025-09-23T06:00:00 --background_time 2025-09-21T18:00:00 \
        --latitude -25 --longitude 5 --half_angle 30 --observers earth stereo_a
"""
import argparse
import os

import numpy as np
from dateutil.parser import parse
from tqdm import tqdm

from sunerf.evaluation.cme.export_density import default_batch_size, model_reference_frame
from sunerf.evaluation.cme.export_density_grid import evaluate, frame_times
from sunerf.evaluation.cme.export_layout import dump_name
from sunerf.evaluation.cme.video import OBSERVERS, ephemeris_positions
from sunerf.evaluation.loader import ThomsonSuNeRFLoader

# base of the observer arrows in units of the outer radius of the cube
ARROW_BASE = 0.9

def cube_axis(max_radius, pixels_per_solRad):
    """Coordinate axis of the cube [R_sun], symmetric about the Sun."""
    return np.linspace(-max_radius, max_radius, int(round(2 * max_radius * pixels_per_solRad)) + 1)


def cube_points(axis):
    """Cartesian points of the cube [n, n, n, 3] and their radius."""
    points = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), -1)
    return points, np.linalg.norm(points, axis=-1)


def cme_box(points, latitude, longitude, half_angle):
    """Points within ``half_angle`` of the CME direction in latitude and in HCI longitude [deg]."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    point_latitude = np.degrees(np.arctan2(z, np.hypot(x, y)))
    offset_longitude = (np.degrees(np.arctan2(y, x)) - longitude + 180) % 360 - 180
    return (np.abs(point_latitude - latitude) <= half_angle) & (np.abs(offset_longitude) <= half_angle)


def vtk_fields(radius, rho, background, velocity):
    """Fields of a frame (see the module docstring)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return {'ne_r2': rho * radius ** 2, 'excess_r2': (rho - background) * radius ** 2,
                'log10_ne': np.log10(np.clip(rho, 1e-30, None)), 'v_radial': velocity,
                'radial_distance': np.where(np.isfinite(rho), radius, np.nan)}


def background_fields(radius, background):
    """Fields of the background file."""
    return {'ne_r2': background * radius ** 2, 'log10_ne': np.log10(np.clip(background, 1e-30, None))}


def write_vtk(path, axis, scalars):
    """Legacy binary VTK image data of ``scalars[name][x, y, z]`` on the cube ``axis`` [R_sun]; NaN is written as 0."""
    n, spacing = axis.size, axis[1] - axis[0]
    with open(path, 'wb') as f:
        f.write(b'# vtk DataFile Version 3.0\nSuNeRF density\nBINARY\nDATASET STRUCTURED_POINTS\n')
        f.write(f'DIMENSIONS {n} {n} {n}\nORIGIN {axis[0]} {axis[0]} {axis[0]}\n'.encode())
        f.write(f'SPACING {spacing} {spacing} {spacing}\nPOINT_DATA {n ** 3}\n'.encode())
        for name, values in scalars.items():
            f.write(f'SCALARS {name} float 1\nLOOKUP_TABLE default\n'.encode())
            # VTK runs over the first axis fastest
            f.write(np.nan_to_num(values, nan=0.0).transpose(2, 1, 0).astype('>f4').tobytes())
            f.write(b'\n')


def write_arrow_point(path, base, direction):
    """Legacy ASCII VTK polygon data of one vertex at ``base`` with the vector ``direction``."""
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nSuNeRF observer\nASCII\nDATASET POLYDATA\nPOINTS 1 float\n')
        f.write('{:.4f} {:.4f} {:.4f}\nVERTICES 1 2\n1 0\n'.format(*base))
        f.write('POINT_DATA 1\nVECTORS direction float\n{:.6f} {:.6f} {:.6f}\n'.format(*direction))


def write_observers(out_path, observers, time, outer_radius):
    """Arrow base and direction toward every observer at ``time`` (HCI)."""
    for name in observers:
        latitude, longitude, distance = ephemeris_positions(name, [time])[0]
        lat, lon = np.radians(latitude), np.radians(longitude)
        direction = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        write_arrow_point(os.path.join(out_path, f'observer_{name}.vtk'), ARROW_BASE * outer_radius * direction,
                          direction)
        print(f'Observer {name}: HCI latitude {latitude:.1f}, longitude {longitude:.1f} deg, {distance:.0f} R_sun')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--out_path', required=True, help='Output directory of the VTK files')
    parser.add_argument('--time_range', nargs=2, required=True, metavar=('START', 'END'), type=parse,
                        help='First and last frame')
    parser.add_argument('--cadence_minutes', type=float, default=60.0, help='Time between the frames')
    parser.add_argument('--radius_range', type=float, nargs=2, default=(3.0, 100.0), help='Radial range [R_sun]')
    parser.add_argument('--pixels_per_solRad', type=float, default=0.5, help='Resolution of the cube')
    parser.add_argument('--background_time', type=parse, required=True,
                        help='Background from the reconstruction at this time, before the CME')
    parser.add_argument('--latitude', type=float, default=None,
                        help='CME direction, latitude [deg]; frames are 0 outside the CME box (default: no box)')
    parser.add_argument('--longitude', type=float, default=None, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--half_angle', type=float, default=30.0,
                        help='Half width of the CME box in latitude and in longitude [deg]')
    parser.add_argument('--observers', nargs='+', choices=sorted(OBSERVERS), default=(),
                        help='Observers drawn as arrows pointing toward them')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Query points per forward pass (default: 2^18 per GPU, 2^16 without CUDA)')
    args = parser.parse_args(argv)
    if (args.latitude is None) != (args.longitude is None):
        parser.error('--latitude and --longitude are given together')
    os.makedirs(args.out_path, exist_ok=True)
    write_observers(args.out_path, args.observers, args.time_range[0] + (args.time_range[1] - args.time_range[0]) / 2,
                    args.radius_range[1])

    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device)
    if model_reference_frame(loader) != 'inertial':
        raise ValueError('The VTK export expects a model in the inertial frame.')
    batch_size = args.batch_size or default_batch_size(loader)
    axis = cube_axis(args.radius_range[1], args.pixels_per_solRad)
    points, radius = cube_points(axis)
    box = None if args.latitude is None else cme_box(points, args.latitude, args.longitude, args.half_angle)
    inside = (radius >= args.radius_range[0]) & (radius <= args.radius_range[1])
    points = np.where(inside[..., None], points, np.nan)
    times = frame_times(args.time_range, args.cadence_minutes)
    print(f'Rendering {len(times)} VTK frames on a {axis.size}^3 cube, {inside.sum()} points inside the radial '
          f'range ({loader.device})')

    background = evaluate(loader, points, args.background_time, batch_size)[0]
    write_vtk(os.path.join(args.out_path, 'background.vtk'), axis, background_fields(radius, background))

    for index, time in enumerate(tqdm(times, desc='vtk')):
        rho, velocity = evaluate(loader, points, time, batch_size)
        fields = vtk_fields(radius, rho, background, velocity)
        if box is not None:
            fields = {name: np.where(box, values, 0.0) for name, values in fields.items()}
        write_vtk(os.path.join(args.out_path, f'{dump_name(index)}.vtk'), axis, fields)


if __name__ == '__main__':
    main()
