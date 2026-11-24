"""Evolution of the primary CME in cuts: two planes through the CME axis and one spherical shell at five times.

Reads the spherical frames of ``export_density_grid``.  The
shown quantity is ``log10(n_e)`` (``COLOR_PERCENTILES`` of the shown values); the cuts share one
color scale over the rows, the shell map of every row has its own.  Rows are the frames closest to
``--times``, the first one the state before the CME; columns:

1. the meridional plane through the CME axis (``--latitude``, ``--longitude``, inertial HCI frame),
   north up, the half plane on the side of the CME;
2. the plane through the CME axis perpendicular to the first one, turned about the axis by
   ``--tilt`` (counterclockwise from the direction of increasing longitude toward north);
3. the shell through the CME as a latitude-longitude map centered on the CME longitude, with the
   traces of both planes.  The shell of the pre-CME row lies at the inner tracking boundary
   ``TRACK_MIN_RADIUS`` of ``plot_cme_propagation``; the shell of the other rows follows the CME:
   it is the radius of the maximum of the compensated density ``r^2 n_e`` averaged over the cone of
   ``AXIS_HALF_ANGLE`` around the CME axis, beyond ``TRACK_MIN_RADIUS``.

With ``--centroid``, each event row's planes intersect at the positive excess-density centroid
on its selected shell, inside a fixed 45-degree cone about the supplied axis. The pre-event row
uses the supplied axis. The shell radius is marked in the cuts.

Example::

    python -m sunerf.evaluation.cme.plot_cme_cuts --export_path /path/export --out_path /path/cme_cuts.jpg \
        --latitude -25 --longitude 5 \
        --times 2025-09-21T18:00 2025-09-22T04:00 2025-09-22T10:00 2025-09-22T16:00 2025-09-23T00:00
"""
import argparse
import glob
import os

import numpy as np
from dateutil.parser import parse
from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates

from sunerf.evaluation.cme.export_density_grid import corotation_angle, rotate_background
from sunerf.evaluation.cme.mass_transport import cone_mask
from sunerf.evaluation.cme.plot_cme_propagation import TRACK_MIN_RADIUS

# color range of log10(n_e)
COLOR_PERCENTILES = (1.0, 99.5)
# samples of the cut planes per axis
PLANE_SAMPLES = 400
# half angle of the cone around the CME axis that places the shell [deg]
AXIS_HALF_ANGLE = 10.0


def selected_frames(export_path, times):
    """Paths and actual timestamps of the nearest exported frames."""
    files = sorted(glob.glob(os.path.join(export_path, 'rho*.npz')))
    if not files:
        raise FileNotFoundError(f'No rho*.npz frames in {export_path}')
    frame_times = []
    for file in files:
        with np.load(file) as frame:
            frame_times.append(frame['time'])
    frame_times = np.array(frame_times, dtype='datetime64[s]')
    selected = [int(np.argmin(np.abs(frame_times - np.datetime64(time, 's')))) for time in times]
    return [(files[i], frame_times[i]) for i in selected]


def color_range(images):
    values = np.concatenate([np.ravel(image) for image in images])
    return np.nanpercentile(values[np.isfinite(values)], COLOR_PERCENTILES)


def sample(values, radius, latitude, longitude, points):
    """Linear interpolation of ``values[lat, lon, r]`` at Cartesian ``points[..., 3]`` [R_sun]; NaN outside
    the radial range."""
    r = np.linalg.norm(points, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(points[..., 2] / np.where(r > 0, r, 1.0), -1.0, 1.0)))
    lon = np.degrees(np.arctan2(points[..., 1], points[..., 0])) % 360.0
    # longitude periodic: the first column repeated after the last
    wrapped = np.concatenate([values, values[:, :1]], axis=1)
    index = np.stack([(lat - latitude[0]) / (latitude[1] - latitude[0]),
                      (lon - longitude[0]) % 360.0 / (longitude[1] - longitude[0]),
                      (r - radius[0]) / (radius[1] - radius[0])])
    result = map_coordinates(np.nan_to_num(wrapped, nan=0.0), index.reshape(3, -1), order=1, mode='nearest')
    result = result.reshape(r.shape)
    result[(r < radius[0]) | (r > radius[-1])] = np.nan
    return result


def plane_points(u, v, extent):
    """Half plane ``x u + y v`` with ``0 <= x <= extent``, ``|y| <= extent``: points ``[y, x, 3]`` and the axes."""
    x = np.linspace(0.0, extent, PLANE_SAMPLES // 2)
    y = np.linspace(-extent, extent, PLANE_SAMPLES)
    return x[None, :, None] * u + y[:, None, None] * v, x, y


def unit(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def cme_shell(density, radius, latitude, longitude, axis):
    """Index of the radius of the maximum of ``r^2 n_e`` averaged over the cone of ``AXIS_HALF_ANGLE``
    around ``axis``, beyond ``TRACK_MIN_RADIUS``."""
    lat, lon = np.meshgrid(latitude, longitude, indexing='ij')
    cone = np.tensordot(axis, unit(lat, lon), axes=1) >= np.cos(np.radians(AXIS_HALF_ANGLE))
    compensated = np.nanmean(density[cone], axis=0) * radius ** 2
    outer = np.flatnonzero(radius >= TRACK_MIN_RADIUS)
    return int(outer[np.nanargmax(compensated[outer])])


def shell_centroid(excess, latitude, longitude, seed_lat, seed_lon):
    """Area-weighted mean latitude and circular mean longitude of the positive shell excess."""
    lat, lon = np.radians(latitude), np.radians(longitude)
    mask = cone_mask(lat, lon, *np.radians((seed_lat, seed_lon, 45)))
    weights = np.where(mask & np.isfinite(excess) & (excess > 0), excess, 0.) * np.cos(lat[:, None])
    if weights.sum() <= 0:
        return None
    center_lat = np.average(latitude, weights=weights.sum(axis=1))
    center_lon = np.degrees(np.angle(np.sum(weights.sum(axis=0) * np.exp(1j * lon))))
    return center_lat, seed_lon + ((center_lon - seed_lon + 180.) % 360. - 180.)


def load_centroid_reference(path, radius, latitude, longitude):
    if path is None:
        return None
    with np.load(path) as ref:
        for key, grid in [('radius', radius), ('latitude', latitude), ('longitude', longitude)]:
            if ref[key].shape != grid.shape or not np.allclose(ref[key], grid):
                raise ValueError(f'Centroid reference {key} differs from the export grid')
        return ref['density'], ref['times'][-1].astype('datetime64[us]').item()


def frame_centroid(frame, density, shell, latitude, longitude, seed, reference):
    if reference is None:
        background = frame['background'][:, :, shell]
    else:
        field, epoch = reference
        time = frame['time'].astype('datetime64[us]').item()
        angle = corotation_angle(time, epoch)
        background = rotate_background(field[:, :, shell:shell + 1], np.radians(longitude), angle)[:, :, 0]
    return shell_centroid(density[:, :, shell] - background, latitude, longitude, *seed)


def cut_geometry(center_lat, center_lon, tilt, longitude):
    lon0 = np.radians(center_lon)
    meridional = (np.array([np.cos(lon0), np.sin(lon0), 0.0]), np.array([0.0, 0.0, 1.0]))
    axis = unit(center_lat, center_lon)
    east = np.array([-np.sin(lon0), np.cos(lon0), 0.0])
    north = np.cross(axis, east)
    tilt = np.radians(tilt)
    perpendicular = (axis, np.cos(tilt) * east + np.sin(tilt) * north)
    angle = np.radians(np.linspace(-90.0, 90.0, 361))
    points = np.cos(angle)[:, None] * perpendicular[0] + np.sin(angle)[:, None] * perpendicular[1]
    trace_lat = np.degrees(np.arcsin(np.clip(points[:, 2], -1.0, 1.0)))
    trace_lon = (np.degrees(np.arctan2(points[:, 1], points[:, 0])) - center_lon + 180.0) % 360.0 - 180.0
    traces = [(np.full(2, center_lon), np.array([-90.0, 90.0])), (trace_lon + center_lon, trace_lat)]

    # shell map centered on the CME longitude
    offset = (longitude - center_lon + 180.0) % 360.0 - 180.0
    order = np.argsort(offset)

    return (meridional, perpendicular), traces, center_lon + offset[order], order


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--export_path', required=True, help='Export directory of export_density_grid')
    parser.add_argument('--out_path', required=True, help='Output figure (jpg)')
    parser.add_argument('--latitude', type=float, required=True, help='CME direction, latitude [deg]')
    parser.add_argument('--longitude', type=float, required=True, help='CME direction, HCI longitude [deg]')
    parser.add_argument('--centroid', action='store_true', help='Center each event row on its shell excess centroid')
    parser.add_argument('--centroid_reference', help='Pre-event NPZ reference on the export grid, aligned to its '
                        'last reference timestamp. Defaults to the exported corotating background.')
    parser.add_argument('--tilt', type=float, default=0.0,
                        help='Rotation of the second cut about the CME axis, from the direction of increasing '
                             'longitude toward north [deg]')
    parser.add_argument('--times', nargs=5, type=parse, required=True,
                        help='Times of the five rows, the first one before the CME')
    args = parser.parse_args(argv)

    if args.centroid_reference and not args.centroid:
        parser.error('--centroid_reference requires --centroid')
    frames = selected_frames(args.export_path, args.times)
    with np.load(frames[0][0]) as first:
        radius, latitude, longitude = (first[key] for key in ('radius', 'latitude', 'longitude'))
    reference = load_centroid_reference(args.centroid_reference, radius, latitude, longitude)

    extent = radius[-1]
    seed = args.latitude, args.longitude
    rows = []
    cuts = []
    for row, (path, time) in enumerate(frames):
        center = seed
        center_label = 'Reference axis' if row == 0 or not args.centroid else 'Centroid'
        with np.load(path) as frame:
            density = frame['density']
            shell = (cme_shell(density, radius, latitude, longitude, unit(*seed)) if row else
                     int(np.argmin(np.abs(radius - TRACK_MIN_RADIUS))))
            if row and args.centroid:
                centroid = frame_centroid(frame, density, shell, latitude, longitude, seed, reference)
                if centroid is None:
                    center_label = 'Reference axis (no excess)'
                else:
                    center = centroid

        bases, traces, map_lon, order = cut_geometry(*center, args.tilt, longitude)
        with np.errstate(invalid='ignore', divide='ignore'):
            values = np.log10(density)
        images = []
        for u, v in bases:
            points, x, y = plane_points(u, v, extent)
            images.append(sample(values, radius, latitude, longitude, points))
        cuts.append(images)
        rows.append((time, shell, center, center_label, traces, map_lon, values[:, order, shell]))
        print(f'{time}  {radius[shell]:g} R_sun  {center_label}  lat {center[0]:+.2f}  lon {center[1]:+.2f}')
    cut_range = color_range([image for pair in cuts for image in pair])

    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4 * len(rows)), constrained_layout=True,
                             gridspec_kw={'width_ratios': (1.0, 1.0, 2.0)})
    circle = np.radians(np.linspace(0.0, 360.0, 361))
    for row, (time, shell, (center_lat, center_lon), center_label, traces, map_lon, shell_map) in enumerate(rows):
        for column, image in enumerate(cuts[row]):
            ax = axes[row, column]
            cut_image = ax.pcolormesh(x, y, image, cmap='plasma', vmin=cut_range[0], vmax=cut_range[1],
                                      shading='nearest')
            ax.plot(radius[shell] * np.cos(circle), radius[shell] * np.sin(circle), color='black', lw=0.6, ls='--')
            ax.add_patch(plt.Circle((0, 0), radius[0], color='0.3'))
            ax.set_xlim(0, extent)
            ax.set_ylim(-extent, extent)
            ax.set_aspect('equal')
        axes[row, 0].set_ylabel(f'{str(time)[:16].replace("T", " ")}\n\nnorth [R$_\\odot$]')
        axes[row, 1].set_ylabel(r'along the tilt [R$_\odot$]')

        ax = axes[row, 2]
        shell_range = color_range([shell_map])
        shell_image = ax.pcolormesh(map_lon, latitude, shell_map, cmap='plasma',
                                    vmin=shell_range[0], vmax=shell_range[1], shading='nearest')
        fig.colorbar(shell_image, ax=ax, pad=0.01, label=r'$\log_{10} n_e$ [cm$^{-3}$]')
        ax.set_title(f'Shell at {radius[shell]:g} R$_\\odot$', fontsize=10)
        if args.centroid:
            ax.set_title(f'Shell at {radius[shell]:g} R$_\\odot$   ·   {center_label} '
                         f'{center_lat:+.1f}° lat, {center_lon:+.1f}° lon', fontsize=10)
        for trace_lon, trace_lat in traces:
            ax.plot(trace_lon, trace_lat, color='black', lw=0.6, ls='--')
        ax.plot(center_lon, center_lat, marker='+', color='black', ms=10)
        ax.set_xlim(center_lon - 180., center_lon + 180.)
        ax.set_ylim(-90.0, 90.0)
        ax.set_aspect('equal')
        ax.set_ylabel('Latitude [deg]')
    axes[-1, 0].set_xlabel(r'horizontal toward the CME [R$_\odot$]')
    axes[-1, 1].set_xlabel(r'along the CME axis [R$_\odot$]')
    axes[-1, 2].set_xlabel('HCI longitude [deg]')
    axes[0, 0].set_title('Meridional cut' if args.centroid else f'Meridional cut, HCI longitude {args.longitude:g}°')
    axes[0, 1].set_title(f'Perpendicular cut through the CME axis, tilt {args.tilt:g}°')
    fig.colorbar(cut_image, ax=axes[:, :2], location='bottom', shrink=0.8, pad=0.01,
                 label=r'$\log_{10} n_e$ [cm$^{-3}$], cuts')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    fig.savefig(args.out_path, dpi=150)
    plt.close(fig)
    print(f'Figure written to {args.out_path}')


if __name__ == '__main__':
    main()
