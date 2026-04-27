import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def parse_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def build_parser():
    parser = argparse.ArgumentParser(
        description='Create a generic CME video sweep.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, default=None, help='Path to output directory')
    parser.add_argument('--lon_range', type=float, nargs=2, metavar=('START_DEG', 'END_DEG'),
                        action='append', default=None,
                        help='Longitude sweep in degrees; repeat for multiple path segments')
    parser.add_argument('--lat_range', type=float, nargs=2, metavar=('START_DEG', 'END_DEG'),
                        action='append', default=None,
                        help='Carrington latitude sweep in degrees; repeat for multiple path segments')
    parser.add_argument('--lon_frame', type=str, choices=['carrington', 'hci'], default='carrington',
                        help='Longitude frame used by --lon_range')
    parser.add_argument('--time_range', type=parse_time, nargs=2, metavar=('START', 'END'),
                        action='append', default=None,
                        help='Time sweep in ISO format; repeat for multiple path segments')
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument('--occ_range', type=float, nargs=2, metavar=('INNER_RS', 'OUTER_RS'),
                             action='append', default=None,
                             help='Occulting range in solar radii; repeat once per segment if needed')
    range_group.add_argument('--radius_range', type=float, nargs=2, metavar=('INNER_RS', 'OUTER_RS'),
                             action='append', default=None,
                             help='Projected plane-of-sky radial range in solar radii; repeat once per segment if needed')
    parser.add_argument('--steps', type=int, action='append', default=None,
                        help='Number of frames in each segment; repeat to match the number of path segments')
    parser.add_argument('--distance_au', type=float, default=1.0, help='Observer distance in AU')
    parser.add_argument('--resolution', type=int, default=256, help='Square output resolution in pixels')
    parser.add_argument('--dpi', type=int, default=300, help='Saved frame DPI')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames even if output files already exist')
    return parser


def resolve_time_range(loader, time_range):
    observer_times = sorted({o['time'] for o in loader.observers})
    min_time = observer_times[0]
    max_time = observer_times[-1]
    if time_range is None:
        return min_time, max_time
    if isinstance(time_range[0], (list, tuple)):
        return time_range[0][0], time_range[0][1]
    return time_range[0], time_range[1]


def resolve_segments(args, default_start_time, default_end_time):
    lon_ranges = args.lon_range if args.lon_range is not None else [[-90.0, 270.0]]
    lat_ranges = args.lat_range if args.lat_range is not None else [[0.0, 0.0]]
    time_ranges = args.time_range if args.time_range is not None else [[default_start_time, default_end_time]]
    steps_list = args.steps if args.steps is not None else [80]
    if args.radius_range is not None:
        projected_ranges = args.radius_range
    elif args.occ_range is not None:
        projected_ranges = args.occ_range
    else:
        projected_ranges = [[2.5, 15.0]]

    n_segments = max(len(lon_ranges), len(lat_ranges), len(time_ranges), len(steps_list), len(projected_ranges))

    def expand(values, name):
        if len(values) == 1:
            return values * n_segments
        if len(values) != n_segments:
            raise ValueError(f'{name} must be provided once or once per segment')
        return values

    lon_ranges = expand(lon_ranges, '--lon_range')
    lat_ranges = expand(lat_ranges, '--lat_range')
    time_ranges = expand(time_ranges, '--time_range')
    steps_list = expand(steps_list, '--steps')
    projected_ranges = expand(projected_ranges, '--radius_range/--occ_range')

    segments = []
    for lon_range, lat_range, time_range, steps, projected_range in zip(
        lon_ranges, lat_ranges, time_ranges, steps_list, projected_ranges
    ):
        if steps < 1:
            raise ValueError('--steps values must be at least 1')
        segments.append({
            'lon_range': lon_range,
            'lat_range': lat_range,
            'time_range': time_range,
            'steps': steps,
            'projected_range': projected_range,
        })
    return segments


def build_points(args, segments):
    points = []
    for segment_index, segment in enumerate(segments):
        segment_points = list(zip(
            np.full(segment['steps'], args.distance_au) * u.AU,
            np.linspace(segment['lat_range'][0], segment['lat_range'][1], segment['steps']) * u.deg,
            np.linspace(segment['lon_range'][0], segment['lon_range'][1], segment['steps']) * u.deg,
            pd.date_range(
                start=segment['time_range'][0],
                end=segment['time_range'][1],
                periods=segment['steps'],
            ).to_pydatetime(),
            np.full(segment['steps'], segment['projected_range'][0]) * u.R_sun,
            np.full(segment['steps'], segment['projected_range'][1]) * u.R_sun,
        ))
        if segment_index > 0 and segment_points:
            segment_points = segment_points[1:]
        points.extend(segment_points)
    return points


def spherical_to_cartesian(radius, lat, lon):
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def plot_observer_geometry(ax, lat, lon, distance, time, earth_position=None, stereo_a_position=None):
    sphere_u = np.linspace(0, 2 * np.pi, 120)
    sphere_v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color='0.75', linewidth=0.45, alpha=0.7)
    ax.scatter(0, 0, 0, color='#ffd54a', s=240, edgecolors='#ffb300', linewidths=0.9, depthshade=False)

    distance_au = distance.to_value(u.AU)
    display_radius = 1.08 + 0.08 * np.log10(max(distance_au, 1e-3) * 10)
    obs_x, obs_y, obs_z = spherical_to_cartesian(display_radius, lat, lon)
    if earth_position is not None:
        earth_lat, earth_lon, earth_distance = earth_position
        earth_display_radius = 1.08 + 0.08 * np.log10(max(earth_distance.to_value(u.AU), 1e-3) * 10)
        earth_x, earth_y, earth_z = spherical_to_cartesian(earth_display_radius, earth_lat, earth_lon)
        ax.scatter(earth_x, earth_y, earth_z, color='#b3d9ff', s=220, alpha=0.28, depthshade=False)
        ax.scatter(earth_x, earth_y, earth_z, color='dodgerblue', s=50, edgecolors='white', linewidths=0.6, depthshade=False)
        ax.text(earth_x, earth_y, earth_z, ' Earth', color='dodgerblue', fontsize=9)

    if stereo_a_position is not None:
        stereo_a_lat, stereo_a_lon, stereo_a_distance = stereo_a_position
        stereo_a_display_radius = 1.08 + 0.08 * np.log10(max(stereo_a_distance.to_value(u.AU), 1e-3) * 10)
        stereo_a_x, stereo_a_y, stereo_a_z = spherical_to_cartesian(stereo_a_display_radius, stereo_a_lat, stereo_a_lon)
        ax.scatter(stereo_a_x, stereo_a_y, stereo_a_z, color='#ffd8a8', s=220, alpha=0.28, depthshade=False)
        ax.scatter(stereo_a_x, stereo_a_y, stereo_a_z, color='darkorange', s=50, edgecolors='white', linewidths=0.6, depthshade=False)
        ax.text(stereo_a_x, stereo_a_y, stereo_a_z, ' STEREO-A', color='darkorange', fontsize=9)

    ax.plot([0, obs_x], [0, obs_y], [0, obs_z], color='red', linewidth=1.6, alpha=0.45, linestyle=':')
    ax.scatter(obs_x, obs_y, obs_z, color='#ffb3b3', s=220, alpha=0.28, depthshade=False)
    ax.scatter(obs_x, obs_y, obs_z, color='red', s=55, edgecolors='white', linewidths=0.6, depthshade=False)
    ax.text(obs_x, obs_y, obs_z - 0.08, 'SuNeRF', color='red', fontsize=9, ha='center')

    ax.set_title(time.isoformat(' ', timespec='minutes'))
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    panel_limit = 1.06
    ax.set_xlim(-panel_limit, panel_limit)
    ax.set_ylim(-panel_limit, panel_limit)
    ax.set_zlim(-panel_limit, panel_limit)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()


def get_stereo_a_position(obstime, lon_frame):
    stereo_a = get_horizons_coord('STEREO-A', obstime)
    if lon_frame == 'hci':
        stereo_a = stereo_a.transform_to(frames.HeliocentricInertial(obstime=obstime))
        return stereo_a.lat.to(u.deg), stereo_a.lon.to(u.deg), stereo_a.distance.to(u.AU)

    stereo_a = stereo_a.transform_to(frames.HeliographicCarrington(observer='self', obstime=obstime))
    return stereo_a.lat.to(u.deg), stereo_a.lon.to(u.deg), stereo_a.radius.to(u.AU)


def get_earth_position(obstime, lon_frame):
    earth = get_body_heliographic_stonyhurst('earth', obstime)
    if lon_frame == 'hci':
        earth = earth.transform_to(frames.HeliocentricInertial(obstime=obstime))
        return earth.lat.to(u.deg), earth.lon.to(u.deg), earth.distance.to(u.AU)

    earth = earth.transform_to(frames.HeliographicCarrington(observer='self', obstime=obstime))
    return earth.lat.to(u.deg), earth.lon.to(u.deg), earth.radius.to(u.AU)


def get_input_observer_coord(lat, lon, distance, obstime, lon_frame):
    if lon_frame == 'hci':
        return SkyCoord(
            lat=lat,
            lon=lon,
            distance=distance,
            frame=frames.HeliocentricInertial,
            obstime=obstime,
        )

    return SkyCoord(
        lat=lat,
        lon=lon,
        radius=distance,
        frame=frames.HeliographicCarrington,
        obstime=obstime,
        observer='self',
    )


def format_position(label, coord):
    lat, lon, distance = coord
    return (
        f'{label}: '
        f'lat={lat.to_value(u.deg):7.3f} deg, '
        f'lon={lon.to_value(u.deg):8.3f} deg, '
        f'r={distance.to_value(u.AU):6.3f} AU'
    )


def main():
    args = build_parser().parse_args()
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'video')
    os.makedirs(args.out_path, exist_ok=True)

    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    start_time, end_time = resolve_time_range(sunerf_loader, args.time_range)
    segments = resolve_segments(args, start_time, end_time)
    points = build_points(args, segments)

    resolution = (args.resolution, args.resolution) * u.pix

    brightness_norm = LogNorm()
    density_norm = LogNorm()

    for i, (distance, lat, lon_value, time, occ_min, occ_max) in tqdm(enumerate(points), total=len(points)):
        frame_path = os.path.join(args.out_path, f'cme_frame{i:03d}.jpg')
        if os.path.exists(frame_path) and not args.overwrite:
            print(f'Skipping existing frame: {frame_path}')
            continue

        input_obs_coord = get_input_observer_coord(lat, lon_value, distance, time, args.lon_frame)

        if args.lon_frame == 'hci':
            hci_obs_coord = input_obs_coord
            plot_obs_coord = input_obs_coord
            plot_distance = input_obs_coord.distance.to(u.AU)
        else:
            hci_obs_coord = input_obs_coord.transform_to(frames.HeliocentricInertial(obstime=time))
            plot_obs_coord = input_obs_coord.transform_to(
                frames.HeliographicCarrington(observer='self', obstime=time)
            )
            plot_distance = plot_obs_coord.radius.to(u.AU)

        model_out = sunerf_loader.load_image(
            hci_obs_coord.lat,
            hci_obs_coord.lon,
            time,
            distance=distance,
            resolution=resolution,
            occ_min=occ_min,
            occ_max=occ_max,
            progress=False,
        )

        pB_map = model_out['pB_map']
        density_map = model_out['density_map']

        fig = plt.figure(figsize=(12, 4))
        observer_ax = fig.add_subplot(1, 3, 1, projection='3d')
        axs = [
            fig.add_subplot(1, 3, 2, projection=pB_map),
            fig.add_subplot(1, 3, 3, projection=pB_map),
        ]

        earth_position = get_earth_position(time, args.lon_frame)
        stereo_a_position = get_stereo_a_position(time, args.lon_frame)
        observer_position = (
            plot_obs_coord.lat.to(u.deg),
            plot_obs_coord.lon.to(u.deg),
            plot_distance,
        )
        print(
            f'[{time.isoformat(sep=" ", timespec="minutes")}] {args.lon_frame.upper()} | '
            f'{format_position("Observer", observer_position)} | '
            f'{format_position("Earth", earth_position)} | '
            f'{format_position("STEREO-A", stereo_a_position)}'
        )
        plot_observer_geometry(observer_ax, plot_obs_coord.lat, plot_obs_coord.lon, plot_distance, time,
                               earth_position=earth_position, stereo_a_position=stereo_a_position)

        ax = axs[0]
        im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax)
        ax.set_title('Polarized Brightness [MSB]')
        pB_map.draw_grid(ax, color='blue')
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')

        ax = axs[1]
        im = ax.imshow(density_map.data, cmap='RdPu', origin='lower', norm=density_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax)
        ax.set_title(r'$n_e$ [cm$^{-3}$]')
        pB_map.draw_grid(ax, color='blue')
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')

        fig.suptitle(
            f'Latitude ({args.lon_frame.upper()}): {plot_obs_coord.lat.to_value(u.deg):.1f} deg, '
            f'Longitude ({args.lon_frame.upper()}): {plot_obs_coord.lon.to_value(u.deg):.1f} deg, '
            f'Time: {time.isoformat(" ", timespec="minutes")}',
            fontsize=16,
        )

        fig.tight_layout()
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close('all')


if __name__ == '__main__':
    main()
