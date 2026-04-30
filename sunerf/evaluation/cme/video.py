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
    parser.add_argument('--lon', type=float, action='append', default=None,
                        help='Longitude target in degrees; the first entry is the start value, later entries are interpolation targets')
    parser.add_argument('--lat', type=float, action='append', default=None,
                        help='Latitude target in degrees; the first entry is the start value, later entries are interpolation targets')
    parser.add_argument('--lon_frame', type=str, choices=['carrington', 'hci'], default='carrington',
                        help='Longitude frame used by --lon')
    parser.add_argument('--time', type=parse_time, action='append', default=None,
                        help='Time target in ISO format; the first entry is the start value, later entries are interpolation targets')
    parser.add_argument('--radius_min', type=float, action='append', default=None,
                        help='Inner projected radius target in solar radii; the first entry is the start value, later entries are interpolation targets')
    parser.add_argument('--radius_max', type=float, action='append', default=None,
                        help='Outer projected radius target in solar radii; the first entry is the start value, later entries are interpolation targets')
    parser.add_argument('--steps', type=int, action='append', default=None,
                        help='Frames per waypoint; first entry should normally be 1, later entries interpolate from the previous waypoint to the new target')
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
    return time_range[0], time_range[-1]


def resolve_waypoints(args, default_start_time, default_end_time):
    lons = args.lon if args.lon is not None else [-90.0, 270.0]
    lats = args.lat if args.lat is not None else [0.0, 0.0]
    times = args.time if args.time is not None else [default_start_time, default_end_time]
    steps_list = args.steps if args.steps is not None else [1, 80]
    radius_mins = args.radius_min if args.radius_min is not None else [2.5, 2.5]
    radius_maxs = args.radius_max if args.radius_max is not None else [15.0, 15.0]

    n_waypoints = max(len(lons), len(lats), len(times), len(steps_list), len(radius_mins), len(radius_maxs))

    def expand(values, name):
        if len(values) == 1:
            return values * n_waypoints
        if len(values) != n_waypoints:
            raise ValueError(f'{name} must be provided once or once per waypoint')
        return values

    lons = expand(lons, '--lon')
    lats = expand(lats, '--lat')
    times = expand(times, '--time')
    steps_list = expand(steps_list, '--steps')
    radius_mins = expand(radius_mins, '--radius_min')
    radius_maxs = expand(radius_maxs, '--radius_max')

    waypoints = []
    for lon, lat, time, steps, radius_min, radius_max in zip(
        lons, lats, times, steps_list, radius_mins, radius_maxs
    ):
        if steps < 1:
            raise ValueError('--steps values must be at least 1')
        waypoints.append({
            'lon': lon,
            'lat': lat,
            'time': time,
            'steps': steps,
            'radius_min': radius_min,
            'radius_max': radius_max,
        })
    if len(waypoints) < 1:
        raise ValueError('At least one waypoint is required')
    waypoints[0]['steps'] = 1
    return waypoints


def build_points(args, waypoints):
    points = []
    first = waypoints[0]
    points.append((
        args.distance_au * u.AU,
        first['lat'] * u.deg,
        first['lon'] * u.deg,
        first['time'],
        first['radius_min'] * u.R_sun,
        first['radius_max'] * u.R_sun,
    ))

    for previous, current in zip(waypoints[:-1], waypoints[1:]):
        segment_points = list(zip(
            np.full(current['steps'], args.distance_au) * u.AU,
            np.linspace(previous['lat'], current['lat'], current['steps']) * u.deg,
            np.linspace(previous['lon'], current['lon'], current['steps']) * u.deg,
            pd.date_range(start=previous['time'], end=current['time'], periods=current['steps']).to_pydatetime(),
            np.linspace(previous['radius_min'], current['radius_min'], current['steps']) * u.R_sun,
            np.linspace(previous['radius_max'], current['radius_max'], current['steps']) * u.R_sun,
        ))
        if segment_points:
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
    start_time, end_time = resolve_time_range(sunerf_loader, args.time)
    waypoints = resolve_waypoints(args, start_time, end_time)
    points = build_points(args, waypoints)

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
