import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib import patheffects
from matplotlib.patches import Circle
from sunpy.coordinates import frames
from sunerf.data.horizons import get_horizons_coord
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

# --observer keywords: JPL Horizons targets
OBSERVERS = {'earth': '399', 'stereo_a': 'STEREO-A', 'psp': 'Parker Solar Probe', 'solo': 'Solar Orbiter'}
# distance panel: symmetric color range [R_sun], opacity from the log column density between these
# percentiles of the frame
DISTANCE_LIMIT = 30.0
DISTANCE_ALPHA_PERCENTILES = (5.0, 99.5)
# image panels: title and colormap
PANELS = {'tB': ('Total Brightness [MSB]', cm.soholasco2), 'pB': ('Polarized Brightness [MSB]', cm.soholasco2),
          'density': (r'$n_e$ [cm$^{-3}$]', 'RdPu'),
          'distance': (r'$x$ distance from plane of sky [R$_\odot$]', 'berlin'),
          'observer': ('Observer', None)}


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
    parser.add_argument('--observer', type=str, choices=sorted(OBSERVERS) + ['none'], action='append', default=None,
                        help='Real observer per waypoint: its ephemeris position (JPL Horizons) at the time of '
                             'every frame replaces --lat, --lon and --distance_au of the waypoint; the frames '
                             'between two different observers move from one to the other. none keeps the '
                             'manual position of the waypoint (HCI).')
    parser.add_argument('--resolution', type=int, default=256, help='Square output resolution in pixels')
    parser.add_argument('--dpi', type=int, default=300, help='Saved frame DPI')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames even if output files already exist')
    parser.add_argument('--panels', nargs='+', choices=sorted(PANELS),
                        default=['tB', 'pB', 'density', 'distance', 'observer'],
                        help='Panels of every frame, in this order')
    parser.add_argument('--brightness_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the tB and pB color scale [MSB]; default: 1st and 99.9th percentile '
                             'of the first frame')
    parser.add_argument('--density_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the density color scale [cm^-3]; default: percentiles of the first frame')
    return parser


def resolve_time_range(loader, time_range):
    observer_times = sorted({o['time'] for o in loader.observers})
    min_time = observer_times[0]
    max_time = observer_times[-1]
    if time_range is None:
        return min_time, max_time
    return time_range[0], time_range[-1]


def resolve_waypoints(args, default_start_time, default_end_time):
    # single defaults expand to every waypoint; the old default sweep was -90 to 270 deg in longitude
    lons = args.lon if args.lon is not None else ([-90.0, 270.0] if args.observer is None else [0.0])
    lats = args.lat if args.lat is not None else [0.0]
    times = args.time if args.time is not None else [default_start_time, default_end_time]
    steps_list = args.steps if args.steps is not None else [1, 80]
    radius_mins = args.radius_min if args.radius_min is not None else [2.5]
    radius_maxs = args.radius_max if args.radius_max is not None else [15.0]
    observers = [None if o == 'none' else o for o in args.observer] if args.observer is not None else [None]

    n_waypoints = max(len(lons), len(lats), len(times), len(steps_list), len(radius_mins), len(radius_maxs),
                      len(observers))

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
    observers = expand(observers, '--observer')

    waypoints = []
    for lon, lat, time, steps, radius_min, radius_max, observer in zip(
        lons, lats, times, steps_list, radius_mins, radius_maxs, observers
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
            'observer': observer,
        })
    if len(waypoints) < 1:
        raise ValueError('At least one waypoint is required')
    waypoints[0]['steps'] = 1
    return waypoints


def build_points(args, waypoints):
    """Frames: (distance, lat, lon, time, radius_min, radius_max, (observer of the segment start, observer of
    its end, fraction along the segment)); the observers are None for a manual position."""
    points = []
    first = waypoints[0]
    points.append((
        args.distance_au * u.AU,
        first['lat'] * u.deg,
        first['lon'] * u.deg,
        first['time'],
        first['radius_min'] * u.R_sun,
        first['radius_max'] * u.R_sun,
        (first['observer'], first['observer'], 1.0),
    ))

    for previous, current in zip(waypoints[:-1], waypoints[1:]):
        fractions = np.linspace(0.0, 1.0, current['steps'])
        segment_points = list(zip(
            np.full(current['steps'], args.distance_au) * u.AU,
            np.linspace(previous['lat'], current['lat'], current['steps']) * u.deg,
            np.linspace(previous['lon'], current['lon'], current['steps']) * u.deg,
            pd.date_range(start=previous['time'], end=current['time'], periods=current['steps']).to_pydatetime(),
            np.linspace(previous['radius_min'], current['radius_min'], current['steps']) * u.R_sun,
            np.linspace(previous['radius_max'], current['radius_max'], current['steps']) * u.R_sun,
            [(previous['observer'], current['observer'], f) for f in fractions],
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


def add_front_text(ax, x, y, z, text, **kwargs):
    label = ax.text(x, y, z, text, zorder=1000, **kwargs)
    label.set_path_effects([
        patheffects.withStroke(linewidth=2.5, foreground='white', alpha=0.85),
    ])
    return label


def plot_observer_geometry(ax, lat, lon, distance, time, earth_position=None, stereo_a_position=None):
    ax.computed_zorder = False
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
    labels = []
    if earth_position is not None:
        earth_lat, earth_lon, earth_distance = earth_position
        earth_display_radius = 1.08 + 0.08 * np.log10(max(earth_distance.to_value(u.AU), 1e-3) * 10)
        earth_x, earth_y, earth_z = spherical_to_cartesian(earth_display_radius, earth_lat, earth_lon)
        ax.scatter(earth_x, earth_y, earth_z, color='#b3d9ff', s=220, alpha=0.28, depthshade=False)
        ax.scatter(earth_x, earth_y, earth_z, color='dodgerblue', s=50, edgecolors='white', linewidths=0.6, depthshade=False)
        labels.append((earth_x, earth_y, earth_z, ' Earth', {'color': 'dodgerblue', 'fontsize': 9}))

    if stereo_a_position is not None:
        stereo_a_lat, stereo_a_lon, stereo_a_distance = stereo_a_position
        stereo_a_display_radius = 1.08 + 0.08 * np.log10(max(stereo_a_distance.to_value(u.AU), 1e-3) * 10)
        stereo_a_x, stereo_a_y, stereo_a_z = spherical_to_cartesian(stereo_a_display_radius, stereo_a_lat, stereo_a_lon)
        ax.scatter(stereo_a_x, stereo_a_y, stereo_a_z, color='#ffd8a8', s=220, alpha=0.28, depthshade=False)
        ax.scatter(stereo_a_x, stereo_a_y, stereo_a_z, color='darkorange', s=50, edgecolors='white', linewidths=0.6, depthshade=False)
        labels.append((stereo_a_x, stereo_a_y, stereo_a_z, ' STEREO-A', {'color': 'darkorange', 'fontsize': 9}))

    ax.plot([0, obs_x], [0, obs_y], [0, obs_z], color='red', linewidth=1.6, alpha=0.45, linestyle=':')
    ax.scatter(obs_x, obs_y, obs_z, color='#ffb3b3', s=220, alpha=0.28, depthshade=False)
    ax.scatter(obs_x, obs_y, obs_z, color='red', s=55, edgecolors='white', linewidths=0.6, depthshade=False)
    labels.append((obs_x, obs_y, obs_z - 0.08, 'SuNeRF', {'color': 'red', 'fontsize': 9, 'ha': 'center'}))

    for label_x, label_y, label_z, label_text, label_kwargs in labels:
        add_front_text(ax, label_x, label_y, label_z, label_text, **label_kwargs)

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


def ephemeris_positions(observer, times):
    """HCI latitude, longitude [deg] and distance [R_sun] of a real observer at ``times`` (one Horizons query)."""
    coords = get_horizons_coord(OBSERVERS[observer], times)
    hci = [coords[i].transform_to(frames.HeliocentricInertial(obstime=times[i])) for i in range(len(times))]
    return np.array([[c.lat.to_value(u.deg), c.lon.to_value(u.deg), c.distance.to_value(u.R_sun)] for c in hci])


def blend_positions(start, end, fraction):
    """Position between two observers: linear in latitude and distance, shortest arc in longitude."""
    delta_lon = (end[1] - start[1] + 180.0) % 360.0 - 180.0
    return np.array([start[0] + fraction * (end[0] - start[0]), start[1] + fraction * delta_lon,
                     start[2] + fraction * (end[2] - start[2])])


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


def log_density_alpha(density):
    """Opacity from the logarithm of the column density, 0 to 1 between two percentiles of the frame."""
    valid = np.isfinite(density) & (density > 0)
    log_density = np.full(density.shape, np.nan)
    log_density[valid] = np.log10(density[valid])
    low, high = np.nanpercentile(log_density, DISTANCE_ALPHA_PERCENTILES)
    return np.clip(np.nan_to_num((log_density - low) / (high - low), nan=0.0), 0.0, 1.0)


def frame_range(data, percentiles=(1.0, 99.9)):
    """Percentiles of the finite positive values of an image."""
    values = data[np.isfinite(data) & (data > 0)]
    return tuple(np.percentile(values, percentiles))


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
    # one color scale for the whole video: given ranges, or the percentiles of the first frame
    brightness_norm = LogNorm(*args.brightness_range) if args.brightness_range is not None else None
    density_norm = LogNorm(*args.density_range) if args.density_range is not None else None
    distance_norm = Normalize(vmin=-DISTANCE_LIMIT, vmax=DISTANCE_LIMIT)
    # ephemeris positions of every real observer at all frame times (one query per observer)
    observers = {o for point in points for o in point[6][:2] if o is not None}
    if observers:
        args.lon_frame = 'hci'
    ephemeris = {o: ephemeris_positions(o, [point[3] for point in points]) for o in observers}

    for i, (distance, lat, lon_value, time, occ_min, occ_max, segment) in tqdm(enumerate(points), total=len(points)):
        frame_path = os.path.join(args.out_path, f'cme_frame{i:03d}.jpg')
        if os.path.exists(frame_path) and not args.overwrite:
            print(f'Skipping existing frame: {frame_path}')
            continue

        start_observer, end_observer, fraction = segment
        if start_observer is None and end_observer is None:
            input_obs_coord = get_input_observer_coord(lat, lon_value, distance, time, args.lon_frame)
        else:
            manual = np.array([lat.to_value(u.deg), lon_value.to_value(u.deg), distance.to_value(u.R_sun)])
            start = manual if start_observer is None else ephemeris[start_observer][i]
            end = manual if end_observer is None else ephemeris[end_observer][i]
            position = end if start_observer == end_observer else blend_positions(start, end, fraction)
            distance = position[2] * u.R_sun
            input_obs_coord = get_input_observer_coord(position[0] * u.deg, position[1] * u.deg, distance, time,
                                                       'hci')

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

        maps = {'tB': model_out['tB_map'], 'pB': model_out['pB_map'], 'density': model_out['density_map'],
                'distance': model_out['plane_of_sky_offset_map']}  # MSB, cm^-3, R_sun
        pB_map = maps['pB']

        if brightness_norm is None:
            brightness_norm = LogNorm(*frame_range(pB_map.data))
        if density_norm is None:
            density_norm = LogNorm(*frame_range(maps['density'].data))

        n_cols = len(args.panels)
        fig = plt.figure(figsize=(3.4 * n_cols, 4.6), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)

        observer_position = (
            plot_obs_coord.lat.to(u.deg),
            plot_obs_coord.lon.to(u.deg),
            plot_distance,
        )
        print(
            f'[{time.isoformat(sep=" ", timespec="minutes")}] {args.lon_frame.upper()} | '
            f'{format_position("Observer", observer_position)}'
        )
        # image plane in projected solar radii: tangent-plane pixel offsets times the observer distance
        rsun_per_pixel = [float(scale.to_value(u.rad / u.pix)) * float(pB_map.dsun.to_value(u.R_sun))
                          for scale in pB_map.scale]
        center = [float(pB_map.reference_pixel.x.to_value(u.pix)), float(pB_map.reference_pixel.y.to_value(u.pix))]
        ny, nx = pB_map.data.shape
        extent = [(-0.5 - center[0]) * rsun_per_pixel[0], (nx - 0.5 - center[0]) * rsun_per_pixel[0],
                  (-0.5 - center[1]) * rsun_per_pixel[1], (ny - 0.5 - center[1]) * rsun_per_pixel[1]]

        first_image_ax = None
        for column, panel in enumerate(args.panels):
            if panel == 'observer':
                observer_ax = fig.add_subplot(1, n_cols, column + 1, projection='3d')
                earth_position = get_earth_position(time, args.lon_frame)
                stereo_a_position = get_stereo_a_position(time, args.lon_frame)
                plot_observer_geometry(observer_ax, plot_obs_coord.lat, plot_obs_coord.lon, plot_distance, time,
                                       earth_position=earth_position, stereo_a_position=stereo_a_position)
                continue
            title, cmap = PANELS[panel]
            # the image panels share the axes; y tick labels and label only on the first one
            ax = fig.add_subplot(1, n_cols, column + 1, sharex=first_image_ax, sharey=first_image_ax)
            # NaN (outside the field) is transparent on the white background
            colormap = plt.get_cmap(cmap).copy()
            colormap.set_bad((0.0, 0.0, 0.0, 0.0))
            if panel == 'distance':
                # opacity from the log column density
                norm = distance_norm
                rgba = colormap(norm(np.nan_to_num(maps[panel].data, nan=0.0)))
                rgba[..., 3] = log_density_alpha(maps['density'].data)
                ax.imshow(rgba, origin='lower', extent=extent)
                im = ScalarMappable(norm=norm, cmap=colormap)
            else:
                norm = density_norm if panel == 'density' else brightness_norm
                im = ax.imshow(maps[panel].data, cmap=colormap, origin='lower', extent=extent, norm=norm)
            ax.set_facecolor('white')
            ax.add_patch(Circle((0.0, 0.0), 1.0, color='orange', linewidth=0, zorder=5))
            ax.set_aspect('equal')
            # the colorbar sits above the panel and its label is the panel title
            colorbar = fig.colorbar(im, ax=ax, location='top', aspect=30, pad=0.02)
            colorbar.set_label(title)
            ax.set_xlabel(r'HPC-x [R$_\odot$]')
            if first_image_ax is None:
                first_image_ax = ax
                ax.set_ylabel(r'HPC-y [R$_\odot$]')
            else:
                ax.tick_params(labelleft=False)

        fig.suptitle(
            f'Lat. ({args.lon_frame.upper()}): {plot_obs_coord.lat.to_value(u.deg):.1f} deg, '
            f'Lon. ({args.lon_frame.upper()}): {plot_obs_coord.lon.to_value(u.deg):.1f} deg, '
            f'Time: {time.isoformat(" ", timespec="minutes")}',
            fontsize=16,
        )

        fig.savefig(frame_path, dpi=args.dpi)
        plt.close('all')


if __name__ == '__main__':
    main()
