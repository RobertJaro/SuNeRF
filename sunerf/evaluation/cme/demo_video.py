import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def build_segment(start_distance, end_distance,
                  start_lat, end_lat,
                  start_lon, end_lon,
                  start_time, end_time,
                  start_occ_min, end_occ_min,
                  start_occ_max, end_occ_max,
                  n_points):
    return list(zip(
        np.linspace(start_distance.to_value(u.AU), end_distance.to_value(u.AU), n_points) * u.AU,
        np.linspace(start_lat.to_value(u.deg), end_lat.to_value(u.deg), n_points) * u.deg,
        np.linspace(start_lon.to_value(u.deg), end_lon.to_value(u.deg), n_points) * u.deg,
        pd.date_range(start=start_time, end=end_time, periods=n_points),
        np.linspace(start_occ_min.to_value(u.R_sun), end_occ_min.to_value(u.R_sun), n_points) * u.R_sun,
        np.linspace(start_occ_max.to_value(u.R_sun), end_occ_max.to_value(u.R_sun), n_points) * u.R_sun,
    ))


def positive_lognorm(data):
    finite_positive = data[np.isfinite(data) & (data > 0)]
    if finite_positive.size == 0:
        return LogNorm(vmin=1e-12, vmax=1.0)

    vmin = finite_positive.min()
    vmax = finite_positive.max()
    if vmin == vmax:
        vmax = vmin * 10
    return LogNorm(vmin=vmin, vmax=vmax)


def spherical_to_cartesian(radius, lat, lon):
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def plot_observer_geometry(ax, lat, lon, distance, time):
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
    ax.plot([0, obs_x], [0, obs_y], [0, obs_z],
            color='red', linewidth=1.6, alpha=0.45, linestyle=':')
    ax.scatter(obs_x, obs_y, obs_z, color='#ffb3b3', s=220, alpha=0.28, depthshade=False)
    ax.scatter(obs_x, obs_y, obs_z, color='red', s=55, edgecolors='white', linewidths=0.6, depthshade=False)

    ax.set_title(time.isoformat(' ', timespec='minutes'))
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    panel_limit = 1.06
    ax.set_xlim(-panel_limit, panel_limit)
    ax.set_ylim(-panel_limit, panel_limit)
    ax.set_zlim(-panel_limit, panel_limit)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a CME demo video sequence')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, default=None, help='Path to output directory')
    parser.add_argument('--start_longitude', type=float, default=-90.0, help='Starting Carrington longitude in degrees')
    parser.add_argument('--start_latitude', type=float, default=0.0, help='Starting Carrington latitude in degrees')
    parser.add_argument('--start_distance', type=float, default=1.0, help='Starting observer distance in AU')
    parser.add_argument('--occ_range', type=float, nargs=2, default=[2.5, 15.0],
                        help='Initial occulting range in solar radii')
    parser.add_argument('--final_occ_max', type=float, default=100.0,
                        help='Final outer field-of-view radius in solar radii')
    parser.add_argument('--rotation_latitude', type=float, default=60.0,
                        help='Intermediate latitude reached during the frozen-time rotation')
    parser.add_argument('--rotation_longitude_shift', type=float, default=60.0,
                        help='Longitude shift applied during the frozen-time rotation')
    parser.add_argument('--polar_latitude', type=float, default=85.0,
                        help='Latitude used for the final polar view')
    parser.add_argument('--segment_points', type=int, nargs=6, default=[30, 30, 30, 30, 30, 30],
                        metavar=('SEG1', 'SEG2', 'SEG3', 'SEG4', 'SEG5', 'SEG6'),
                        help='Frame counts for the six path segments')
    parser.add_argument('--resolution', type=int, default=256, help='Square output resolution in pixels')
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'demo_video')
    os.makedirs(args.out_path, exist_ok=True)

    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    observer_times = sorted({o['time'] for o in sunerf_loader.observers})

    min_time = observer_times[0]
    max_time = observer_times[-1]
    timeline = max_time - min_time

    first_third_time = min_time + timeline / 3
    half_time = min_time + timeline / 2
    second_third_time = min_time + 2 * timeline / 3

    start_distance = args.start_distance * u.AU
    start_lat = args.start_latitude * u.deg
    start_lon = args.start_longitude * u.deg
    swept_lon = start_lon + 360 * u.deg
    rotated_lat = args.rotation_latitude * u.deg
    rotated_lon = swept_lon + args.rotation_longitude_shift * u.deg
    polar_lat = args.polar_latitude * u.deg

    occ_min = args.occ_range[0] * u.R_sun
    occ_max = args.occ_range[1] * u.R_sun
    final_occ_max = args.final_occ_max * u.R_sun
    resolution = (args.resolution, args.resolution) * u.pix

    segment_counts = args.segment_points
    segments = [
        build_segment(start_distance, start_distance,
                      start_lat, start_lat,
                      start_lon, start_lon,
                      first_third_time, half_time,
                      occ_min, occ_min,
                      occ_max, occ_max,
                      segment_counts[0]),
        build_segment(start_distance, start_distance,
                      start_lat, start_lat,
                      start_lon, swept_lon,
                      half_time, half_time,
                      occ_min, occ_min,
                      occ_max, occ_max,
                      segment_counts[1]),
        build_segment(start_distance, start_distance,
                      start_lat, rotated_lat,
                      swept_lon, rotated_lon,
                      half_time, half_time,
                      occ_min, occ_min,
                      occ_max, occ_max,
                      segment_counts[2]),
        build_segment(start_distance, start_distance,
                      rotated_lat, rotated_lat,
                      rotated_lon, rotated_lon,
                      half_time, second_third_time,
                      occ_min, occ_min,
                      occ_max, final_occ_max,
                      segment_counts[3]),
        build_segment(start_distance, start_distance,
                      rotated_lat, polar_lat,
                      rotated_lon, rotated_lon,
                      second_third_time, second_third_time,
                      occ_min, occ_min,
                      final_occ_max, final_occ_max,
                      segment_counts[4]),
        build_segment(start_distance, start_distance,
                      polar_lat, polar_lat,
                      rotated_lon, rotated_lon,
                      second_third_time, max_time,
                      occ_min, occ_min,
                      final_occ_max, final_occ_max,
                      segment_counts[5]),
    ]

    points = []
    for segment_index, segment in enumerate(segments):
        if segment_index > 0:
            segment = segment[1:]
        points.extend(segment)

    for i, (distance, lat, lon, time, occ_min_i, occ_max_i) in tqdm(enumerate(points), total=len(points)):
        obs_coord = SkyCoord(radius=distance, lat=lat, lon=lon,
                             frame=frames.HeliographicCarrington, obstime=time, observer='self')
        hci_lon = obs_coord.transform_to(frames.HeliocentricInertial).lon
        model_out = sunerf_loader.load_image(
            lat, hci_lon, time,
            distance=distance,
            resolution=resolution,
            occ_min=occ_min_i,
            occ_max=occ_max_i,
            progress=False,
        )

        pB_map = model_out['pB_map']
        density_map = model_out['density_map']
        pB_norm = positive_lognorm(pB_map.data)
        density_norm = positive_lognorm(density_map.data)

        fig = plt.figure(figsize=(10., 3.0))
        axs = [
            fig.add_subplot(1, 3, 2, projection=pB_map),
            fig.add_subplot(1, 3, 3, projection=pB_map),
        ]
        observer_ax = fig.add_subplot(1, 3, 1, projection='3d')
        fig.subplots_adjust(left=0.03, right=0.972, bottom=0.08, top=0.86, wspace=0.12)

        ax = axs[0]
        im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=pB_norm, origin='lower')
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
        ax.set_title(r'Density [cm$^{-3}$]')
        pB_map.draw_grid(ax, color='blue')
        ax.coords[0].set_axislabel(' ')
        ax.coords[1].set_axislabel(' ')
        ax.coords[0].set_ticklabel_visible(True)
        ax.coords[1].set_ticklabel_visible(False)

        plot_observer_geometry(observer_ax, lat, lon, distance, time)
        # fig.tight_layout()
        fig.savefig(os.path.join(args.out_path, f'cme_frame{i:03d}.jpg'), dpi=300)
        plt.close('all')
