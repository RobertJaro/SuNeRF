import argparse
import glob
import os
from datetime import timedelta

import numpy as np
import scipy
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from sunerf.data.date_util import normalize_datetime
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.evaluation.util import none_or_float
from sunerf.train.coordinate_transformation import spherical_to_cartesian


##########################################################
# utility functions

def longitude_mask(longitudes, min_lon, max_lon):
    """
    Create a mask for longitudes between min_lon and max_lon, handling wrap-around
    and supporting input longitudes in the -180 to 180 or 0 to 360 range.

    Parameters:
        longitudes (np.ndarray): Array of longitudes (can include negative values).
        min_lon (float): Minimum longitude of the desired range.
        max_lon (float): Maximum longitude of the desired range.

    Returns:
        np.ndarray: Boolean mask where True means the longitude is within the range.
    """
    # Normalize all longitudes and bounds to [0, 360)
    if min_lon is None and max_lon is None:
        return np.ones_like(longitudes, dtype=bool)
    norm_lons = np.mod(longitudes, 360)
    min_lon = min_lon % 360
    max_lon = max_lon % 360

    if min_lon <= max_lon:
        return (norm_lons >= min_lon) & (norm_lons <= max_lon)
    else:
        # Handle wrap-around case, e.g., min_lon=350, max_lon=10
        return (norm_lons >= min_lon) | (norm_lons <= max_lon)


def load_ref_file(Rs_per_ds, date0, file, max_latitude, max_longitude, max_radius, min_latitude, min_longitude,
                  min_radius):
    o = scipy.io.readsav(file)
    time = date0 + timedelta(hours=float(o['this_time']))
    original_time = time
    density = o['dens'].astype(np.float32).T
    ph = o['ph1d'].astype(np.float32)
    r = o['r1d'].astype(np.float32)
    th = o['th1d'].astype(np.float32) - np.pi / 2
    # create mask
    mask_r = (r < max_radius) & (r > min_radius)
    # mask longitude including wraparound
    ph_deg = np.rad2deg(ph)
    mask_ph = longitude_mask(ph_deg, min_lon=min_longitude, max_lon=max_longitude)
    mark_th = np.ones_like(th, dtype=bool)
    if min_latitude is not None:
        mark_th = mark_th & (th > np.deg2rad(min_latitude))
    if max_latitude is not None:
        mark_th = mark_th & (th < np.deg2rad(max_latitude))
    # apply mask
    r = r[mask_r]
    th = th[mark_th]
    ph = ph[mask_ph]
    density = density[mask_r, :, :]
    density = density[:, mark_th, :]
    density = density[:, :, mask_ph]
    rho_true = density
    # spherical coordinates
    radius, theta, phi, t = np.meshgrid(r, th, ph, np.array([time]), indexing="ij")
    spherical_coords = np.stack([radius, theta, phi], axis=-1)
    # convert to cartesian coordinates
    cartesian_coords = spherical_to_cartesian(spherical_coords)
    cartesian_coords = cartesian_coords / Rs_per_ds
    x, y, z = cartesian_coords[..., 0], cartesian_coords[..., 1], cartesian_coords[..., 2]
    # convert to query points
    return rho_true, spherical_coords


def plot_latitude_slice(rho, spherical_coords, title, img_path, target_latitude=0, target_longitude=135,
                        add_observers=True, plot_white_r_ticks=False):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]
    z = rho[:, lat_idx, :]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(4.5, 4.5))

    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    # fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]', orientation='horizontal', shrink=0.7)
    # ax.set_title(title, va='bottom')
    if min_longitude is not None:
        ax.set_xlim(np.deg2rad([min_longitude, max_longitude]))
    ax.set_axis_off()
    # ax.set_ticklabels([30, 60, 90, 120], ['', '', '', ''])
    # make r ticks white
    if plot_white_r_ticks:
        ax.tick_params(axis='y', which='major', labelcolor='white')
        ax.tick_params(axis='y', which='minor', labelcolor='white')
    ax.set_rlim(0, max_radius)
    # add arrows for observers
    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')


def plot_latitude_velocity_slice(velocity, spherical_coords, title, img_path, target_latitude=0, target_longitude=135,
                                 add_observers=True, plot_white_r_ticks=False):
    v_norm = Normalize(vmin=200, vmax=800)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]
    velocity = velocity[:, lat_idx, :, :]
    velocity_norm = np.linalg.norm(velocity, axis=-1)
    z = velocity_norm

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(4.5, 4.5))

    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=v_norm, cmap='cividis')
    fig.colorbar(pc, ax=ax, label=r'Velocity [km s$^{-1}$]', orientation='horizontal', shrink=0.7)
    # ax.set_title(title, va='bottom')
    if min_longitude is not None:
        ax.set_xlim(np.deg2rad([min_longitude, max_longitude]))
    # add dashed white line at longitude 135
    ax.plot(np.deg2rad([target_longitude, target_longitude]), np.array([min_radius, max_radius]), color='red',
            linestyle='--',
            linewidth=1)
    ax.set_rticks([30, 60, 90, 120])
    # make r ticks white
    if plot_white_r_ticks:
        ax.tick_params(axis='y', which='major', labelcolor='white')
        ax.tick_params(axis='y', which='minor', labelcolor='white')
    ax.set_rlim(0, max_radius)
    # add arrows for observers
    if add_observers:
        for observer in observers:
            obs_lon = observer['longitude'].to_value(u.rad) % (2 * np.pi)
            if max_longitude is not None and min_longitude is not None:
                arrow_mid_radius = max_radius if (np.rad2deg(obs_lon) < max_longitude) and (
                        np.rad2deg(obs_lon) > min_longitude) else max_radius / 2
                text_lon = (obs_lon + 0.12) if (np.rad2deg(obs_lon) < max_longitude) and (
                        np.rad2deg(obs_lon) > min_longitude) else (obs_lon + 0.25)
            else:
                arrow_mid_radius = max_radius
                text_lon = (obs_lon + 0.17)
            arrow_length = 20
            r_start = arrow_mid_radius + arrow_length / 2
            r_end = arrow_mid_radius - arrow_length
            dx = -arrow_length * np.cos(obs_lon)
            dy = -arrow_length * np.sin(obs_lon)
            arrowprops = dict(
                arrowstyle='->',
                color='cyan',
                linewidth=2,
            )
            ax.annotate(f'',
                        xy=(obs_lon, r_end),
                        xytext=(obs_lon, r_start),
                        arrowprops=arrowprops, annotation_clip=False)
            ax.annotate(f'{np.rad2deg(obs_lon):.0f}°',
                        xy=(obs_lon, arrow_mid_radius),
                        xytext=(text_lon, arrow_mid_radius - 4),
                        color='cyan', ha='center', va='center',
                        annotation_clip=False)

    quiver_vel = velocity[::8, ::3]
    quiver_vel[np.linalg.norm(quiver_vel, axis=-1) < 300] = np.nan
    ax.quiver(ph[::8, ::3], r[::8, ::3],
              quiver_vel[:, :, 0], quiver_vel[:, :, 1],
              scale=10000,
              color='white')

    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')


def plot_longitude_slice(rho, spherical_coords, title, img_path, target_latitude=0, target_longitude=135,
                         add_observers=True, ):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude)))

    r = spherical_coords[:, :, lon_idx, 0, 0]
    th = spherical_coords[:, :, lon_idx, 0, 1]
    z = rho[:, :, lon_idx]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(3.5, 3.5))

    pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]', orientation='horizontal', shrink=0.7)

    ax.set_rlim(0, max_radius)

    # ax.set_title(title, va='bottom')
    if add_observers:
        for observer in observers:
            obs_lat = observer['latitude'].to_value(u.rad)
            arrow_mid_radius = max_radius
            arrow_length = 24
            r_start = arrow_mid_radius + arrow_length / 2
            r_end = arrow_mid_radius - arrow_length
            dx = -arrow_length * np.cos(obs_lat)
            dy = -arrow_length * np.sin(obs_lat)
            arrowprops = dict(
                arrowstyle='->',
                color='cyan',
                linewidth=2,
            )
            ax.annotate(f'',
                        xy=(obs_lat, r_end),
                        xytext=(obs_lat, r_start),
                        arrowprops=arrowprops, annotation_clip=False)
            ax.annotate(f'{np.rad2deg(obs_lat):.0f}°',
                        xy=(obs_lat, arrow_mid_radius),
                        xytext=(obs_lat + 0.15, arrow_mid_radius - 5),
                        color='cyan', ha='center', va='center',
                        annotation_clip=False)

    if min_latitude is not None:
        ax.set_xlim(np.deg2rad([min_latitude, max_latitude]))
    # add dashed white line at latitude 0
    ax.plot(np.deg2rad([target_latitude, target_latitude]), np.array([min_radius, max_radius]), color='white',
            linestyle='--', linewidth=1)
    ax.set_rticks([30, 60, 90, 120])

    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')


def plot_longitude_velocity_slice(velocity, spherical_coords, title, img_path, target_latitude=0, target_longitude=135,
                                  add_observers=True, ):
    v_norm = Normalize(vmin=200, vmax=800)

    lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude)))

    r = spherical_coords[:, :, lon_idx, 0, 0]
    th = spherical_coords[:, :, lon_idx, 0, 1]
    velocity = velocity[:, :, lon_idx, :]
    velocity_norm = np.linalg.norm(velocity, axis=-1)
    z = velocity_norm

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(3.5, 3.5))

    pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=v_norm, cmap='cividis')
    fig.colorbar(pc, ax=ax, label=r'Velocity [km s$^{-1}$]', orientation='horizontal', shrink=0.7)

    ax.set_rlim(0, max_radius)

    # ax.set_title(title, va='bottom')
    if add_observers:
        for observer in observers:
            obs_lat = observer['latitude'].to_value(u.rad)
            arrow_mid_radius = max_radius
            arrow_length = 24
            r_start = arrow_mid_radius + arrow_length / 2
            r_end = arrow_mid_radius - arrow_length
            dx = -arrow_length * np.cos(obs_lat)
            dy = -arrow_length * np.sin(obs_lat)
            arrowprops = dict(
                arrowstyle='->',
                color='cyan',
                linewidth=2,
            )
            ax.annotate(f'',
                        xy=(obs_lat, r_end),
                        xytext=(obs_lat, r_start),
                        arrowprops=arrowprops, annotation_clip=False)
            ax.annotate(f'{np.rad2deg(obs_lat):.0f}°',
                        xy=(obs_lat, arrow_mid_radius),
                        xytext=(obs_lat + 0.15, arrow_mid_radius - 5),
                        color='cyan', ha='center', va='center',
                        annotation_clip=False)

    if min_latitude is not None:
        ax.set_xlim(np.deg2rad([min_latitude, max_latitude]))
    # add dashed white line at latitude 0
    ax.plot(np.deg2rad([target_latitude, target_latitude]), np.array([min_radius, max_radius]), color='red',
            linestyle='--', linewidth=1)
    ax.set_rticks([30, 60, 90, 120])

    quiver_vel = velocity[::8, ::3]
    quiver_vel[np.linalg.norm(quiver_vel, axis=-1) < 300] = np.nan
    ax.quiver(th[::8, ::3], r[::8, ::3],
              quiver_vel[:, :, 1], quiver_vel[:, :, 2],
              scale=10000, color='white')

    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate CME parameters')
    parser.add_argument('--data_path', type=str, required=True, help='Path to density cube data files')
    parser.add_argument('--out_path', type=str, help='Path to output directory')
    parser.add_argument('--max_radius', type=float, default=120.0, help='Maximum radius in solar radii')
    parser.add_argument('--min_radius', type=float, default=30.0, help='Minimum radius in solar radii')
    parser.add_argument('--min_longitude', type=none_or_float, default=None, help='Minimum longitude in degrees')
    parser.add_argument('--max_longitude', type=none_or_float, default=None, help='Maximum longitude in degrees')
    parser.add_argument('--min_latitude', type=none_or_float, default=None, help='Minimum latitude in degrees')
    parser.add_argument('--max_latitude', type=none_or_float, default=None, help='Maximum latitude in degrees')
    parser.add_argument('--plot_ground_truth', action='store_true', help='Plot ground truth data')
    parser.add_argument('--plot_white_r_ticks', action='store_true', help='Plot white ticks for r axis')

    args = parser.parse_args()

    # set default path
    os.makedirs(args.out_path, exist_ok=True)


    ##########################################################
    max_radius = args.max_radius
    min_radius = args.min_radius
    min_longitude = args.min_longitude
    max_longitude = args.max_longitude
    min_latitude = args.min_latitude
    max_latitude = args.max_latitude
    Rs_per_ds = 1

    date0 = parse("2010-04-03T09:04:00.000")
    files = sorted(glob.glob(args.data_path))

    target_latitude = 0
    target_longitude = 135

    background_files = files[:7]
    foreground_files = files[10:40]

    print('Loading CME files')
    for i, file in enumerate(foreground_files):
        ##########################################################
        # load data
        rho_true, spherical_coords  = load_ref_file(Rs_per_ds, date0,
                                                                                         file, max_latitude,
                                                                                         max_longitude,
                                                                                         max_radius,
                                                                                         min_latitude,
                                                                                         min_longitude,
                                                                                         min_radius)

        plot_latitude_slice(rho_true, spherical_coords,
                            title=r"Ground-truth $\theta=0^\cdot$",
                            img_path=os.path.join(args.out_path, f"gt_lat_{i:03d}.png"), add_observers=False,
                            plot_white_r_ticks=args.plot_white_r_ticks)

        plot_longitude_slice(rho_true, spherical_coords,
                             title=r"Ground-truth $\phi=135^\circ$",
                             img_path=os.path.join(args.out_path, f"gt_lon_{i:03d}.png"), add_observers=False)
