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
from nf2.data.util import cartesian_to_spherical
from sklearn.linear_model import LinearRegression

from sunerf.data.date_util import normalize_datetime
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.evaluation.util import none_or_float
from sunerf.train.coordinate_transformation import spherical_to_cartesian


##########################################################
# utility functions
def compute_center_of_mass(density, coords):
    spherical_coords = cartesian_to_spherical(coords)
    r, theta, phi = spherical_coords[..., 0], spherical_coords[..., 1], spherical_coords[..., 2]
    dr = np.gradient(r, axis=0)
    area_element = r ** 2 * np.sin(theta)
    total_mass = np.nansum(density * area_element * dr)
    center_of_mass = np.nansum(density[..., None] * area_element[..., None] * coords * dr[..., None],
                               axis=(0, 1, 2)) / total_mass
    return center_of_mass

def compute_total_mass(density, coords):
    spherical_coords = cartesian_to_spherical(coords)
    r, theta, phi = spherical_coords[..., 0], spherical_coords[..., 1], spherical_coords[..., 2]
    dr = np.gradient(r, axis=0)
    area_element = r ** 2 * np.sin(theta)
    cm_per_Rs = (1 * u.Rsun).to_value(u.cm)
    total_mass = np.nansum(density * area_element * dr) * cm_per_Rs ** 3
    return total_mass


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
                  min_radius, ref_date, seconds_per_dt):
    o = scipy.io.readsav(file)
    time = date0 + timedelta(hours=float(o['this_time']))
    original_time = time
    time = normalize_datetime(time, seconds_per_dt, ref_date)
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
    query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)
    return cartesian_coords, original_time, query_points, rho_true, spherical_coords


def compute_velocity(times, radius):
    """Compute CME velocity from time series data using linear regression."""
    time_seconds = np.array([(t - times[0]).total_seconds() for t in times])
    #
    velocity_model = LinearRegression()
    velocity_model.fit(time_seconds.reshape(-1, 1), radius)
    velocity = velocity_model.coef_[0] * u.Rsun / u.s
    velocity_km_s = velocity.to_value(u.km / u.s)
    #
    return velocity_km_s


def plot_latitude_slice(rho, spherical_coords, title, img_path, target_latitude=0, target_longitude=135,
                        add_observers=True, plot_white_r_ticks=False):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]
    z = rho[:, lat_idx, :]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(4.5, 4.5))

    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    fig.colorbar(pc, ax=ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]', orientation='horizontal', shrink=0.7)
    # ax.set_title(title, va='bottom')
    if min_longitude is not None:
        ax.set_xlim(np.deg2rad([min_longitude, max_longitude]))
    # add dashed white line at longitude 135
    ax.plot(np.deg2rad([target_longitude, target_longitude]), np.array([min_radius, max_radius]), color='white',
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
                arrow_mid_radius = max_radius if (np.rad2deg(obs_lon) < max_longitude) and (np.rad2deg(obs_lon) > min_longitude) else max_radius / 2
                text_lon = (obs_lon + 0.12) if (np.rad2deg(obs_lon) < max_longitude) and (np.rad2deg(obs_lon) > min_longitude) else (obs_lon + 0.25)
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
                arrow_mid_radius = max_radius if (np.rad2deg(obs_lon) < max_longitude) and (np.rad2deg(obs_lon) > min_longitude) else max_radius / 2
                text_lon = (obs_lon + 0.12) if (np.rad2deg(obs_lon) < max_longitude) and (np.rad2deg(obs_lon) > min_longitude) else (obs_lon + 0.25)
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
    ax.plot(np.deg2rad([target_latitude, target_latitude]), np.array([min_radius, max_radius]), color='white', linestyle='--', linewidth=1)
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
    ax.plot(np.deg2rad([target_latitude, target_latitude]), np.array([min_radius, max_radius]), color='red', linestyle='--', linewidth=1)
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
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory')
    parser.add_argument('--max_radius', type=float, default=120.0, help='Maximum radius in solar radii')
    parser.add_argument('--min_radius', type=float, default=30.0, help='Minimum radius in solar radii')
    parser.add_argument('--min_longitude', type=none_or_float, default=90 - 20, help='Minimum longitude in degrees')
    parser.add_argument('--max_longitude', type=none_or_float, default=180 + 20, help='Maximum longitude in degrees')
    parser.add_argument('--min_latitude', type=none_or_float, default=-60.0, help='Minimum latitude in degrees')
    parser.add_argument('--max_latitude', type=none_or_float, default=60.0, help='Maximum latitude in degrees')
    parser.add_argument('--plot_ground_truth', action='store_true', help='Plot ground truth data')
    parser.add_argument('--plot_white_r_ticks', action='store_true', help='Plot white ticks for r axis')
    parser.add_argument('--plot_velocity', action='store_true', help='Plot velocity data')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'evaluation')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    ##########################################################
    max_radius = args.max_radius
    min_radius = args.min_radius
    min_longitude = args.min_longitude
    max_longitude = args.max_longitude
    min_latitude = args.min_latitude
    max_latitude = args.max_latitude
    Rs_per_ds = sunerf_loader.Rs_per_ds


    times = []
    center_of_mass_true = []
    center_of_mass_pred = []
    center_of_mass_diff = []
    shock_front_true = []
    shock_front_pred = []
    mass_true = []
    mass_pred = []

    date0 = parse("2010-04-03T09:04:00.000")
    files = sorted(glob.glob(args.data_path))

    target_latitude = 0
    target_longitude = 135

    background_files = files[:7]
    foreground_files = files[10:40]

    background_true = []
    background_pred = []
    print('Loading background files')
    for file in background_files:
        cartesian_coords, time, query_points, rho_true, spherical_coords = load_ref_file(Rs_per_ds, date0,
                                                                                         file, max_latitude,
                                                                                         max_longitude,
                                                                                         max_radius,
                                                                                         min_latitude,
                                                                                         min_longitude,
                                                                                         min_radius, ref_date,
                                                                                         seconds_per_dt)

        ##########################################################
        outputs = sunerf_loader.load_coords(query_points)
        rho_pred = outputs['rho']
        background_true.append(rho_true)
        background_pred.append(rho_pred)

    background_true = np.stack(background_true).mean(axis=0)
    background_pred = np.stack(background_pred).mean(axis=0)[..., 0, 0]

    metrics = {'mae': [], 'corr_coeff': [], 'mae_relative': []}

    print('Loading CME files')
    for i, file in enumerate(foreground_files):
        ##########################################################
        # load data
        cartesian_coords, time, query_points, rho_true, spherical_coords = load_ref_file(Rs_per_ds, date0,
                                                                                         file, max_latitude,
                                                                                         max_longitude,
                                                                                         max_radius,
                                                                                         min_latitude,
                                                                                         min_longitude,
                                                                                         min_radius, ref_date,
                                                                                         seconds_per_dt)

        ##########################################################
        outputs = sunerf_loader.load_coords(query_points)
        rho_pred = outputs['rho'][:, :, :, 0, 0]
        v_pred = outputs['v'][:, :, :, 0, :]

        ##############################################################
        # plot latitude slice
        plot_latitude_slice(rho_pred, spherical_coords,
                            title=r"SuNeRF $\theta={0}^\circ$",
                            img_path=os.path.join(args.out_path, f"lat_{i:03d}.png"), plot_white_r_ticks=args.plot_white_r_ticks)
        if args.plot_velocity:
            plot_latitude_velocity_slice(v_pred, spherical_coords,
                                         title=r"SuNeRF $\theta={0}^\circ$",
                                         img_path=os.path.join(args.out_path, f"lat_{i:03d}_v.png"), plot_white_r_ticks=args.plot_white_r_ticks)
        if args.plot_ground_truth:
            plot_latitude_slice(rho_true, spherical_coords,
                                title=r"Ground-truth $\theta=0^\cdot$",
                                img_path=os.path.join(args.out_path, f"gt_lat_{i:03d}.png"), add_observers=False, plot_white_r_ticks=args.plot_white_r_ticks)

        ##############################################################
        # plot longitude slice

        plot_longitude_slice(rho_pred, spherical_coords,
                             title=r"SuNeRF $\phi=135^\circ$",
                             img_path=os.path.join(args.out_path, f"lon_{i:03d}.png"))
        if args.plot_velocity:
            plot_longitude_velocity_slice(v_pred, spherical_coords,
                                            title=r"SuNeRF $\phi=135^\circ$",
                                            img_path=os.path.join(args.out_path, f"lon_{i:03d}_v.png"))
        if args.plot_ground_truth:
            plot_longitude_slice(rho_true, spherical_coords,
                                 title=r"Ground-truth $\phi=135^\circ$",
                                 img_path=os.path.join(args.out_path, f"gt_lon_{i:03d}.png"), add_observers=False)

        ##########################################################
        # compute center of mass
        cme_true = rho_true - background_true
        cme_true[cme_true < 1] = np.nan

        cme_pred = (rho_pred - background_pred)
        cme_pred[cme_pred < 1] = np.nan

        com_true = compute_center_of_mass(cme_true, cartesian_coords[:, :, :, 0, :] * Rs_per_ds)
        com_spherical_true = cartesian_to_spherical(com_true)
        com_spherical_true[..., 1] = np.rad2deg(com_spherical_true[..., 1]) - 90
        com_spherical_true[..., 2] = np.rad2deg(com_spherical_true[..., 2]) % 360

        com_pred = compute_center_of_mass(cme_pred, cartesian_coords[:, :, :, 0, :] * Rs_per_ds)
        com_spherical_pred = cartesian_to_spherical(com_pred)
        com_spherical_pred[..., 1] = np.rad2deg(com_spherical_pred[..., 1]) - 90
        com_spherical_pred[..., 2] = np.rad2deg(com_spherical_pred[..., 2]) % 360

        # add to lists
        center_of_mass_true.append(com_spherical_true)
        center_of_mass_pred.append(com_spherical_pred)
        times.append(time)
        center_of_mass_diff.append(np.linalg.norm(com_true - com_pred, axis=-1))

        ###################################################
        # compute mean absolute error and cross-correlation
        mae = np.abs(rho_true - rho_pred).mean()
        mae_relative = np.abs(rho_true - rho_pred).sum() / rho_true.sum() * 100
        corr_coeff = np.corrcoef(rho_true.flatten(), rho_pred.flatten())[0, 1]

        metrics['mae'].append(mae)
        metrics['mae_relative'].append(mae_relative)
        metrics['corr_coeff'].append(corr_coeff)

        #####################################################
        # determine shock front
        lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
        lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude)))

        grad_r_true = np.gradient(rho_true[:, lat_idx, lon_idx])
        grad_r_pred = np.gradient(rho_pred[:, lat_idx, lon_idx])

        sf_radius = spherical_coords[:, lat_idx, lon_idx, 0, 0]
        max_idx_true = np.argmin(grad_r_true * sf_radius ** 2)
        max_idx_pred = np.argmin(grad_r_pred * sf_radius ** 2)

        sf_true = sf_radius[max_idx_true]
        sf_pred = sf_radius[max_idx_pred]

        shock_front_true.append(sf_true)
        shock_front_pred.append(sf_pred)

        ############################################################
        # compute total CME mass
        min_lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude - 45)))
        max_lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude + 45)))
        min_lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude - 45)))
        max_lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude + 45)))

        cme_m_true = compute_total_mass(rho_true[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx],
                                        cartesian_coords[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, 0, :] * Rs_per_ds)
        cme_m_pred = compute_total_mass(rho_pred[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx],
                                        cartesian_coords[:, min_lat_idx:max_lat_idx, min_lon_idx:max_lon_idx, 0, :] * Rs_per_ds)

        mass_true.append(cme_m_true)
        mass_pred.append(cme_m_pred)

    ##################################################
    # convert to numpy arrays
    times = np.array(times)
    center_of_mass_true = np.array(center_of_mass_true)
    center_of_mass_pred = np.array(center_of_mass_pred)
    center_of_mass_diff = np.array(center_of_mass_diff)
    mass_true = np.array(mass_true)
    mass_pred = np.array(mass_pred)
    metrics = {k: np.array(v) for k, v in metrics.items()}

    ##################################################
    # estimate velocity of CoM
    velocity_com_true = compute_velocity(times, center_of_mass_true[:, 0])
    velocity_com_pred = compute_velocity(times, center_of_mass_pred[:, 0])

    metrics['velocity_com_true'] = velocity_com_true
    metrics['velocity_com_pred'] = velocity_com_pred

    ###################################################
    # estimate velocity of shock front
    velocity_sf_true = compute_velocity(times, shock_front_true)
    velocity_sf_pred = compute_velocity(times, shock_front_pred)

    metrics['velocity_sf_true'] = velocity_sf_true
    metrics['velocity_sf_pred'] = velocity_sf_pred

    #####################################################
    # estimate mass difference
    mass_diff = np.abs(mass_true - mass_pred).mean()
    mass_relative_diff = np.abs(mass_true - mass_pred).mean() / mass_true.mean() * 100
    metrics['mass_diff'] = mass_diff
    metrics['mass_relative_diff'] = mass_relative_diff

    ####################################################
    # plot center of mass

    fig, axs = plt.subplots(1, 6, figsize=(15, 3))

    ax = axs[0]
    ax.plot(times, center_of_mass_true[:, 0], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_pred[:, 0], '-o', label='SuNeRF', alpha=0.7)
    ax.set_ylabel('Radius [R$_\odot$]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Radius', va='bottom')
    ax.legend()

    ax = axs[1]
    ax.plot(times, shock_front_true, '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, shock_front_pred, '-o', label='SuNeRF', alpha=0.7)
    ax.set_ylabel('Shock front [R$_\odot$]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Shock front', va='bottom')

    ax = axs[2]
    ax.plot(times, center_of_mass_true[:, 1], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_pred[:, 1], '-o', label='SuNeRF', alpha=0.7)
    ax.set_ylabel(r'$\theta$ [deg]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Theta', va='bottom')

    ax = axs[3]
    ax.plot(times, center_of_mass_true[:, 2], '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, center_of_mass_pred[:, 2], '-o', label='SuNeRF', alpha=0.7)
    ax.set_ylabel(r'$\phi$ [deg]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Phi', va='bottom')

    ax = axs[4]
    ax.plot(times, mass_true, '-o', label='Ground-truth', alpha=0.7)
    ax.plot(times, mass_pred, '-o', label='SuNeRF', alpha=0.7)
    ax.set_ylabel('Mass [Ne]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Mass', va='bottom')

    ax = axs[5]
    ax.plot(times, center_of_mass_diff, '-o', color='red', alpha=0.7)
    ax.set_ylabel(r'$\Delta \vec{R}_\text{CoM}$ [R$_\odot$]')
    ax.set_xlabel('Time [UTC]')
    ax.set_title('Center of mass difference', va='bottom')

    # format x-axis
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_path, "center_of_mass.png"), dpi=300, transparent=True)
    plt.close('all')

    ###########################################################
    # print mean error

    with open(os.path.join(args.out_path, "center_of_mass.txt"), 'w') as f:
        mean_error = np.mean(center_of_mass_diff)
        std_error = np.std(center_of_mass_diff)
        print(f"Mean error CoM: {mean_error:.2f} +/- {std_error:.2f} R$_\odot$", file=f)

        mean_error = np.mean(np.abs(center_of_mass_true - center_of_mass_pred), axis=0)
        std_error = np.std(np.abs(center_of_mass_true - center_of_mass_pred), axis=0)
        print(f"Mean error radius: {mean_error[0]:.2f} +/- {std_error[0]:.2f} R$_\odot$", file=f)
        print(f"Mean error theta: {mean_error[1]:.2f} +/- {std_error[1]:.2f} deg", file=f)
        print(f"Mean error phi: {mean_error[2]:.2f} +/- {std_error[2]:.2f} deg", file=f)

        mean_mae = np.mean(metrics['mae'])
        std_mae = np.std(metrics['mae'])
        print(f"Mean MAE: {mean_mae:.2f} +/- {std_mae:.2f} (density)", file=f)

        mean_relative_mae = np.mean(metrics['mae_relative'])
        std_relative_mae = np.std(metrics['mae_relative'])
        print(f"Mean relative MAE: {mean_relative_mae:.2f} +/- {std_relative_mae:.2f} (%)", file=f)

        mean_corr = np.mean(metrics['corr_coeff'])
        std_corr = np.std(metrics['corr_coeff'])
        print(f"Mean correlation coefficient: {mean_corr:.2f} +/- {std_corr:.2f} (density)", file=f)

        velocity_diff = np.abs(metrics['velocity_com_true'] - metrics['velocity_com_pred'])
        print(f'Mean velocity difference: {velocity_diff:.2f} km/s; {velocity_diff / velocity_com_true * 100:.2f}%',
              file=f)

        velocity_diff = np.abs(metrics['velocity_sf_true'] - metrics['velocity_sf_pred'])
        print(f'Mean shock front velocity difference: {velocity_diff:.2f} km/s; {velocity_diff / velocity_sf_true * 100:.2f}%',
              file=f)

        print(f'Mean CME mass difference: {mass_diff:.2e} Ne; {mass_relative_diff:.2f}%', file=f)

    ########################################################
    # save center of mass to file
    np.savez(os.path.join(args.out_path, "center_of_mass.npz"),
             times=times,
             center_of_mass_true=center_of_mass_true,
             center_of_mass_pred=center_of_mass_pred,
             center_of_mass_diff=center_of_mass_diff,
             shock_front_true=shock_front_true,
             shock_front_pred=shock_front_pred,
             mass_true=mass_true,
             mass_pred=mass_pred,
             **metrics)

    # save center of mass to csv
    with open(os.path.join(args.out_path, "center_of_mass.csv"), 'w') as f:
        f.write('time,radius_true,theta_true,phi_true,radius_pred,theta_pred,phi_pred,diff\n')
        for i in range(len(times)):
            f.write(f"{times[i]},{center_of_mass_true[i, 0]},{center_of_mass_true[i, 1]},{center_of_mass_true[i, 2]},"
                    f"{center_of_mass_pred[i, 0]},{center_of_mass_pred[i, 1]},{center_of_mass_pred[i, 2]},"
                    f"{center_of_mass_diff[i]}\n")
