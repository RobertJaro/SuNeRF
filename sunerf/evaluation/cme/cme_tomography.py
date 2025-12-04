import argparse
import glob
import os

import numpy as np
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from sklearn.linear_model import LinearRegression

from sunerf.evaluation.cme.center_of_mass import load_ref_file
from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.evaluation.util import none_or_float


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


def plot_longitude_slice(rho, spherical_coords, img_path, target_longitude=135,
                         slices=[-20, -10, 0, 10, 20], target_latitude=0):
    """Plot multiple longitude slices of the density cube.
    
    Args:
        rho: Density cube data
        spherical_coords: Coordinates in spherical system
        img_path: Output image path
        target_longitude: Center longitude for slices in degrees
        slices: List of longitude offsets from target in degrees
    """
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)
    subplot_kw = {str(i): {"projection": "polar"} for i in range(len(slices))}
    fig, axs = plt.subplot_mosaic([[str(i) for i in range(len(slices))] + ['CB']],
                                  width_ratios=[1 for _ in range(len(slices))] + [0.1],
                                  per_subplot_kw=subplot_kw, figsize=(1.7 * len(slices), 2.2))

    for i, shift in enumerate(slices):
        ax = axs[str(i)]
        lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude + shift)))

        r = spherical_coords[:, :, lon_idx, 0, 0]
        th = spherical_coords[:, :, lon_idx, 0, 1]
        z = rho[:, :, lon_idx]

        pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')

        ax.plot(np.deg2rad([target_latitude, target_latitude]), np.array([min_radius, max_radius]),
                color='cyan', linestyle='--', linewidth=1)

    cbar_ax = axs['CB']
    fig.colorbar(pc, cax=cbar_ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')

    for ax in [axs[str(i)] for i in range(len(slices))]:
        if min_latitude is not None:
            ax.set_xlim(np.deg2rad([min_latitude, max_latitude]))
        ax.set_rticks([30, 60, 90, 120])
        ax.set_xticks(np.deg2rad([-45, 0, 45]))
        ax.set_rlim(0, max_radius)
    fig.tight_layout(w_pad=0.1)
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')


def plot_longitude_diff_slice(rho_diff, spherical_coords, img_path, target_longitude=135,
                         slices=[-20, -10, 0, 10, 20]):
    """Plot multiple longitude slices of the density cube.

    Args:
        rho: Density cube data
        spherical_coords: Coordinates in spherical system
        img_path: Output image path
        target_longitude: Center longitude for slices in degrees
        slices: List of longitude offsets from target in degrees
    """
    subplot_kw = {str(i): {"projection": "polar"} for i in range(len(slices))}
    fig, axs = plt.subplot_mosaic([[str(i) for i in range(len(slices))] + ['CB']],
                                  width_ratios=[1 for _ in range(len(slices))] + [0.1],
                                  per_subplot_kw=subplot_kw, figsize=(1.7 * len(slices), 2.2))

    for i, shift in enumerate(slices):
        ax = axs[str(i)]
        lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude + shift)))

        r = spherical_coords[:, :, lon_idx, 0, 0]
        th = spherical_coords[:, :, lon_idx, 0, 1]
        z = rho_diff[:, :, lon_idx]

        pc = ax.pcolormesh(th, r, z, edgecolors='face', cmap='Reds', norm=LogNorm(vmin=1e1, vmax=1e3))

    cbar_ax = axs['CB']
    fig.colorbar(pc, cax=cbar_ax, label=r'Error [N$_\text{e}$ cm$^{-3}$]')

    for ax in [axs[str(i)] for i in range(len(slices))]:
        if min_latitude is not None:
            ax.set_xlim(np.deg2rad([min_latitude, max_latitude]))
        ax.set_rticks([30, 60, 90, 120])
        ax.set_xticks(np.deg2rad([-45, 0, 45]))
    fig.tight_layout(w_pad=0.1)
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')

def _plot_latitude_slice(rho, spherical_coords, img_path, target_latitude=0, target_longitudes=[115, 125, 135, 145, 155], add_observers=False):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]
    z = rho[:, lat_idx, :]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(3.5, 3.5))

    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')
    # ax.set_title(title, va='bottom')
    if min_longitude is not None:
        ax.set_xlim(np.deg2rad([min_longitude, max_longitude]))
    # add dashed cyan line at longitude 135
    for lon in target_longitudes:
        ax.plot(np.deg2rad([lon, lon]), np.array([min_radius, max_radius]),
                color='cyan', linestyle='--', linewidth=1)
    ax.set_rticks([30, 60, 90, 120])

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

    ax.set_rlim(0, max_radius)
    # add arrows for observers
    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')

def _plot_latitude_diff_slice(rho, spherical_coords, img_path, target_latitude=0, target_longitudes=[115, 125, 135, 145, 155]):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)

    lat_idx = np.argmin(np.abs(spherical_coords[0, :, 0, 0, 1] - np.deg2rad(target_latitude)))
    r = spherical_coords[:, lat_idx, :, 0, 0]
    ph = spherical_coords[:, lat_idx, :, 0, 2]
    z = rho[:, lat_idx, :]

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(3.5, 3.5))

    pc = ax.pcolormesh(ph, r, z, edgecolors='face', norm=rho_norm, cmap='Reds',)
    # ax.set_title(title, va='bottom')
    if min_longitude is not None:
        ax.set_xlim(np.deg2rad([min_longitude, max_longitude]))
    # add dashed cyan line at longitude 135
    ax.set_rticks([30, 60, 90, 120])

    ax.set_rlim(0, max_radius)
    # add arrows for observers
    fig.tight_layout()
    fig.savefig(img_path, dpi=300, transparent=True)
    plt.close('all')

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Evaluate CME parameters')
    parser.add_argument('--data_path', type=str, required=True, help='Path to density cube data files')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory')
    parser.add_argument('--date0', type=str, default="2010-04-03T09:04:00.000", help='Reference date for CME data')
    parser.add_argument('--max_radius', type=float, default=120.0, help='Maximum radius in solar radii')
    parser.add_argument('--min_radius', type=float, default=30.0, help='Minimum radius in solar radii')
    parser.add_argument('--min_longitude', type=none_or_float, default=90 - 20, help='Minimum longitude in degrees')
    parser.add_argument('--max_longitude', type=none_or_float, default=180 + 20, help='Maximum longitude in degrees')
    parser.add_argument('--min_latitude', type=float, default=-60.0, help='Minimum latitude in degrees')
    parser.add_argument('--max_latitude', type=float, default=60.0, help='Maximum latitude in degrees')
    parser.add_argument('--target_longitude', type=float, default=135.0, help='Target longitude for slices in degrees')
    parser.add_argument('--target_latitude', type=float, default=0.0, help='Target latitude for slices in degrees')
    parser.add_argument('--plot_ground_truth', action='store_true', help='Plot ground truth data')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'tomography')
    os.makedirs(args.out_path, exist_ok=True)

    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    max_radius = args.max_radius
    min_radius = args.min_radius
    min_longitude = args.min_longitude
    max_longitude = args.max_longitude
    min_latitude = args.min_latitude
    max_latitude = args.max_latitude
    Rs_per_ds = sunerf_loader.Rs_per_ds

    date0 = parse(args.date0)
    files = sorted(glob.glob(args.data_path))

    target_latitude = args.target_latitude
    target_longitude = args.target_longitude
    target_longitudes = np.linspace(-20, 20, 5) + target_longitude

    foreground_files = files#[10:40]

    metrics = {'mae': [], 'corr_coeff': []}

    print('Loading CME files')
    for i, file in enumerate(foreground_files):
        # load data
        cartesian_coords, time, query_points, rho_true, spherical_coords = load_ref_file(Rs_per_ds, date0,
                                                                                         file, max_latitude,
                                                                                         max_longitude,
                                                                                         max_radius,
                                                                                         min_latitude,
                                                                                         min_longitude,
                                                                                         min_radius, ref_date,
                                                                                         seconds_per_dt)

        outputs = sunerf_loader.load_coords(query_points)
        rho_pred = outputs['rho'][:, :, :, 0, 0]

        # plot slices
        plot_longitude_slice(rho_pred, spherical_coords, img_path=os.path.join(args.out_path, f"tomography_{i:03d}.png"),
                             target_longitude=target_longitude, target_latitude=target_latitude)
        _plot_latitude_slice(rho_pred, spherical_coords, img_path=os.path.join(args.out_path, f"tomography_lat_{i:03d}.png"),
                             target_longitudes=target_longitudes, target_latitude=target_latitude, add_observers=True)

        # plot differences
        rho_diff = np.abs(rho_pred - rho_true)
        plot_longitude_diff_slice(rho_diff, spherical_coords,
                             img_path=os.path.join(args.out_path, f"diff_{i:03d}.png"))
        _plot_latitude_diff_slice(rho_diff, spherical_coords,
                             img_path=os.path.join(args.out_path, f"diff_lat_{i:03d}.png"))

        if args.plot_ground_truth:
            plot_longitude_slice(rho_true, spherical_coords,
                                 img_path=os.path.join(args.out_path, f"gt_tomography_{i:03d}.png"),
                                 target_longitude=target_longitude, target_latitude=target_latitude)
            _plot_latitude_slice(rho_true, spherical_coords,
                                 img_path=os.path.join(args.out_path, f"gt_tomography_lat_{i:03d}.png"),
                                 target_longitudes=target_longitudes, target_latitude=target_latitude)