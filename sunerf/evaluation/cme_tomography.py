import argparse
import glob
import os

import numpy as np
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.linear_model import LinearRegression

from sunerf.evaluation.center_of_mass import load_ref_file
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
                         slices=[-20, -10, 0, 10, 20]):
    rho_norm = LogNorm(vmin=1e1, vmax=1e3)
    subplot_kw = {str(i): {"projection": "polar"} for i in range(len(slices))}
    fig, axs = plt.subplot_mosaic([[str(i) for i in range(len(slices))] + ['CB']],
                                  width_ratios=[1 for _ in range(len(slices))] + [0.1],
                                  per_subplot_kw=subplot_kw, figsize=(2 * len(slices), 2.5))

    for i, shift in enumerate(slices):
        ax = axs[str(i)]
        lon_idx = np.argmin(np.abs(spherical_coords[0, 0, :, 0, 2] - np.deg2rad(target_longitude + shift)))

        r = spherical_coords[:, :, lon_idx, 0, 0]
        th = spherical_coords[:, :, lon_idx, 0, 1]
        z = rho[:, :, lon_idx]

        pc = ax.pcolormesh(th, r, z, edgecolors='face', norm=rho_norm, cmap='inferno')

    cbar_ax = axs['CB']
    fig.colorbar(pc, cax=cbar_ax, label=r'Density [N$_\text{e}$ cm$^{-3}$]')

    for ax in [axs[str(i)] for i in range(len(slices))]:
        if min_latitude is not None:
            ax.set_xlim(np.deg2rad([min_latitude, max_latitude]))
        # add dashed white line at latitude 0
        ax.set_rticks([30, 60, 90, 120])
    fig.tight_layout(w_pad=0.1)
    fig.savefig(img_path, dpi=300)
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
    parser.add_argument('--min_latitude', type=float, default=-60.0, help='Minimum latitude in degrees')
    parser.add_argument('--max_latitude', type=float, default=60.0, help='Maximum latitude in degrees')
    parser.add_argument('--plot_ground_truth', action='store_true', help='Plot ground truth data')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'tomography')
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

    foreground_files = files[10:40]

    metrics = {'mae': [], 'corr_coeff': []}

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

        ##############################################################
        # plot longitude slice
        plot_longitude_slice(rho_pred, spherical_coords, img_path=os.path.join(args.out_path, f"tomography_{i:03d}.jpg"))
        if args.plot_ground_truth:
            plot_longitude_slice(rho_true, spherical_coords,
                                 img_path=os.path.join(args.out_path, f"gt_tomography_{i:03d}.jpg"))
