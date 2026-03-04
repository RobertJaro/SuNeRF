import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--radius', type=float, nargs='+', help='Radii to plot in solar radii', default=[3, 5, 8, 10])
    parser.add_argument('--projection', type=str, choices=['lat', 'sinlat'], default='sinlat',
                        help='Latitude projection for radius maps')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'radius_maps')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    observer_longitudes = set([o['longitude'] % (360 * u.deg) for o in observers])
    observer_times = set([o['time'] for o in observers])
    min_time = min(observer_times)
    max_time = max(observer_times)
    mid_time = (max_time - min_time) / 2 + min_time

    n_points = 100
    radius = args.radius * u.R_sun

    times = pd.date_range(start=min_time, end=max_time, periods=n_points)

    rho_norm = None
    velocity_norm = LogNorm(vmin=200, vmax=1000)

    for i, time in tqdm(enumerate(times), total=len(times)):
        out = sunerf_loader.load_radius(radius=radius, time=time, projection=args.projection)

        rho = out["rho"][:, :, :, 0]  # (r, theta, phi, time, 1)
        v = out["v"][:, :, :, 0, :]  # (r, theta, phi, time, 3)
        v_abs = np.linalg.norm(v, axis=-1)
        latitude_axis = out["latitude_axis"]
        latitude_label = 'Latitude [deg]' if args.projection == 'lat' else r'$\sin(Latitude)$'

        if rho_norm is None:
            rho_norm = LogNorm(vmin=rho.min(), vmax=rho.max())

        fig = plt.figure(figsize=(12, 3 * len(radius) + 0.8), layout='constrained')
        gs = fig.add_gridspec(
            nrows=len(radius) + 1,
            ncols=2,
            height_ratios=[1] * len(radius) + [0.06]
        )
        axs = np.empty((len(radius), 2), dtype=object)
        for j in range(len(radius)):
            axs[j, 0] = fig.add_subplot(gs[j, 0])
            axs[j, 1] = fig.add_subplot(gs[j, 1])
        cax_rho = fig.add_subplot(gs[-1, 0])
        cax_v = fig.add_subplot(gs[-1, 1])

        for j in range(len(radius)):
            ax_rho = axs[j, 0]
            ax_rho.imshow(
                rho[j, :, :], norm=rho_norm,
                extent=[0, 360, latitude_axis.min(), latitude_axis.max()], cmap='inferno', origin='lower',
                aspect='auto'
            )
            ax_rho.set_title(f'Radius = {radius[j].to_value(u.R_sun):.2f} R☉ at {time.strftime("%Y-%m-%d %H:%M:%S")}')
            ax_rho.set_xlabel('Longitude [deg]')
            ax_rho.set_ylabel(latitude_label)

            ax_v = axs[j, 1]
            ax_v.imshow(
                v_abs[j, :, :], norm=velocity_norm,
                extent=[0, 360, latitude_axis.min(), latitude_axis.max()], cmap='viridis', origin='lower',
                aspect='auto'
            )
            ax_v.set_title(f'|v| at {radius[j].to_value(u.R_sun):.2f} R☉')
            ax_v.set_xlabel('Longitude [deg]')
            ax_v.set_ylabel(latitude_label)

        mappable_rho = ScalarMappable(norm=rho_norm, cmap='inferno')
        mappable_v = ScalarMappable(norm=velocity_norm, cmap='viridis')
        cbar_rho = fig.colorbar(mappable_rho, cax=cax_rho, orientation='horizontal')
        cbar_rho.set_label('Density [g/cm³]')
        cbar_v = fig.colorbar(mappable_v, cax=cax_v, orientation='horizontal')
        cbar_v.set_label('|v| [km/s]')

        plt.savefig(os.path.join(args.out_path, f'slice_{i:04d}.jpg'), dpi=150)
        plt.close()
