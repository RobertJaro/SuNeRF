import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--radius', type=float, nargs='+', help='Radii to plot in solar radii', default=[3, 5, 8, 10])

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

    rho_norm = LogNorm(vmin=1e-12, vmax=1e-6)
    velocity_norm = Normalize(vmin=100, vmax=2000)

    for i, time in tqdm(enumerate(times), total=len(times)):
        out = sunerf_loader.load_radius(radius=radius, time=time)

        rho = out["rho"][:, :, :, 0] # (r, theta, phi, time, 1)

        fig, axs = plt.subplots(len(radius), 1, figsize=(6, 3 * len(radius)))

        for j, ax in enumerate(axs):
            im_rho = ax.imshow(rho[j, :, :], norm=rho_norm, extent=[0, 360, -90, 90], cmap='inferno', origin='lower')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(im_rho, cax=cax)
            cbar.set_label('Density [g/cm³]')
            ax.set_title(f'Radius = {radius[j].to_value(u.R_sun):.2f} R☉ at {time.strftime("%Y-%m-%d %H:%M:%S")}')
            ax.set_xlabel('Longitude [deg]')
            ax.set_ylabel('Latitude [deg]')

        plt.savefig(os.path.join(args.out_path, f'slice_{i:04d}.jpg'), dpi=150)
        plt.close()

