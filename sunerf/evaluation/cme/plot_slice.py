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

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'slices')
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
    max_radius = 15

    times = pd.date_range(start=min_time, end=max_time, periods=n_points)

    density_norm = LogNorm()
    velocity_norm = Normalize(vmin=100, vmax=2000)

    for i, (time) in tqdm(enumerate(times), total=len(times)):
        out = sunerf_loader.load_slice(radius_range=[1.5, max_radius] * u.R_sun, time=time, z=0, pixel_per_Rs=32)

        rho = out["rho"][:, :, 0, 0]
        v = out["v"][:, :, 0, 0]
        cartesian_coords = out["cartesian_coords"][:, :, 0, 0]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        ax = axs[0]
        im_rho = ax.imshow(rho, norm='log', extent=[-max_radius, max_radius, -max_radius, max_radius], cmap='inferno', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_rho, cax=cax)
        cbar.set_label('Electron Density [cm$^{-3}$]')

        # overlay velocity vectors
        strides = 32
        quiver_pos = cartesian_coords[::strides, ::strides]
        quiver_vel = v[::strides, ::strides]
        ax.quiver(quiver_pos[:, :, 0], quiver_pos[:, :, 1],
                  quiver_vel[:, :, 0], quiver_vel[:, :, 1],
                  scale=10000,
                  color='white')

        ax = axs[1]
        im_v = ax.imshow(np.linalg.norm(v, axis=-1), cmap='cividis',
                         extent=[-max_radius, max_radius, -max_radius, max_radius], origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(im_v, cax=cax)
        cbar.set_label('Velocity [km/s]')

        plt.savefig(os.path.join(args.out_path, f'slice_{i:04d}.jpg'), dpi=150)
        plt.close()

