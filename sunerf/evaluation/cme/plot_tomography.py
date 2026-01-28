import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

def inertial_to_carrington(longitude, time):
    coord = SkyCoord(lon=longitude, lat=0 * u.deg, distance=1 * u.AU, frame=frames.HeliocentricInertial, obstime=time, observer='self')
    carrington_coord = coord.transform_to(frames.HeliographicCarrington)
    return carrington_coord.lon

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--longitudes', type=float, nargs='+', help='Slices longitudes in degrees', default=None)

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

    # observer_times = set([o['time'] for o in observers])
    # min_time = min(observer_times)
    # max_time = max(observer_times)

    min_time = datetime(2024, 9, 26, 0, 0)
    max_time = datetime(2024, 9, 28, 0, 0)
    t_points = 30

    min_radius = 2.5
    max_radius = 15

    longitudes = args.longitudes * u.deg


    times = pd.date_range(start=min_time, end=max_time, periods=t_points)

    density_norm = LogNorm()
    velocity_norm = Normalize(vmin=100, vmax=1000)

    for i, time in tqdm(enumerate(times), total=len(times)):
        carr_longitudes = u.Quantity([inertial_to_carrington(lon, time) for lon in longitudes])
        out = sunerf_loader.load_spherical_cube(radius=np.linspace(min_radius, max_radius, 100) * u.Rsun,
                                                longitude=carr_longitudes, latitude=np.linspace(0, 360, 180, endpoint=False) * u.deg,
                                                time=time)
        ##########################################################

        fig = plt.figure(figsize=(5 * len(longitudes), 10), constrained_layout=True)

        axd = fig.subplot_mosaic(
            [[f"rho{j}" for j in range(len(longitudes))],
             [f"vel{j}" for j in range(len(longitudes))]],
            height_ratios=[1, 2],
            per_subplot_kw={f"rho{j}": {"projection": "polar"} for j in range(len(longitudes))} |
                           {f"vel{j}": {"projection": "polar"} for j in range(len(longitudes))},

        )

        for j in range(len(longitudes)):
            target_longitude = longitudes[j]
            rho = out["rho"][:, :, j, 0, 0]  # (r, theta, phi, time, 1) -> (r, theta)
            r = out["spherical_coords"][:, :, j, 0, 0]  # (r, theta, phi, time, 3) -> (r, theta)
            theta = out["spherical_coords"][:, :, j, 0, 1]  # (r, theta, phi, time, 3) -> (r, theta)
            velocity = out["v"][:, :, j, 0]  # (r, theta, phi, time, 3) -> (r, theta, 3)


            # --- Polar plot - density ---
            ax = axd[f"rho{j}"]
            pc = ax.pcolormesh(theta, r, rho, shading="auto", norm=density_norm, cmap="inferno")
            cb = fig.colorbar(pc, ax=ax, pad=0.05, shrink=0.8, label=r"Density (cm$^{-3}$)")
            ax.set_title(f"Density Slice at {target_longitude.to_value(u.deg):.1f}° Longitude")
            ax.set_xlabel("Latitude (rad)")
            ax.set_ylabel(r"Radius (R$_\odot$)")
            ax.set_theta_zero_location("W")
            ax.tick_params(axis="y", colors='lightgray')
            ax.set_theta_direction(-1)

            # --- Polar plot - velocity ---
            vel_mag = np.linalg.norm(velocity, axis=-1)
            ax = axd[f"vel{j}"]
            pc = ax.pcolormesh(theta, r, vel_mag, shading="auto", norm=velocity_norm, cmap="viridis")
            cb = fig.colorbar(pc, ax=ax, pad=0.05, shrink=0.8, label="Velocity (km/s)")
            ax.set_title(f"Velocity Magnitude Slice at {target_longitude.to_value(u.deg):.1f}° Longitude")
            ax.set_xlabel("Latitude (rad)")
            ax.set_ylabel(r"Radius (R$_\odot$)")
            ax.set_theta_zero_location("W")
            ax.tick_params(axis="y", colors='lightgray')
            ax.set_theta_direction(-1)

        fig.suptitle(f"Time: {time} UTC", fontsize=16)

        img_path = os.path.join(args.out_path, f"{time.isoformat('T', timespec='seconds')}.jpg")
        fig.savefig(img_path, dpi=150)
        plt.close('all')

