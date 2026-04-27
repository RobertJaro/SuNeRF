import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from sunpy.coordinates import frames
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def carrington_to_inertial(longitude, time):
    coord = SkyCoord(lon=longitude, lat=0 * u.deg, radius=1 * u.AU, frame=frames.HeliographicCarrington, obstime=time,
                     observer='self')
    carrington_coord = coord.transform_to(frames.HeliocentricInertial)
    return carrington_coord.lon


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--longitudes', type=float, nargs='+', help='Slices longitudes in degrees', default=None)
    parser.add_argument('--time_range', type=str, nargs=2, help='Time range for visualization in ISO format (e.g., 2024-09-26T00:00:00 2024-09-28T00:00:00)')
    parser.add_argument('--radius_range', type=float, nargs=2, help='Radius range for visualization in Rsun', default=[4, 15])
    parser.add_argument('--latitude_range', type=float, nargs=2, help='Latitude range for visualization in degrees', default=[-90, 90])
    parser.add_argument('--t_points', type=int, default=30, help='Number of time points between start and end time')

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

    # min_time = datetime(2024, 9, 26, 0, 0)
    # max_time = datetime(2024, 9, 28, 0, 0)
    # min_time = datetime(2010, 4, 2, 0, 0)
    # max_time = datetime(2010, 4, 3, 0, 0)
    min_time = parse(args.time_range[0])
    max_time = parse(args.time_range[1])
    t_points = args.t_points

    min_radius = args.radius_range[0]
    max_radius = args.radius_range[1]
    min_latitude = args.latitude_range[0]
    max_latitude = args.latitude_range[1]
    min_theta = np.deg2rad(min_latitude)
    max_theta = np.deg2rad(max_latitude)

    longitudes = args.longitudes * u.deg

    times = pd.date_range(start=min_time, end=max_time, periods=t_points)

    density_norm = LogNorm()
    velocity_norm = Normalize()

    for i, time in tqdm(enumerate(times), total=len(times)):
        carr_longitudes = u.Quantity([carrington_to_inertial(lon, time) for lon in longitudes])
        out = sunerf_loader.load_spherical_cube(radius=np.linspace(min_radius, max_radius, 100) * u.Rsun,
                                                longitude=carr_longitudes,
                                                latitude=np.linspace(min_latitude, max_latitude, 180) * u.deg,
                                                time=time)
        ##########################################################

        fig = plt.figure(figsize=(5 * len(longitudes) + 0.8, 10), constrained_layout=True)

        axd = fig.subplot_mosaic(
            [[f"rho{j}" for j in range(len(longitudes))] + ["rho_cb"],
             [f"vel{j}" for j in range(len(longitudes))] + ["vel_cb"]],
            height_ratios=[1, 1],
            width_ratios=[1] * len(longitudes) + [0.06],
            per_subplot_kw={f"rho{j}": {"projection": "polar"} for j in range(len(longitudes))} |
                           {f"vel{j}": {"projection": "polar"} for j in range(len(longitudes))},

        )

        axd["rho0"].text(-0.22, 0.5, "Density Slice", transform=axd["rho0"].transAxes,
                          rotation=90, va="center", ha="center")
        axd["vel0"].text(-0.22, 0.5, "Velocity Magnitude Slice", transform=axd["vel0"].transAxes,
                          rotation=90, va="center", ha="center")

        density_pc = None
        velocity_pc = None

        for j in range(len(longitudes)):
            target_longitude = longitudes[j]
            rho = out["rho"][:, :, j, 0, 0]  # (r, theta, phi, time, 1) -> (r, theta)
            r = out["spherical_coords"][:, :, j, 0, 0]  # (r, theta, phi, time, 3) -> (r, theta)
            theta = out["spherical_coords"][:, :, j, 0, 1]  # (r, theta, phi, time, 3) -> (r, theta)
            velocity = out["v"][:, :, j, 0]  # (r, theta, phi, time, 3) -> (r, theta, 3)

            # --- Polar plot - density ---
            ax = axd[f"rho{j}"]
            density_pc = ax.pcolormesh(theta, r, rho, shading="auto", norm=density_norm, cmap="RdPu")
            ax.set_thetamin(min_latitude)
            ax.set_thetamax(max_latitude)
            ax.set_xlim(min_theta, max_theta)
            ax.set_title(f"{target_longitude.to_value(u.deg):.1f}° Longitude")
            ax.set_xlabel("Latitude")
            ax.set_ylabel(r"Radius (R$_\odot$)")
            ax.tick_params(axis="y", colors='lightgray')

            # --- Polar plot - velocity ---
            vel_mag = np.linalg.norm(velocity, axis=-1)
            ax = axd[f"vel{j}"]
            velocity_pc = ax.pcolormesh(theta, r, vel_mag, shading="auto", norm=velocity_norm, cmap="viridis")
            ax.set_thetamin(min_latitude)
            ax.set_thetamax(max_latitude)
            ax.set_xlim(min_theta, max_theta)
            ax.set_title(f"{target_longitude.to_value(u.deg):.1f}° Longitude")
            ax.set_xlabel("Latitude")
            ax.set_ylabel(r"Radius (R$_\odot$)")
            ax.tick_params(axis="y", colors='lightgray')

        fig.colorbar(density_pc, cax=axd["rho_cb"], label=r"Density (cm$^{-3}$)")
        fig.colorbar(velocity_pc, cax=axd["vel_cb"], label="Velocity (km/s)")

        fig.suptitle(f"Time: {time.strftime('%Y-%m-%d %H:%M')} UTC", fontsize=16)

        img_path = os.path.join(args.out_path, f"{time.isoformat('T', timespec='minutes')}.jpg")
        fig.savefig(img_path, dpi=150)
        plt.close('all')
