import argparse
import glob
import os
from pathlib import Path

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def plot_radii(ax, s_map, radii=[2, 3, 4, 5], n=360, **plot_kwargs):
    theta = np.linspace(0, 2 * np.pi, n)

    coords = all_coordinates_from_map(s_map)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs
    radius = radius.to_value(u.dimensionless_unscaled)

    cs = ax.contour(radius, levels=radii, cmap='cividis', **plot_kwargs)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f R☉')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--ref_map_path', type=str, required=False, help='Path to reference maps (glob pattern)')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'ref_series')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    ref_paths = sorted(glob.glob(args.ref_map_path))
    ref_paths = ref_paths[::10]

    ##########################################################
    # plot settings
    min_radius = 2.0
    max_radius = 5
    radii = np.linspace(min_radius, max_radius, 5)
    # plotting norms - define based on first plot
    density_norm = LogNorm(vmin=5e-13, vmax=1e-8)
    brightness_norm = LogNorm(vmin=1e-12, vmax=2e-8)
    velocity_norm = Normalize(vmin=100, vmax=2000)

    ##########################################################
    for ref_path in tqdm(ref_paths):
        ref_map = Map(ref_path)
        ref_map = ref_map.rotate(order=3)  # rotate to solar north

        ##########################################################
        # load reference map
        model_out = sunerf_loader.load_map(ref_map, progress=False)

        tB_map = model_out['tB_map']
        pB_map = model_out['pB_map']
        density_map = model_out['density_map']

        ##########################################################
        # observer info
        time = ref_map.date.datetime
        obs_coords = ref_map.observer_coordinate.transform_to(frames.HeliocentricInertial)
        obs_lat = obs_coords.lat
        obs_lon = obs_coords.lon
        # obs_lat = ref_map.carrington_latitude
        # obs_lon = ref_map.carrington_longitude

        target_latitude = 0 * u.deg
        target_longitude = obs_lon + 90 * u.deg
        ##########################################################
        # load latitude slice at the given time
        longitude_range = None # [obs_lon - 110 * u.deg, obs_lon + 110 * u.deg]
        out = sunerf_loader.load_latitude(radius_range=[min_radius, max_radius] * u.R_sun, time=time,
                                          latitude=target_latitude, Nr=128, Nphi=128, longitude_range=longitude_range)
        rho_lat = out["rho"][:, 0, :, 0, 0]  # (r, theta, phi, time, 1) -> (r, phi)
        r_lat = out["spherical_coords"][:, 0, :, 0, 0]  # (r, theta, phi, time, 3) -> (r, phi)
        phi_lat = out["spherical_coords"][:, 0, :, 0, 2]  # (r, theta, phi, time, 3) -> (r, phi)
        ##########################################################
        # load longitude slice at the given time

        out = sunerf_loader.load_longitude(radius_range=[min_radius, max_radius] * u.R_sun, time=time,
                                           longitude=target_longitude, Nr=128, Ntheta=128)
        rho_lon = out["rho"][:, :, 0, 0, 0]  # (r, theta, phi, time, 1) -> (r, theta)
        r_lon = out["spherical_coords"][:, :, 0, 0, 0]  # (r, theta, phi, time, 3) -> (r, theta)
        theta_lon = out["spherical_coords"][:, :, 0, 0, 1]  # (r, theta, phi, time, 3) -> (r, theta)

        ##########################################################

        fig = plt.figure(figsize=(12, 10), constrained_layout=True)

        axd = fig.subplot_mosaic(
            [["im0", "im1"],
             ["lat", "lon"]],
            height_ratios=[1, 2],
            per_subplot_kw={
                "im0": {"projection": tB_map},
                "im1": {"projection": tB_map},
                "lat": {"projection": "polar"},
                "lon": {"projection": "polar"},
            },

        )

        # --- Image 1 (tB_map projection) ---
        ax = axd["im0"]
        im = ax.imshow(ref_map.data, cmap="plasma", norm=brightness_norm, origin='lower')
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, label="pB (MSB)")
        ax.set_title("Polarized Brightness (Reference)")
        tB_map.draw_grid(ax, color="blue")
        ax.set_xlabel('Helioprojective X (arcsec)')
        ax.set_ylabel('Helioprojective Y (arcsec)')
        plot_radii(ax, ref_map, radii=radii)

        # --- Image 2 (tB_map projection) ---
        ax = axd["im1"]
        im = ax.imshow(pB_map.data, cmap="plasma", norm=brightness_norm, origin='lower')
        cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax, label="pB (MSB)")
        ax.set_title("Polarized Brightness")
        tB_map.draw_grid(ax, color="blue")
        ax.set_xlabel('Helioprojective X (arcsec)')
        ax.set_ylabel('Helioprojective Y (arcsec)')
        plot_radii(ax, pB_map, radii=radii)

        # --- Polar plot - latitude ---
        ax = axd["lat"]
        pc = ax.pcolormesh(phi_lat, r_lat, rho_lat, shading="auto", norm=density_norm, cmap="inferno")
        cb = fig.colorbar(pc, ax=ax, pad=0.05, shrink=0.8, label=r"Density (cm$^{-3}$)")
        ax.set_title(f"Density Slice at {target_latitude.to_value(u.deg):.1f}° Latitude")
        ax.set_xlabel("Longitude (rad)")
        ax.set_ylabel(r"Radius (R$_\odot$)")
        ax.plot([obs_lon.to_value(u.rad), obs_lon.to_value(u.rad)], [min_radius, max_radius],
                color="cyan", linestyle="--", linewidth=1)
        ax.set_theta_zero_location("S")
        ax.tick_params(axis="y", colors='lightgray')

        # --- Polar plot - longitude ---
        ax = axd["lon"]
        pc = ax.pcolormesh(theta_lon, r_lon, rho_lon, shading="auto", norm=density_norm, cmap="inferno")
        cb = fig.colorbar(pc, ax=ax, pad=0.05, shrink=0.8, label=r"Density (cm$^{-3}$)")
        ax.set_title(f"Density Slice at {target_longitude.to_value(u.deg):.1f}° Longitude")
        ax.set_xlabel("Latitude (rad)")
        ax.set_ylabel(r"Radius (R$_\odot$)")
        ax.plot([obs_lat.to_value(u.rad), obs_lat.to_value(u.rad)], [min_radius, max_radius],
                color="cyan", linestyle="--", linewidth=1)
        ax.set_theta_zero_location("W")
        ax.tick_params(axis="y", colors='lightgray')
        ax.set_theta_direction(-1)

        fig.suptitle(f"Time: {time} UTC", fontsize=16)

        img_path = os.path.join(args.out_path, Path(ref_path).stem + '.jpg')
        fig.savefig(img_path, dpi=150)
        plt.close('all')
