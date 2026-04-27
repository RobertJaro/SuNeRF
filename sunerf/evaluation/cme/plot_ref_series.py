import argparse
import glob
import os
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def plot_radii(ax, s_map, radii=[2, 3, 4, 5], **plot_kwargs):
    coords = all_coordinates_from_map(s_map)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / s_map.rsun_obs
    radius = radius.to_value(u.dimensionless_unscaled)

    cs = ax.contour(radius, levels=radii, cmap='cividis', **plot_kwargs)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%1.1f R☉')


def spherical_to_cartesian(radius, lat, lon):
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def plot_observer_geometry(ax, lat, lon, distance, title_lines):
    sphere_u = np.linspace(0, 2 * np.pi, 120)
    sphere_v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color='0.75', linewidth=0.45, alpha=0.7)
    ax.scatter(0, 0, 0, color='#ffd54a', s=240, edgecolors='#ffb300', linewidths=0.9, depthshade=False)

    distance_au = distance.to_value(u.AU)
    display_radius = 1.08 + 0.08 * np.log10(max(distance_au, 1e-3) * 10)
    obs_x, obs_y, obs_z = spherical_to_cartesian(display_radius, lat, lon)
    ax.plot([0, obs_x], [0, obs_y], [0, obs_z],
            color='red', linewidth=1.6, alpha=0.45, linestyle=':')
    ax.scatter(obs_x, obs_y, obs_z, color='#ffb3b3', s=220, alpha=0.28, depthshade=False)
    ax.scatter(obs_x, obs_y, obs_z, color='red', s=55, edgecolors='white', linewidths=0.6, depthshade=False)

    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    panel_limit = 1.06
    ax.set_xlim(-panel_limit, panel_limit)
    ax.set_ylim(-panel_limit, panel_limit)
    ax.set_zlim(-panel_limit, panel_limit)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()


def get_occultor_mask_from_ref_map(ref_map):
    coords = all_coordinates_from_map(ref_map)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / ref_map.rsun_obs
    radius = radius.to_value(u.dimensionless_unscaled)
    valid = np.isfinite(ref_map.data) & np.isfinite(radius)
    if not np.any(valid):
        raise ValueError("Reference map does not contain any finite pixels to derive an occultor mask.")

    valid_radius = radius[valid]
    min_radius = np.nanmin(valid_radius)
    max_radius = np.nanmax(valid_radius)
    mask = (radius < min_radius) | (radius > max_radius)
    return mask, min_radius, max_radius


def crop_map(s_map, xlim=None, ylim=None):
    if xlim is None and ylim is None:
        return s_map

    bottom_left = SkyCoord(
        (xlim[0] if xlim is not None else s_map.bottom_left_coord.Tx.to_value(u.arcsec)) * u.arcsec,
        (ylim[0] if ylim is not None else s_map.bottom_left_coord.Ty.to_value(u.arcsec)) * u.arcsec,
        frame=s_map.coordinate_frame,
    )
    top_right = SkyCoord(
        (xlim[1] if xlim is not None else s_map.top_right_coord.Tx.to_value(u.arcsec)) * u.arcsec,
        (ylim[1] if ylim is not None else s_map.top_right_coord.Ty.to_value(u.arcsec)) * u.arcsec,
        frame=s_map.coordinate_frame,
    )
    return s_map.submap(bottom_left=bottom_left, top_right=top_right)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--ref_map_path', type=str, required=False, help='Path to reference maps (glob pattern)')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--xlim', type=float, nargs=2, default=None,
                        help='Optional x-axis limits in arcsec for the image panels')
    parser.add_argument('--ylim', type=float, nargs=2, default=None,
                        help='Optional y-axis limits in arcsec for the image panels')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'ref_series')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    ref_paths = sorted(glob.glob(args.ref_map_path))
    n_samples = 20
    ref_paths = ref_paths[::max(1, len(ref_paths) // n_samples)]

    ##########################################################
    # plot settings
    radii = [2.5, 5, 10, 15]
    brightness_norm = LogNorm()

    ##########################################################
    for ref_path in tqdm(ref_paths):
        ref_map = Map(ref_path)
        ref_map = ref_map.rotate(order=3)  # rotate to solar north
        occultor_mask, ref_min_radius, ref_max_radius = get_occultor_mask_from_ref_map(ref_map)

        ##########################################################
        # load reference map
        model_out = sunerf_loader.load_map(ref_map, progress=False, filter_occ=False)

        tB_map = model_out['tB_map']
        pB_map = model_out['pB_map']
        density_map = model_out['density_map']
        tB_map.data[occultor_mask] = np.nan
        pB_map.data[occultor_mask] = np.nan
        density_map.data[occultor_mask] = np.nan
        ref_plot_map = crop_map(ref_map, args.xlim, args.ylim)
        model_plot_map = crop_map(pB_map, args.xlim, args.ylim)

        ##########################################################
        # observer info
        time = ref_map.date.datetime
        observer_hci = ref_map.observer_coordinate.transform_to(frames.HeliocentricInertial)
        observer = ref_map.observer_coordinate.transform_to(frames.HeliographicCarrington(observer='self'))
        obs_lat = observer.lat
        obs_lon = observer.lon
        obs_distance = observer.radius.to(u.AU)
        obs_hci_lon = observer_hci.lon

        fig = plt.figure(figsize=(12.8, 3.8), constrained_layout=True)
        axd = fig.subplot_mosaic(
            [["observer", "im0", "im1"]],
            per_subplot_kw={
                "observer": {"projection": "3d"},
                "im0": {"projection": ref_plot_map},
                "im1": {"projection": model_plot_map},
            },
            width_ratios=[1.05, 1.0, 1.0],
        )

        observer_title_lines = [
            time.isoformat(' ', timespec='minutes'),
            f"Lat {obs_lat.to_value(u.deg):.1f}°, Lon {obs_lon.to_value(u.deg):.1f}°, Dist {obs_distance.to_value(u.AU):.2f} AU",
            f"HCI Lon {obs_hci_lon.to_value(u.deg):.1f}°",
        ]
        plot_observer_geometry(axd["observer"], obs_lat, obs_lon, obs_distance, observer_title_lines)

        ax = axd["im0"]
        im = ax.imshow(ref_plot_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        fig.colorbar(im, ax=ax, location="right", fraction=0.046, pad=0.03, label="pB (MSB)")
        ax.set_title("Polarized Brightness (Reference)")
        ref_plot_map.draw_grid(ax, color="blue")
        ax.set_xlabel('Helioprojective X (arcsec)')
        ax.set_ylabel('Helioprojective Y (arcsec)')
        plot_radii(ax, ref_plot_map, radii=radii)

        ax = axd["im1"]
        im = ax.imshow(model_plot_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        fig.colorbar(im, ax=ax, location="right", fraction=0.046, pad=0.03, label="pB (MSB)")
        ax.set_title(f"Polarized Brightness (Model)\nOcculter {ref_min_radius:.2f}-{ref_max_radius:.2f} R☉")
        model_plot_map.draw_grid(ax, color="blue")
        ax.set_xlabel('Helioprojective X (arcsec)')
        ax.set_ylabel('Helioprojective Y (arcsec)')
        plot_radii(ax, model_plot_map, radii=radii)

        img_path = os.path.join(args.out_path, Path(ref_path).stem + '.jpg')
        fig.savefig(img_path, dpi=150)
        plt.close('all')
