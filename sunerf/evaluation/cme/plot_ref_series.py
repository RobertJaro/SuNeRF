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

    cs = ax.contour(radius, levels=radii, colors='0.2', **plot_kwargs)
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


def make_ratio_map(pB_map, tB_map):
    ratio = np.full_like(pB_map.data, np.nan, dtype=np.float32)
    valid = np.isfinite(pB_map.data) & np.isfinite(tB_map.data) & (tB_map.data > 0)
    ratio[valid] = pB_map.data[valid] / tB_map.data[valid]
    ratio = np.clip(ratio, 0, 1)
    return Map(ratio, pB_map.meta)


def build_reference_paths(tB_pattern=None, pB_pattern=None, legacy_pattern=None, n_samples=20):
    if legacy_pattern is not None and pB_pattern is None:
        pB_pattern = legacy_pattern

    tB_paths = sorted(glob.glob(tB_pattern)) if tB_pattern is not None else []
    pB_paths = sorted(glob.glob(pB_pattern)) if pB_pattern is not None else []

    if not tB_paths and not pB_paths:
        raise ValueError("Provide at least one reference glob via --ref_tB_path, --ref_pB_path, or --ref_map_path.")
    if tB_paths and pB_paths and len(tB_paths) != len(pB_paths):
        raise ValueError(
            f"Reference tB/pB glob counts differ: {len(tB_paths)} tB paths and {len(pB_paths)} pB paths."
        )

    n_refs = max(len(tB_paths), len(pB_paths))
    step = max(1, n_refs // n_samples)
    ref_items = []
    for i in range(0, n_refs, step):
        ref_items.append({
            'tB': tB_paths[i] if tB_paths else None,
            'pB': pB_paths[i] if pB_paths else None,
        })
    return ref_items


def load_reference_map(path):
    if path is None:
        return None
    return Map(path).rotate(order=3)


def apply_mask(s_map, mask):
    if s_map is not None:
        s_map.data[mask] = np.nan


def shared_lognorm(*maps):
    values = []
    for s_map in maps:
        if s_map is None:
            continue
        data = s_map.data
        valid = np.isfinite(data) & (data > 0)
        if np.any(valid):
            values.append(data[valid])
    if not values:
        return LogNorm()
    values = np.concatenate(values)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0 or vmax <= vmin:
        return LogNorm()
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_map_panel(ax, s_map, radii, norm=None, vmin=None, vmax=None, xlabel=None, ylabel=None, cmap=cm.soholasco2):
    im = ax.imshow(s_map.data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, origin='lower')
    s_map.draw_grid(ax, color="blue")
    ax.set_xlabel('' if xlabel is None else xlabel)
    ax.set_ylabel('' if ylabel is None else ylabel)
    plot_radii(ax, s_map, radii=radii)
    return im


def set_outer_tick_labels(ax, show_x=False, show_y=False):
    if hasattr(ax, 'coords'):
        ax.coords[0].set_ticklabel_visible(show_x)
        ax.coords[1].set_ticklabel_visible(show_y)
    else:
        ax.tick_params(axis='x', which='both', labelbottom=show_x)
        ax.tick_params(axis='y', which='both', labelleft=show_y)


def add_top_colorbar(fig, cax, mappable, label):
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(label)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    return cbar


def hide_panel(ax):
    ax.set_axis_off()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--ref_map_path', '--ref_map', dest='ref_map_path', type=str, required=False,
                        help='Legacy path to reference pB maps (glob pattern)')
    parser.add_argument('--ref_tB_path', type=str, required=False, help='Path to reference tB maps (glob pattern)')
    parser.add_argument('--ref_pB_path', type=str, required=False, help='Path to reference pB maps (glob pattern)')
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
    n_samples = 20
    ref_items = build_reference_paths(args.ref_tB_path, args.ref_pB_path, args.ref_map_path, n_samples=n_samples)

    ##########################################################
    # plot settings
    product_labels = {
        "tB": "Total Brightness [MSB]",
        "pB": "Polarized Brightness [MSB]",
        "ratio": "Polarization Ratio",
    }

    ##########################################################
    for ref_item in tqdm(ref_items):
        ref_tB_map = load_reference_map(ref_item['tB'])
        ref_pB_map = load_reference_map(ref_item['pB'])
        ref_map = ref_tB_map if ref_tB_map is not None else ref_pB_map
        occultor_mask, ref_min_radius, ref_max_radius = get_occultor_mask_from_ref_map(ref_map)
        radii = [ref_min_radius, ref_max_radius]

        ##########################################################
        # load reference map
        model_out = sunerf_loader.load_map(ref_map, progress=False, filter_occ=False)

        tB_map = model_out['tB_map']
        pB_map = model_out['pB_map']
        density_map = model_out['density_map']
        ratio_map = make_ratio_map(pB_map, tB_map)
        for s_map in [ref_tB_map, ref_pB_map, tB_map, pB_map, ratio_map, density_map]:
            apply_mask(s_map, occultor_mask)

        ref_ratio_map = (
            make_ratio_map(ref_pB_map, ref_tB_map)
            if ref_tB_map is not None and ref_pB_map is not None
            else None
        )
        apply_mask(ref_ratio_map, occultor_mask)

        tB_norm = shared_lognorm(ref_tB_map, tB_map)
        pB_norm = shared_lognorm(ref_pB_map, pB_map)
        reference_panels = [
            ("ref_tB", None, "tB", None, None, tB_norm, None, None),
            ("ref_pB", None, "pB", None, None, pB_norm, None, None),
            ("ref_ratio", None, "ratio", None, None, None, 0, 1),
        ]
        if ref_tB_map is not None:
            reference_panels[0] = ("ref_tB", crop_map(ref_tB_map, args.xlim, args.ylim),
                                   "tB", None, "Reference\nHelioprojective Y (arcsec)", tB_norm, None, None)
        if ref_pB_map is not None:
            reference_panels[1] = ("ref_pB", crop_map(ref_pB_map, args.xlim, args.ylim),
                                   "pB", None, None, pB_norm, None, None)
        if ref_ratio_map is not None:
            reference_panels[2] = ("ref_ratio", crop_map(ref_ratio_map, args.xlim, args.ylim),
                                   "ratio", None, None, None, 0, 1)

        model_panels = [
            ("model_tB", crop_map(tB_map, args.xlim, args.ylim), "tB", "Helioprojective X (arcsec)",
             "Model\nHelioprojective Y (arcsec)", tB_norm, None, None),
            ("model_pB", crop_map(pB_map, args.xlim, args.ylim), "pB", "Helioprojective X (arcsec)",
             None, pB_norm, None, None),
            ("model_ratio", crop_map(ratio_map, args.xlim, args.ylim), "ratio", "Helioprojective X (arcsec)", None,
             None, 0, 1),
        ]

        ##########################################################
        # observer info
        time = ref_map.date.datetime
        observer_hci = ref_map.observer_coordinate.transform_to(frames.HeliocentricInertial)
        observer = ref_map.observer_coordinate.transform_to(frames.HeliographicCarrington(observer='self'))
        obs_lat = observer.lat
        obs_lon = observer.lon
        obs_distance = observer.radius.to(u.AU)
        obs_hci_lon = observer_hci.lon

        fig = plt.figure(figsize=(10.2, 9.8), constrained_layout=True)
        mosaic = [
            ["top_left", "observer", "top_right"],
            ["cbar_tB", "cbar_pB", "cbar_ratio"],
            [panel[0] for panel in reference_panels],
            [panel[0] for panel in model_panels],
        ]
        subplot_kw = {"observer": {"projection": "3d"}}
        subplot_kw.update({panel[0]: {"projection": panel[1]} for panel in reference_panels if panel[1] is not None})
        subplot_kw.update({panel[0]: {"projection": panel[1]} for panel in model_panels})
        axd = fig.subplot_mosaic(
            mosaic,
            per_subplot_kw=subplot_kw,
            height_ratios=[1.0, 0.06, 1.0, 1.0],
            gridspec_kw={"wspace": 0.05, "hspace": 0.05},
        )
        hide_panel(axd["top_left"])
        hide_panel(axd["top_right"])

        observer_title_lines = [
            time.isoformat(' ', timespec='minutes'),
            f"Lat {obs_lat.to_value(u.deg):.1f}°, Lon {obs_lon.to_value(u.deg):.1f}°, Dist {obs_distance.to_value(u.AU):.2f} AU",
            f"HCI Lon {obs_hci_lon.to_value(u.deg):.1f}°",
        ]
        plot_observer_geometry(axd["observer"], obs_lat, obs_lon, obs_distance, observer_title_lines)

        product_axes = {product: [] for product in product_labels}
        product_mappables = {}
        for panel_id, s_map, product, xlabel, ylabel, norm, vmin, vmax in reference_panels + model_panels:
            if s_map is None:
                hide_panel(axd[panel_id])
                continue
            im = plot_map_panel(
                axd[panel_id], s_map, radii,
                norm=norm, vmin=vmin, vmax=vmax, xlabel=xlabel, ylabel=ylabel,
                cmap='jet' if product == "ratio" else cm.soholasco2,
            )
            set_outer_tick_labels(
                axd[panel_id],
                show_x=panel_id.startswith("model_"),
                show_y=panel_id.endswith("tB"),
            )
            product_axes[product].append(axd[panel_id])
            product_mappables.setdefault(product, im)

        for product in ["tB", "pB", "ratio"]:
            if product not in product_mappables:
                hide_panel(axd[f"cbar_{product}"])
                continue
            add_top_colorbar(fig, axd[f"cbar_{product}"], product_mappables[product], product_labels[product])

        ref_path = ref_item['pB'] or ref_item['tB']
        img_path = os.path.join(args.out_path, Path(ref_path).stem + '.jpg')
        fig.savefig(img_path, dpi=150)
        plt.close('all')
