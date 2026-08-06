import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.cme.plot_ref_series import (
    apply_mask,
    get_occultor_mask_from_ref_map,
    load_reference_map,
    make_learned_corrected_model_maps,
    resolve_reference_instrument_key,
    shared_lognorm,
)
from sunerf.evaluation.cme.video import (
    format_position,
    get_earth_position,
    get_input_observer_coord,
    get_stereo_a_position,
    plot_observer_geometry,
)
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


STAGE_DIRS = {
    "original_tb": "01_original_tB_observations",
    "masked_sunerf": "02_sunerf_rendered_with_noise_mask",
    "clean_sunerf": "03_sunerf_rendered_clean",
    "clean_tb_pb": "04_sunerf_rendered_tB_pB_clean",
    "zoom_50rs": "05_zoom_out_50Rs",
    "pb_density": "06_pB_left_integrated_density_right",
    "longitude_360": "07_coordinate_info_longitude_360",
    "poles": "08_observer_to_poles",
    "time_1day": "09_time_advance_1day",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create staged CME animation frame folders from reference FITS and a SuNeRF state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sunerf_path", type=str, required=True, help="Path to SuNeRF save state")
    parser.add_argument("--ref_path", type=str, default=None,
                        help="Directory containing reference tB FITS; auto-discovers *tB*.fits")
    parser.add_argument("--ref_tB_path", type=str, default=None, help="Reference tB FITS glob pattern")
    parser.add_argument("--instrument_key", type=str, default=None,
                        help="Instrument key to use for rendering/corrections; inferred from reference paths if omitted")
    parser.add_argument("--out_path", type=str, default=None, help="Path to output directory")
    parser.add_argument("--n_ref_samples", type=int, default=20, help="Number of reference FITS samples")
    parser.add_argument("--n_motion_frames", type=int, default=80, help="Frames for each synthetic camera segment")
    parser.add_argument("--resolution", type=int, default=256, help="Square rendered image resolution in pixels")
    parser.add_argument("--dpi", type=int, default=150, help="Saved frame DPI")
    parser.add_argument("--lon_frame", type=str, choices=["carrington", "hci"], default="carrington",
                        help="Frame used for synthetic camera longitude labels and interpolation")
    parser.add_argument("--zoom_max", type=float, default=50.0, help="Zoom-out outer radius in Rsun")
    parser.add_argument("--time_advance_days", type=float, default=1.0, help="Time advance for final segment in days")
    parser.add_argument("--overwrite", action="store_true", help="Re-render existing frames")
    return parser


def validate_args(args):
    if args.n_ref_samples < 1:
        raise ValueError("--n_ref_samples must be at least 1.")
    if args.n_motion_frames < 1:
        raise ValueError("--n_motion_frames must be at least 1.")


def resolve_reference_patterns(args):
    if args.ref_tB_path is not None:
        return args.ref_tB_path
    if args.ref_path is not None:
        return str(Path(args.ref_path) / "**" / "*tB*.fits*")
    return None


def build_reference_paths(tB_pattern=None, n_samples=20):
    tB_paths = sorted(glob.glob(tB_pattern, recursive=True)) if tB_pattern is not None else []

    if not tB_paths:
        raise ValueError("Provide reference tB files via --ref_path or --ref_tB_path.")

    last_index = max(0, (len(tB_paths) - 1) // 3)
    sample_count = min(n_samples, last_index + 1)
    indices = np.linspace(0, last_index, sample_count, dtype=int)
    return [tB_paths[i] for i in indices]


def copy_map(s_map):
    return Map(np.array(s_map.data, dtype=np.float32, copy=True), dict(s_map.meta))


def target_resolution(args):
    return (args.resolution, args.resolution) * u.pix


def frame_path(out_dir, index):
    return os.path.join(out_dir, f"cme_frame{index:03d}.jpg")


def add_top_colorbar(fig, ax, im, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="6%", pad=0.18, axes_class=plt.Axes)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(label)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=7, pad=1)
    cbar.ax.xaxis.labelpad = 2
    return cbar


def format_map_axis(ax, s_map):
    s_map.draw_grid(ax, color="blue")
    ax.coords[0].set_axislabel(" ")
    ax.coords[1].set_axislabel(" ")


def save_single_map(path, s_map, label, norm=None, cmap=cm.soholasco2, dpi=150):
    fig = plt.figure(figsize=(4.6, 4.4))
    ax = fig.add_subplot(1, 1, 1, projection=s_map)
    im = ax.imshow(s_map.data, cmap=cmap, norm=norm, origin="lower")
    add_top_colorbar(fig, ax, im, label)
    format_map_axis(ax, s_map)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_two_maps(path, left_map, right_map, left_label, right_label, left_norm=None, right_norm=None,
                  left_cmap=cm.soholasco2, right_cmap=cm.soholasco2, dpi=150):
    fig = plt.figure(figsize=(8.8, 4.4))
    axs = [
        fig.add_subplot(1, 2, 1, projection=left_map),
        fig.add_subplot(1, 2, 2, projection=right_map),
    ]
    for ax, s_map, label, norm, cmap in [
        (axs[0], left_map, left_label, left_norm, left_cmap),
        (axs[1], right_map, right_label, right_norm, right_cmap),
    ]:
        im = ax.imshow(s_map.data, cmap=cmap, norm=norm, origin="lower")
        add_top_colorbar(fig, ax, im, label)
        format_map_axis(ax, s_map)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def render_reference_maps(sunerf_loader, ref_tB_path, instrument_key, resolution):
    ref_tB_map = load_reference_map(ref_tB_path).resample(resolution)

    occultor_mask, ref_min_radius, ref_max_radius = get_occultor_mask_from_ref_map(ref_tB_map)
    model_out = sunerf_loader.load_map(ref_tB_map, progress=False, filter_occ=False, instrument_key=instrument_key)
    clean_tB_map = model_out["tB_map"]
    clean_pB_map = model_out["pB_map"]
    masked_tB_map, _, _ = make_learned_corrected_model_maps(
        sunerf_loader, ref_tB_map, clean_tB_map, clean_pB_map, instrument_key=instrument_key
    )

    maps = {
        "ref_tB": copy_map(ref_tB_map),
        "masked_tB": copy_map(masked_tB_map),
        "clean_tB": copy_map(clean_tB_map),
        "clean_pB": copy_map(clean_pB_map),
        "ref_map": ref_tB_map,
        "radii": [ref_min_radius, ref_max_radius],
    }
    for key in ["ref_tB", "masked_tB", "clean_tB", "clean_pB"]:
        apply_mask(maps[key], occultor_mask)
    return maps


def render_camera_maps(sunerf_loader, lat, lon, time, distance, occ_min, occ_max, resolution, lon_frame, instrument_key):
    input_obs_coord = get_input_observer_coord(lat, lon, distance, time, lon_frame)
    if lon_frame == "hci":
        hci_obs_coord = input_obs_coord
        plot_obs_coord = input_obs_coord
        plot_distance = input_obs_coord.distance.to(u.AU)
    else:
        hci_obs_coord = input_obs_coord.transform_to(frames.HeliocentricInertial(obstime=time))
        plot_obs_coord = input_obs_coord.transform_to(frames.HeliographicCarrington(observer="self", obstime=time))
        plot_distance = plot_obs_coord.radius.to(u.AU)

    model_out = sunerf_loader.load_image(
        hci_obs_coord.lat,
        hci_obs_coord.lon,
        time,
        distance=distance,
        resolution=resolution,
        occ_min=occ_min,
        occ_max=occ_max,
        instrument_key=instrument_key,
        progress=False,
    )
    return model_out, plot_obs_coord, plot_distance


def save_coordinate_frame(path, pB_map, density_map, plot_obs_coord, plot_distance, time, lon_frame, dpi):
    fig = plt.figure(figsize=(10.0, 3.6))
    axs = [
        fig.add_subplot(1, 3, 1, projection=pB_map),
        fig.add_subplot(1, 3, 2, projection=density_map),
    ]
    observer_ax = fig.add_subplot(1, 3, 3, projection="3d")

    brightness_norm = LogNorm()
    density_norm = LogNorm()
    for ax, s_map, cmap, norm, label in [
        (axs[0], pB_map, cm.soholasco2, brightness_norm, "Polarized Brightness [MSB]"),
        (axs[1], density_map, "RdPu", density_norm, r"$n_e$ [cm$^{-2}$]"),
    ]:
        im = ax.imshow(s_map.data, cmap=cmap, norm=norm, origin="lower")
        add_top_colorbar(fig, ax, im, label)
        format_map_axis(ax, s_map)

    earth_position = None
    stereo_a_position = None
    try:
        earth_position = get_earth_position(time, lon_frame)
        stereo_a_position = get_stereo_a_position(time, lon_frame)
    except Exception as exc:
        print(f"Could not fetch Earth/STEREO-A positions for {time}: {exc}")

    observer_position = (plot_obs_coord.lat.to(u.deg), plot_obs_coord.lon.to(u.deg), plot_distance)
    print(
        f'[{time.isoformat(sep=" ", timespec="minutes")}] {lon_frame.upper()} | '
        f'{format_position("Observer", observer_position)}'
    )
    plot_observer_geometry(
        observer_ax,
        plot_obs_coord.lat,
        plot_obs_coord.lon,
        plot_distance,
        time,
        earth_position=earth_position,
        stereo_a_position=stereo_a_position,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def get_reference_view(ref_map, lon_frame):
    time = ref_map.date.datetime
    observer = ref_map.observer_coordinate
    if lon_frame == "hci":
        observer = observer.transform_to(frames.HeliocentricInertial(obstime=time))
        lat = observer.lat
        lon = observer.lon
        distance = observer.distance.to(u.AU)
    else:
        observer = observer.transform_to(frames.HeliographicCarrington(observer="self", obstime=time))
        lat = observer.lat
        lon = observer.lon
        distance = observer.radius.to(u.AU)
    return lat.to(u.deg), lon.to(u.deg), distance, time


def save_reference_stages(args, sunerf_loader, ref_paths, instrument_key, out_dirs):
    last_maps = None
    for i, ref_path in tqdm(enumerate(ref_paths), total=len(ref_paths), desc="Reference stages"):
        maps = render_reference_maps(sunerf_loader, ref_path, instrument_key, target_resolution(args))
        last_maps = maps
        print(f'[{maps["ref_map"].date.datetime.isoformat(sep=" ", timespec="minutes")}] reference frame {i:03d}')
        reference_tB_norm = shared_lognorm(maps["ref_tB"], maps["masked_tB"])

        out_file = frame_path(out_dirs["original_tb"], i)
        if args.overwrite or not os.path.exists(out_file):
            save_single_map(out_file, maps["ref_tB"], "Total Brightness [MSB]",
                            norm=reference_tB_norm, dpi=args.dpi)

        out_file = frame_path(out_dirs["masked_sunerf"], i)
        if args.overwrite or not os.path.exists(out_file):
            save_single_map(out_file, maps["masked_tB"], "Total Brightness [MSB]",
                            norm=reference_tB_norm, dpi=args.dpi)

        out_file = frame_path(out_dirs["clean_sunerf"], i)
        if args.overwrite or not os.path.exists(out_file):
            save_single_map(out_file, maps["clean_tB"], "Total Brightness [MSB]",
                            norm=shared_lognorm(maps["clean_tB"]), dpi=args.dpi)

        out_file = frame_path(out_dirs["clean_tb_pb"], i)
        if args.overwrite or not os.path.exists(out_file):
            norm = shared_lognorm(maps["clean_tB"], maps["clean_pB"])
            save_two_maps(
                out_file,
                maps["clean_tB"],
                maps["clean_pB"],
                "Total Brightness [MSB]",
                "Polarized Brightness [MSB]",
                left_norm=norm,
                right_norm=norm,
                dpi=args.dpi,
            )
    return last_maps


def save_motion_stage(args, sunerf_loader, out_dir, lats, lons, times, distance, occ_mins, occ_maxs, mode,
                      instrument_key):
    resolution = target_resolution(args)
    for i, (lat, lon, time, occ_min, occ_max) in tqdm(
        enumerate(zip(lats, lons, times, occ_mins, occ_maxs)),
        total=len(times),
        desc=mode,
    ):
        out_file = frame_path(out_dir, i)
        if os.path.exists(out_file) and not args.overwrite:
            continue
        frame_time = pd.Timestamp(time).to_pydatetime()
        print(f'[{frame_time.isoformat(sep=" ", timespec="minutes")}] {mode} frame {i:03d}')
        model_out, plot_obs_coord, plot_distance = render_camera_maps(
            sunerf_loader,
            lat,
            lon,
            frame_time,
            distance,
            occ_min,
            occ_max,
            resolution,
            args.lon_frame,
            instrument_key,
        )
        pB_map = model_out["pB_map"]
        tB_map = model_out["tB_map"]
        density_map = model_out["density_map"]

        if mode == "zoom_50rs":
            norm = shared_lognorm(tB_map, pB_map)
            save_two_maps(
                out_file,
                tB_map,
                pB_map,
                "Total Brightness [MSB]",
                "Polarized Brightness [MSB]",
                left_norm=norm,
                right_norm=norm,
                dpi=args.dpi,
            )
        elif mode == "pB_density":
            save_two_maps(
                out_file,
                pB_map,
                density_map,
                "Polarized Brightness [MSB]",
                r"$n_e$ [cm$^{-2}$]",
                left_norm=LogNorm(),
                right_norm=LogNorm(),
                right_cmap="RdPu",
                dpi=args.dpi,
            )
        else:
            save_coordinate_frame(
                out_file,
                pB_map,
                density_map,
                plot_obs_coord,
                plot_distance,
                frame_time,
                args.lon_frame,
                args.dpi,
            )


def main():
    args = build_parser().parse_args()
    validate_args(args)
    tB_pattern = resolve_reference_patterns(args)

    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), "reference_animation")
    os.makedirs(args.out_path, exist_ok=True)
    out_dirs = {key: os.path.join(args.out_path, value) for key, value in STAGE_DIRS.items()}
    for out_dir in out_dirs.values():
        os.makedirs(out_dir, exist_ok=True)

    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    ref_paths = build_reference_paths(tB_pattern, n_samples=args.n_ref_samples)
    instrument_key = resolve_reference_instrument_key(
        sunerf_loader, {'tB': ref_paths[0], 'pB': None}, explicit_instrument_key=args.instrument_key
    )

    last_maps = save_reference_stages(args, sunerf_loader, ref_paths, instrument_key, out_dirs)
    ref_map = last_maps["ref_map"]
    ref_min_radius, ref_max_radius = last_maps["radii"]
    occ_min = ref_min_radius * u.R_sun
    occ_max = ref_max_radius * u.R_sun

    start_lat, start_lon, distance, start_time = get_reference_view(ref_map, args.lon_frame)
    n = args.n_motion_frames
    static_lats = np.full(n, start_lat.to_value(u.deg)) * u.deg
    static_lons = np.full(n, start_lon.to_value(u.deg)) * u.deg
    static_times = pd.date_range(start=start_time, end=start_time, periods=n)

    save_motion_stage(
        args,
        sunerf_loader,
        out_dirs["zoom_50rs"],
        static_lats,
        static_lons,
        static_times,
        distance,
        np.full(n, occ_min.to_value(u.R_sun)) * u.R_sun,
        np.linspace(occ_max.to_value(u.R_sun), args.zoom_max, n) * u.R_sun,
        "zoom_50rs",
        instrument_key,
    )
    save_motion_stage(
        args,
        sunerf_loader,
        out_dirs["pb_density"],
        static_lats[:1],
        static_lons[:1],
        static_times[:1],
        distance,
        np.array([occ_min.to_value(u.R_sun)]) * u.R_sun,
        np.array([args.zoom_max]) * u.R_sun,
        "pB_density",
        instrument_key,
    )
    save_motion_stage(
        args,
        sunerf_loader,
        out_dirs["longitude_360"],
        static_lats,
        np.linspace(start_lon.to_value(u.deg), start_lon.to_value(u.deg) + 360.0, n) * u.deg,
        static_times,
        distance,
        np.full(n, occ_min.to_value(u.R_sun)) * u.R_sun,
        np.full(n, args.zoom_max) * u.R_sun,
        "longitude_360",
        instrument_key,
    )

    if n == 1:
        pole_lats = np.array([89.0]) * u.deg
    else:
        pole_lats = np.linspace(start_lat.to_value(u.deg), 89.0, n, endpoint=True) * u.deg
    pole_lons = np.full(n, start_lon.to_value(u.deg)) * u.deg
    save_motion_stage(
        args,
        sunerf_loader,
        out_dirs["poles"],
        pole_lats,
        pole_lons,
        static_times,
        distance,
        np.full(n, occ_min.to_value(u.R_sun)) * u.R_sun,
        np.full(n, args.zoom_max) * u.R_sun,
        "poles",
        instrument_key,
    )

    end_time = start_time + pd.Timedelta(days=args.time_advance_days)
    save_motion_stage(
        args,
        sunerf_loader,
        out_dirs["time_1day"],
        np.full(n, pole_lats[-1].to_value(u.deg)) * u.deg,
        np.full(n, pole_lons[-1].to_value(u.deg)) * u.deg,
        pd.date_range(start=start_time, end=end_time, periods=n),
        distance,
        np.full(n, occ_min.to_value(u.R_sun)) * u.R_sun,
        np.full(n, args.zoom_max) * u.R_sun,
        "time_1day",
        instrument_key,
    )


if __name__ == "__main__":
    main()
