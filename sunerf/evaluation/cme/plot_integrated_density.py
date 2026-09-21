import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from dateutil.parser import parse
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from sunerf.evaluation.image_sequence_to_video import write_video
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def longitude_samples(longitude_range, n_longitude):
    """Return an unwrapped inertial grid, allowing ranges such as 350--10 deg."""
    start, stop = longitude_range
    if stop <= start:
        stop += 360
    return np.linspace(start, stop, n_longitude)


def integrate_latitude(values, latitudes):
    """Integrate scalar values over latitude, retaining radius and longitude."""
    values = np.asarray(values)
    input_shape = values.shape
    while values.ndim > 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 3 or values.shape[1] != len(latitudes):
        raise ValueError(
            "Expected values with shape (radius, latitude, longitude) plus optional "
            f"singleton trailing axes, got {input_shape}."
        )

    latitude_radians = np.deg2rad(latitudes)
    return _trapezoid(
        values * np.cos(latitude_radians)[None, :, None],
        x=latitude_radians,
        axis=1,
    )


def integrate_density(rho, latitudes):
    """Integrate density over latitude, retaining radius and longitude."""
    return integrate_latitude(rho, latitudes)


def average_velocity(velocity, latitudes):
    """Return the latitude-weighted mean velocity magnitude at each radius/longitude."""
    velocity = np.asarray(velocity)
    if velocity.ndim == 5 and velocity.shape[-2] == 1:
        velocity = velocity[..., 0, :]
    if velocity.ndim != 4 or velocity.shape[-1] != 3:
        raise ValueError(
            "Expected velocity with shape (radius, latitude, longitude, 3) plus an "
            f"optional singleton time axis, got {np.asarray(velocity).shape}."
        )

    velocity_magnitude = np.linalg.norm(velocity, axis=-1)
    integrated_velocity = integrate_latitude(velocity_magnitude, latitudes)
    latitude_radians = np.deg2rad(latitudes)
    angular_width = _trapezoid(np.cos(latitude_radians), x=latitude_radians)
    return integrated_velocity / angular_width


def density_lognorm(data):
    positive = np.asarray(data)[np.isfinite(data) & (np.asarray(data) > 0)]
    if positive.size == 0:
        raise ValueError("Integrated density contains no finite positive values to plot.")
    vmin = float(np.nanmin(positive))
    vmax = float(np.nanmax(positive))
    if vmax <= vmin:
        vmax = vmin * 10
    return LogNorm(vmin=vmin, vmax=vmax)


def finite_norm(data):
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("Velocity contains no finite values to plot.")
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1
    return Normalize(vmin=vmin, vmax=vmax)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a longitude-radius video of latitude-integrated density and "
            "latitude-averaged velocity."
        )
    )
    parser.add_argument("--sunerf_path", required=True, help="Path to SuNeRF save state")
    parser.add_argument("--out_path", default=None, help="Output directory")
    parser.add_argument("--video_path", default=None, help="Output MP4 path")
    parser.add_argument(
        "--longitude_range", type=float, nargs=2, required=True,
        help="Heliocentric inertial longitude range in degrees; a decreasing range crosses 360 degrees",
    )
    parser.add_argument(
        "--time_range", nargs=2, required=True,
        help="Time range in ISO format (e.g. 2025-09-21T00:00:00 2025-09-24T00:00:00)",
    )
    parser.add_argument(
        "--radius_range", type=float, nargs=2, default=[4, 15],
        help="Radius plotting range in Rsun",
    )
    parser.add_argument(
        "--latitude_range", type=float, nargs=2, default=[-50, 0],
        help="Latitude integration range in degrees",
    )
    parser.add_argument("--t_points", type=int, default=30, help="Number of time samples/frames")
    parser.add_argument("--n_radius", type=int, default=80, help="Number of radial samples")
    parser.add_argument("--n_latitude", type=int, default=64, help="Number of latitude samples")
    parser.add_argument("--n_longitude", type=int, default=64, help="Number of longitude samples")
    parser.add_argument("--dpi", type=int, default=150, help="Output frame resolution")
    parser.add_argument("--fps", type=float, default=10, help="Video playback frame rate")
    args = parser.parse_args()

    for name in ("t_points", "n_radius", "n_latitude", "n_longitude"):
        if getattr(args, name) < 2:
            parser.error(f"--{name} must be at least 2")
    if args.dpi < 1:
        parser.error("--dpi must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.radius_range[0] <= 0 or args.radius_range[1] <= args.radius_range[0]:
        parser.error("--radius_range must contain two increasing positive values")
    if args.latitude_range[1] <= args.latitude_range[0]:
        parser.error("--latitude_range must contain two increasing values")
    if args.latitude_range[0] < -90 or args.latitude_range[1] > 90:
        parser.error("--latitude_range must stay within -90 and 90 degrees")
    if args.longitude_range[0] == args.longitude_range[1]:
        parser.error("--longitude_range must span a non-zero interval")

    out_path = args.out_path or os.path.join(
        os.path.dirname(args.sunerf_path), "latitude_integrated_video"
    )
    os.makedirs(out_path, exist_ok=True)
    frames_path = os.path.join(out_path, "frames")
    os.makedirs(frames_path, exist_ok=True)
    video_path = args.video_path or os.path.join(
        out_path, "density_velocity_longitude_radius.mp4"
    )
    os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

    times = pd.date_range(
        start=parse(args.time_range[0]), end=parse(args.time_range[1]), periods=args.t_points
    )
    radii = np.linspace(*args.radius_range, args.n_radius)
    latitudes = np.linspace(*args.latitude_range, args.n_latitude)
    inertial_longitudes = longitude_samples(args.longitude_range, args.n_longitude)

    loader = ThomsonSuNeRFLoader(args.sunerf_path)
    longitude_radians = np.deg2rad(inertial_longitudes)
    density_norm = None
    velocity_norm = None
    frame_paths = []

    for i, time in enumerate(tqdm(times, desc="Sampling and rendering frames")):
        output = loader.load_spherical_cube(
            radius=radii * u.Rsun,
            latitude=latitudes * u.deg,
            longitude=inertial_longitudes * u.deg,
            time=time,
        )
        integrated_density = integrate_density(output["rho"], latitudes)
        average_velocity_magnitude = average_velocity(output["v"], latitudes)
        if density_norm is None:
            density_norm = density_lognorm(integrated_density)
        if velocity_norm is None:
            velocity_norm = finite_norm(average_velocity_magnitude)

        fig, axes = plt.subplots(
            2, 1, figsize=(9, 11), sharex=True, sharey=True,
            subplot_kw={"projection": "polar"}, constrained_layout=True,
        )
        density_image = axes[0].pcolormesh(
            longitude_radians, radii, integrated_density,
            shading="auto", cmap="RdPu", norm=density_norm,
        )
        density_colorbar = fig.colorbar(density_image, ax=axes[0], pad=0.02)
        density_colorbar.set_label(r"Latitude-integrated density [cm$^{-3}$ rad]")
        axes[0].set_title("Latitude-integrated density")

        velocity_image = axes[1].pcolormesh(
            longitude_radians, radii, average_velocity_magnitude,
            shading="auto", cmap="viridis", norm=velocity_norm,
        )
        velocity_colorbar = fig.colorbar(velocity_image, ax=axes[1], pad=0.02)
        velocity_colorbar.set_label(r"Mean velocity magnitude [km s$^{-1}$]")
        axes[1].set_title("Latitude-averaged velocity magnitude")

        for ax in axes:
            ax.set_xlim(longitude_radians[0], longitude_radians[-1])
            ax.set_ylim(radii[0], radii[-1])
            ax.set_rorigin(0)
            ax.set_ylabel(r"Radius [R$_\odot$]")
            ax.tick_params(axis="y", colors="gray")
        axes[1].set_xlabel("Heliocentric inertial longitude")
        fig.suptitle(
            f"{time.strftime('%Y-%m-%d %H:%M')} UTC | "
            f"latitude {args.latitude_range[0]:g}--{args.latitude_range[1]:g}°"
        )

        frame_path = os.path.join(frames_path, f"frame_{i:04d}.png")
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close(fig)
        frame_paths.append(frame_path)
        del output, integrated_density, average_velocity_magnitude

    write_video(frame_paths, video_path, args.fps)
    print(f"Frames: {frames_path}")
    print(f"Video: {video_path}")


if __name__ == "__main__":
    main()
