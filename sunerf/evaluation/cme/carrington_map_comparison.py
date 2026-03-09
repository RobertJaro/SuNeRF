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
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def exposure_normalize_aia_map(s_map):
    exptime = s_map.meta.get("EXPTIME", s_map.meta.get("exptime", None))
    if exptime is None:
        return s_map, np.nan
    try:
        exptime = float(exptime)
    except (TypeError, ValueError):
        return s_map, np.nan
    if not np.isfinite(exptime) or exptime <= 0:
        return s_map, np.nan

    data = np.asarray(s_map.data, dtype=float) / exptime
    meta = dict(s_map.meta)
    bunit = str(meta.get("BUNIT", meta.get("bunit", "DN"))).strip()
    if "/s" not in bunit and "s^-1" not in bunit:
        bunit = f"{bunit}/s"
    meta["BUNIT"] = bunit
    return Map(data, meta), exptime


def aia_to_carrington_reprojected(s_map, n_theta, n_phi):
    # Build a Carrington latitude-longitude target map in CAR projection.
    carr_center = SkyCoord(
        lon=180 * u.deg,
        lat=0 * u.deg,
        radius=(1 * u.AU),
        frame=frames.HeliographicCarrington,
        observer="self",
        obstime=s_map.date,
    )
    target_data = np.zeros((n_theta, n_phi), dtype=np.float32)
    target_scale = [360 / n_phi, 180 / n_theta] * u.deg / u.pix
    target_header = make_fitswcs_header(
        target_data,
        carr_center,
        scale=target_scale,
        projection_code="CAR",
    )
    target_map = Map(target_data, target_header)

    reprojected = s_map.reproject_to(target_map.wcs)
    return np.asarray(reprojected.data, dtype=float)


def integrate_sunerf_radius(sunerf_loader, time, radius_range, n_radius, n_theta, n_phi):
    radii = np.linspace(radius_range[0], radius_range[1], n_radius) * u.R_sun
    theta = np.linspace(-90, 90, n_theta, endpoint=False) * u.deg
    phi_carr = np.linspace(0, 360, n_phi, endpoint=False) * u.deg

    # SuNeRF cube queries use inertial longitudes; convert Carrington grid to HCI at `time`.
    carr_coords = SkyCoord(
        lon=phi_carr,
        lat=np.zeros_like(phi_carr.value) * u.deg,
        radius=np.ones_like(phi_carr.value) * u.R_sun,
        frame=frames.HeliographicCarrington,
        observer="self",
        obstime=time,
    )
    phi_inertial = carr_coords.transform_to(frames.HeliocentricInertial).lon

    out = sunerf_loader.load_spherical_cube(
        radius=radii, latitude=theta, longitude=phi_inertial, time=time
    )
    rho = out["rho"][:, :, :, 0]
    integrated = np.trapz(rho, x=radii.to_value(u.R_sun), axis=0)
    return integrated


def _compute_lognorm(data, default_min=1e-12, default_max=1.0):
    m = np.nanmin(data[np.isfinite(data)]) if np.isfinite(data).any() else default_min
    M = np.nanmax(data[np.isfinite(data)]) if np.isfinite(data).any() else default_max
    m = max(float(m), 1e-12)
    M = float(M)
    if M <= m:
        M = m * 10
    return LogNorm(vmin=m, vmax=M)


def plot_comparison(aia_carr, sunerf_int, out_file, title, aia_unit="DN/s", aia_norm=None, sunerf_norm=None):
    extent = [0, 360, -90, 90]
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 8.0), constrained_layout=True)

    ax = axs[0]
    if aia_norm is None:
        aia_norm = _compute_lognorm(aia_carr, default_min=1e-3, default_max=1.0)
    im = ax.imshow(aia_carr, origin="lower", extent=extent, aspect="auto", cmap="sdoaia193", norm=aia_norm)
    fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, location="right", label=f"AIA intensity [{aia_unit}]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    ax = axs[1]
    if sunerf_norm is None:
        sunerf_norm = _compute_lognorm(sunerf_int, default_min=1e-12, default_max=1.0)
    im = ax.imshow(
        sunerf_int,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="inferno",
        norm=sunerf_norm,
    )
    fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02, location="right", label=r"Integrated density [cm$^{-3}$ R$_\odot$]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare AIA Carrington maps to integrated radial SuNeRF maps.")
    parser.add_argument("--sunerf_path", type=str, required=True, help="Path to SuNeRF save state")
    parser.add_argument("--aia_glob", type=str, required=True, help="Glob pattern for AIA FITS files")
    parser.add_argument("--out_path", type=str, default=None, help="Output directory")
    parser.add_argument("--radius_range", type=float, nargs=2, default=[5.0, 15.0], help="Radial integration range in Rsun")
    parser.add_argument("--n_radius", type=int, default=64, help="Number of radial samples")
    parser.add_argument("--n_theta", type=int, default=128, help="Carrington latitude bins")
    parser.add_argument("--n_phi", type=int, default=256, help="Carrington longitude bins")
    args = parser.parse_args()

    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), "carrington_map_comparison")
    os.makedirs(args.out_path, exist_ok=True)


    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    aia_files = sorted(glob.glob(args.aia_glob))
    if len(aia_files) == 0:
        raise FileNotFoundError(f"No AIA files found for glob: {args.aia_glob}")

    aia_norm = None
    sunerf_norm = None
    for aia_file in tqdm(aia_files):
        s_map = Map(aia_file)
        s_map, exptime = exposure_normalize_aia_map(s_map)
        time_obs = s_map.date.datetime
        query_time = time_obs

        aia_carr = aia_to_carrington_reprojected(s_map, args.n_theta, args.n_phi)
        sunerf_int = integrate_sunerf_radius(
            sunerf_loader=sunerf_loader,
            time=query_time,
            radius_range=args.radius_range,
            n_radius=args.n_radius,
            n_theta=args.n_theta,
            n_phi=args.n_phi,
        )

        stem = Path(aia_file).stem
        out_plot = os.path.join(args.out_path, f"{stem}.png")
        aia_unit = str(s_map.meta.get("BUNIT", s_map.meta.get("bunit", "DN/s")))
        exptime_text = f"{exptime:.3f}s" if np.isfinite(exptime) else "n/a"
        title = (
            f"{stem}\n"
            f"time_obs={time_obs.isoformat()} | query_time={query_time.isoformat()} | exptime={exptime_text}"
        )
        if aia_norm is None:
            aia_norm = _compute_lognorm(aia_carr, default_min=1e-3, default_max=1.0)
        if sunerf_norm is None:
            sunerf_norm = _compute_lognorm(sunerf_int, default_min=1e-12, default_max=1.0)
        plot_comparison(aia_carr, sunerf_int, out_plot, title, aia_unit=aia_unit, aia_norm=aia_norm, sunerf_norm=sunerf_norm)

    print(f"Processed {len(aia_files)} AIA maps")
    print(f"Plots: {args.out_path}")
