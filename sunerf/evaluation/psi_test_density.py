#!/usr/bin/env python3
"""Compare PSI/MAS ground-truth density with the degraded-data reconstruction."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from sunpy.coordinates import frames
from sunpy.map import Map

from sunerf.data.psi.density_cube import read_psi_density
from sunerf.data.psi.psi_test_paths import (
    FRAME_TOKEN,
)

from sunerf.evaluation.loader import ThomsonSuNeRFLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian


# The supplied HDF4 files contain dimensionless MAS mass density and carry no
# composition metadata.  Match SuNeRF's existing PSI convention by assuming a
# fully ionized, pure-hydrogen plasma.  For helium electron fraction f,
# rho_code = n_e * (1 + 4 f) / (1 + 2 f) in normalized number-density units.
MAS_NUMBER_DENSITY_NORMALIZATION_CM3 = 1.0e8
HELIUM_ELECTRON_FRACTION = 0.0
HELIUM_MASS_PER_ELECTRON = (
    (1.0 + 4.0 * HELIUM_ELECTRON_FRACTION)
    / (1.0 + 2.0 * HELIUM_ELECTRON_FRACTION)
)
MAS_RHO_TO_ELECTRON_CM3 = (
    MAS_NUMBER_DENSITY_NORMALIZATION_CM3 / HELIUM_MASS_PER_ELECTRON
)
HISTOGRAM_SAMPLE_COUNT = 500_000
RADIUS_RANGE = (2.0, 30.0)
SLICE_RADII = (5.0, 10.0, 15.0, 20.0, 25.0)
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sunerf-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--density-path", type=Path, required=True)
    parser.add_argument("--reference-tb-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_density(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return read_psi_density(path)


def carrington_to_hci_longitudes(phi: np.ndarray, time) -> np.ndarray:
    """Convert the fixed PSI Carrington grid into model-query HCI longitudes."""
    coordinates = SkyCoord(
        lon=phi * u.rad,
        lat=np.zeros_like(phi) * u.rad,
        radius=np.ones_like(phi) * u.R_sun,
        frame=frames.HeliographicCarrington,
        observer="self",
        obstime=time,
    )
    return coordinates.transform_to(frames.HeliocentricInertial).lon.to_value(u.rad)


def make_query_points(
    radius: np.ndarray,
    latitude: np.ndarray,
    longitude_hci: np.ndarray,
    normalized_time: float,
    rs_per_ds: float,
) -> np.ndarray:
    spherical = np.stack([radius, latitude, longitude_hci], axis=-1)
    cartesian = spherical_to_cartesian(spherical, np) / float(rs_per_ds)
    time = np.full((cartesian.shape[0], 1), normalized_time, dtype=np.float64)
    return np.concatenate([cartesian, time], axis=-1).astype(np.float32)


def query_density(
    loader: ThomsonSuNeRFLoader,
    radius: np.ndarray,
    latitude: np.ndarray,
    longitude_hci: np.ndarray,
    normalized_time: float,
) -> np.ndarray:
    points = make_query_points(
        radius,
        latitude,
        longitude_hci,
        normalized_time,
        loader.Rs_per_ds,
    )
    return loader.load_coords(points, batch_size=8192, progress=True)["rho"].reshape(-1)


def correlation_metrics(truth: np.ndarray, reconstruction: np.ndarray) -> dict[str, float | int]:
    valid = (
        np.isfinite(truth)
        & np.isfinite(reconstruction)
        & (truth > 0)
        & (reconstruction > 0)
    )
    truth = np.asarray(truth[valid], dtype=np.float64)
    reconstruction = np.asarray(reconstruction[valid], dtype=np.float64)
    if truth.size < 2:
        raise ValueError("Not enough positive finite density samples for correlation.")
    log_truth = np.log10(truth)
    log_reconstruction = np.log10(reconstruction)
    slope, intercept = np.polyfit(log_truth, log_reconstruction, 1)
    return {
        "n_points": int(truth.size),
        "pearson_r_linear": float(np.corrcoef(truth, reconstruction)[0, 1]),
        "pearson_r_log10": float(np.corrcoef(log_truth, log_reconstruction)[0, 1]),
        "fit_slope_log10": float(slope),
        "fit_intercept_log10": float(intercept),
        "rmse_log10": float(np.sqrt(np.mean((log_truth - log_reconstruction) ** 2))),
    }


def sample_volume(
    density: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius_indices = np.flatnonzero(
        (radius >= RADIUS_RANGE[0]) & (radius <= RADIUS_RANGE[1])
    )
    theta_indices = np.flatnonzero((theta >= 0.0) & (theta <= np.pi))
    phi_indices = np.flatnonzero((phi >= 0.0) & (phi < 2.0 * np.pi))
    shape = (phi_indices.size, theta_indices.size, radius_indices.size)
    population = int(np.prod(shape))
    sample_count = min(HISTOGRAM_SAMPLE_COUNT, population)
    generator = np.random.default_rng(RANDOM_SEED)
    flat_indices = generator.choice(population, size=sample_count, replace=False)
    local_phi, local_theta, local_radius = np.unravel_index(flat_indices, shape)
    p_idx = phi_indices[local_phi]
    t_idx = theta_indices[local_theta]
    r_idx = radius_indices[local_radius]
    truth = density[p_idx, t_idx, r_idx] * MAS_RHO_TO_ELECTRON_CM3
    finite = np.isfinite(truth) & (truth > 0)
    return truth[finite], r_idx[finite], t_idx[finite], p_idx[finite]


def plot_histogram(
    truth: np.ndarray,
    reconstruction: np.ndarray,
    metrics: dict,
    output_dir: Path,
) -> Path:
    valid = (
        np.isfinite(truth)
        & np.isfinite(reconstruction)
        & (truth > 0)
        & (reconstruction > 0)
    )
    x = np.log10(truth[valid])
    y = np.log10(reconstruction[valid])
    low = float(min(np.percentile(x, 0.2), np.percentile(y, 0.2)))
    high = float(max(np.percentile(x, 99.8), np.percentile(y, 99.8)))
    bins = np.linspace(low, high, 201)

    fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    histogram = axis.hist2d(x, y, bins=(bins, bins), norm=LogNorm(), cmap="magma")
    line = np.asarray([low, high])
    axis.plot(line, line, color="white", linewidth=1.5, linestyle="--", label="1:1")
    fit = metrics["fit_slope_log10"] * line + metrics["fit_intercept_log10"]
    axis.plot(line, fit, color="cyan", linewidth=1.5, label="log-space fit")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"log$_{10}$ PSI density [cm$^{-3}$]")
    axis.set_ylabel(r"log$_{10}$ SuNeRF density [cm$^{-3}$]")
    axis.set_title(f"PSI frame {FRAME_TOKEN}: volumetric density correlation")
    axis.text(
        0.03,
        0.97,
        (
            f"r (linear) = {metrics['pearson_r_linear']:.4f}\n"
            f"r (log10) = {metrics['pearson_r_log10']:.4f}\n"
            f"fit: y = {metrics['fit_slope_log10']:.3f}x "
            f"{metrics['fit_intercept_log10']:+.3f}\n"
            f"N = {metrics['n_points']:,}"
        ),
        transform=axis.transAxes,
        va="top",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    axis.legend(loc="lower right")
    fig.colorbar(histogram[3], ax=axis, label="samples per bin")
    output_path = output_dir / "density_correlation_histogram.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def evaluate_radial_slices(
    loader: ThomsonSuNeRFLoader,
    density: np.ndarray,
    radius: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    longitude_hci: np.ndarray,
    normalized_time: float,
    output_dir: Path,
) -> tuple[Path, list[dict]]:
    theta_indices = np.flatnonzero((theta >= 0.0) & (theta <= np.pi))
    phi_indices = np.flatnonzero((phi >= 0.0) & (phi < 2.0 * np.pi))
    latitude = np.pi / 2.0 - theta[theta_indices]
    longitude = longitude_hci[phi_indices]
    latitude_grid, longitude_grid = np.meshgrid(latitude, longitude, indexing="ij")

    slices = []
    slice_metrics = []
    for requested_radius in SLICE_RADII:
        radial_index = int(np.argmin(np.abs(radius - requested_radius)))
        actual_radius = float(radius[radial_index])
        truth = density[np.ix_(phi_indices, theta_indices, [radial_index])][..., 0].T
        truth = truth * MAS_RHO_TO_ELECTRON_CM3
        reconstruction = query_density(
            loader,
            np.full(latitude_grid.size, actual_radius),
            latitude_grid.reshape(-1),
            longitude_grid.reshape(-1),
            normalized_time,
        ).reshape(latitude_grid.shape)
        metrics = correlation_metrics(truth, reconstruction)
        metrics.update({"requested_radius_rsun": requested_radius, "actual_radius_rsun": actual_radius})
        slices.append((truth, reconstruction, phi[phi_indices], latitude))
        slice_metrics.append(metrics)

    fig, axes = plt.subplots(
        2,
        len(SLICE_RADII),
        figsize=(18, 7.2),
        constrained_layout=True,
        squeeze=False,
    )
    axes[0, 0].set_ylabel("PSI ground truth\nlatitude [deg]")
    axes[1, 0].set_ylabel("SuNeRF\nlatitude [deg]")
    for column, ((truth, reconstruction, slice_phi, slice_latitude), metrics) in enumerate(
        zip(slices, slice_metrics)
    ):
        positive = truth[np.isfinite(truth) & (truth > 0)]
        vmin = max(float(np.percentile(positive, 1.0)), np.finfo(np.float32).tiny)
        vmax = max(float(np.percentile(positive, 99.0)), vmin * 1.01)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        extent = [
            float(np.rad2deg(slice_phi.min())),
            float(np.rad2deg(slice_phi.max())),
            float(np.rad2deg(slice_latitude.min())),
            float(np.rad2deg(slice_latitude.max())),
        ]
        artist = None
        # theta grows southward, so reverse the first dimension to display
        # monotonically increasing latitude from bottom to top.
        for row, image in enumerate((truth[::-1], reconstruction[::-1])):
            artist = axes[row, column].imshow(
                image,
                origin="lower",
                extent=extent,
                aspect="auto",
                cmap="viridis",
                norm=norm,
                interpolation="none",
            )
            axes[row, column].set_xlabel("Carrington longitude [deg]")
            if column > 0:
                axes[row, column].set_yticklabels([])
        axes[0, column].set_title(
            f"r = {metrics['actual_radius_rsun']:.2f} R$_\\odot$\n"
            f"log r = {metrics['pearson_r_log10']:.3f}"
        )
        fig.colorbar(
            artist,
            ax=axes[:, column],
            orientation="horizontal",
            fraction=0.045,
            pad=0.08,
            label=r"density [cm$^{-3}$]",
        )
    fig.suptitle(f"PSI frame {FRAME_TOKEN}: radial density slices", fontsize=15)
    output_path = output_dir / "density_radial_slices.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, slice_metrics


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(
            "Degraded-data training has not completed "
            f"(missing {args.checkpoint_path})."
        )
    if not args.sunerf_path.is_file():
        raise FileNotFoundError(
            f"Run degraded-data training before evaluation: {args.sunerf_path}"
        )
    if not args.density_path.is_file():
        raise FileNotFoundError(
            "Run sunerf.data.psi.download_psi_density before evaluation: "
            f"{args.density_path}"
        )
    if not args.reference_tb_path.is_file():
        raise FileNotFoundError(f"Missing reference tB map: {args.reference_tb_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # This file is produced locally by save_thomson_sunerf and contains pickled
    # PyTorch modules, including the learned detector-degradation corrections.
    loader = ThomsonSuNeRFLoader(args.sunerf_path, trusted=True)
    density, radius, theta, phi = read_density(args.density_path)
    reference_map = Map(args.reference_tb_path)
    evaluation_time = reference_map.date.datetime
    normalized_time = float(loader.normalize_datetime(evaluation_time))
    longitude_hci = carrington_to_hci_longitudes(phi, evaluation_time)

    truth, r_idx, t_idx, p_idx = sample_volume(density, radius, theta, phi)
    reconstruction = query_density(
        loader,
        radius[r_idx],
        np.pi / 2.0 - theta[t_idx],
        longitude_hci[p_idx],
        normalized_time,
    )
    volume_metrics = correlation_metrics(truth, reconstruction)
    histogram_path = plot_histogram(
        truth,
        reconstruction,
        volume_metrics,
        args.output_dir,
    )
    slices_path, slice_metrics = evaluate_radial_slices(
        loader,
        density,
        radius,
        theta,
        phi,
        longitude_hci,
        normalized_time,
        args.output_dir,
    )

    metrics = {
        "frame": FRAME_TOKEN,
        "time": evaluation_time.isoformat(),
        "source_hdf": str(args.density_path),
        "density_conversion": (
            "n_e = MAS rho code units * 1e8 cm^-3 / he_rho, where "
            "he_rho = (1 + 4 f_He) / (1 + 2 f_He)"
        ),
        "helium_electron_fraction": HELIUM_ELECTRON_FRACTION,
        "helium_mass_per_electron": HELIUM_MASS_PER_ELECTRON,
        "composition_assumption": (
            "fully ionized pure hydrogen; the supplied HDF4 file has no "
            "composition metadata"
        ),
        "coordinate_frame": (
            "PSI phi interpreted as Carrington longitude and transformed to HCI at the "
            "evaluation time before querying SuNeRF"
        ),
        "random_seed": RANDOM_SEED,
        "radius_range_rsun": list(RADIUS_RANGE),
        "volume": volume_metrics,
        "radial_slices": slice_metrics,
    }
    metrics_path = args.output_dir / "density_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    print("Density evaluations:")
    print(f"  {histogram_path}")
    print(f"  {slices_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
