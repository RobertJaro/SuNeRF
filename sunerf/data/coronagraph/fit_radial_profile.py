#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

from sunerf.data.ray_sampling import hpc_impact_parameter


def fit_radial_profile(files: list[str], degree: int) -> tuple[np.ndarray, int]:
    all_r = []
    all_log_brightness = []

    for file_path in tqdm(files, desc="Processing FITS"):
        s_map = Map(file_path)
        coords = all_coordinates_from_map(s_map)

        r = hpc_impact_parameter(coords.Tx, coords.Ty, s_map.dsun).to_value("R_sun")
        brightness = np.asarray(s_map.data, dtype=float)

        valid = (
            np.isfinite(r)
            & np.isfinite(brightness)
            & (r > 0)
            & (brightness > 0)
        )
        if not np.any(valid):
            continue

        all_r.append(r[valid].ravel())
        all_log_brightness.append(np.log(brightness[valid].ravel()))

    if not all_r:
        raise ValueError("No valid positive radius/brightness samples found in the provided FITS files.")

    r_values = np.concatenate(all_r)
    log_brightness_values = np.concatenate(all_log_brightness)

    coeffs = np.polyfit(r_values, log_brightness_values, deg=degree)
    return coeffs, r_values.size


def plot_normalized_frame(file_path: str, coeffs: np.ndarray, output_path: Path) -> None:
    s_map = Map(file_path)
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)

    r = hpc_impact_parameter(coords.Tx, coords.Ty, s_map.dsun).to_value("R_sun")
    brightness = np.asarray(s_map.data, dtype=float)
    fitted_brightness = np.exp(np.polyval(coeffs, r))
    fitted_brightness = np.clip(fitted_brightness, 1e-12, None)

    normalized = brightness / fitted_brightness
    stretched = np.arcsinh(normalized / 1e-2) / np.arcsinh(1/1e-2)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(stretched, origin="lower", cmap="gray")
    fig.colorbar(im, ax=ax, label="Normalized brightness")
    ax.set_title("First frame normalized by fitted radial profile")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a polynomial to log(brightness) as a function of projected radius "
            "(exact line-of-sight impact parameter) across all FITS files."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input glob pattern for FITS files (e.g. '/path/to/*.fits').",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file path for polynomial coefficients.",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="Polynomial degree for np.polyfit (default: 3).",
    )
    parser.add_argument(
        "--plot-output",
        required=True,
        help="Output path for normalized first-frame plot (e.g. '/path/normalized.png').",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise FileNotFoundError(f"No FITS files found for input: {args.input}")

    coeffs, n_samples = fit_radial_profile(files, degree=args.degree)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, coeffs)

    print(f"Processed files: {len(files)}")
    print(f"Samples used: {n_samples}")
    print(f"Saved coefficients to: {output_path}")
    print(f"Polynomial coefficients (highest degree first): {coeffs}")

    plot_normalized_frame(files[50], coeffs, Path(args.plot_output))


if __name__ == "__main__":
    main()
