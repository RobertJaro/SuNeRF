#!/usr/bin/env python3

import argparse
import glob
from pathlib import Path

import numpy as np
from astropy import units as u
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map


def compute_projected_radius_stats(
    file_path: str,
    finite_only: bool = True,
    positive_only: bool = False,
    min_value: float | None = None,
) -> dict[str, float | str | int]:
    s_map = Map(file_path)
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    projected_radius_arcsec = ((coords.Tx**2 + coords.Ty**2) ** 0.5).to_value(u.arcsec)
    projected_radius = (projected_radius_arcsec * u.arcsec / s_map.rsun_obs).to_value(u.one)

    data = np.asarray(s_map.data, dtype=float)
    valid = np.isfinite(projected_radius)

    if finite_only:
        valid &= np.isfinite(data)
    if positive_only:
        valid &= data > 0
    if min_value is not None:
        valid &= data >= min_value

    if not np.any(valid):
        raise ValueError(f"No valid pixels found for {file_path}")

    valid_radius = projected_radius[valid]
    valid_radius_arcsec = projected_radius_arcsec[valid]
    max_index_flat = np.nanargmax(np.where(valid, projected_radius, np.nan))
    max_index = np.unravel_index(max_index_flat, projected_radius.shape)

    return {
        "file": file_path,
        "n_valid_pixels": int(valid.sum()),
        "min_rsun": float(np.nanmin(valid_radius)),
        "max_rsun": float(np.nanmax(valid_radius)),
        "mean_rsun": float(np.nanmean(valid_radius)),
        "min_arcsec": float(np.nanmin(valid_radius_arcsec)),
        "max_arcsec": float(np.nanmax(valid_radius_arcsec)),
        "mean_arcsec": float(np.nanmean(valid_radius_arcsec)),
        "max_row": int(max_index[0]),
        "max_col": int(max_index[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute plane-of-sky projected radial distance statistics for FITS files "
            "and report the maximum extent in units of observed solar radii."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input FITS file path or glob pattern.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Restrict valid pixels to strictly positive data values.",
    )
    parser.add_argument(
        "--min-value",
        type=float,
        default=None,
        help="Restrict valid pixels to data >= this value.",
    )
    parser.add_argument(
        "--all-pixels",
        action="store_true",
        help="Ignore NaN filtering on the image data and use all coordinate-valid pixels.",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files and Path(args.input).is_file():
        files = [args.input]
    if not files:
        raise FileNotFoundError(f"No FITS files found for input: {args.input}")

    for file_path in files:
        stats = compute_projected_radius_stats(
            file_path,
            finite_only=not args.all_pixels,
            positive_only=args.positive_only,
            min_value=args.min_value,
        )
        print(f"File: {stats['file']}")
        print(f"  valid_pixels: {stats['n_valid_pixels']}")
        print(f"  min_projected_radius_rsun: {stats['min_rsun']:.6f}")
        print(f"  max_projected_radius_rsun: {stats['max_rsun']:.6f}")
        print(f"  mean_projected_radius_rsun: {stats['mean_rsun']:.6f}")
        print(f"  min_projected_radius_arcsec: {stats['min_arcsec']:.6f}")
        print(f"  max_projected_radius_arcsec: {stats['max_arcsec']:.6f}")
        print(f"  mean_projected_radius_arcsec: {stats['mean_arcsec']:.6f}")
        print(f"  max_radius_pixel_rowcol: ({stats['max_row']}, {stats['max_col']})")


if __name__ == "__main__":
    main()
