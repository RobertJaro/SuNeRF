#!/usr/bin/env python3
"""Extract one daily-min detector noise mask from one image sequence."""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
from sunpy.map import Map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Glob for one detector/product image sequence.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output FITS mask path; matching .npy and .png files are also written.",
    )
    parser.add_argument(
        "--min-frames-per-day",
        type=int,
        default=6,
        help="Minimum frames required to form a daily median.",
    )
    parser.add_argument(
        "--smooth-angle-deg",
        type=float,
        default=0.0,
        help="Gaussian smoothing width in position angle around the Sun; 0 keeps the mask.",
    )
    parser.add_argument(
        "--smooth-fraction",
        type=float,
        default=0.0,
        help="Gaussian smoothing width along the radius as a fraction of the image width.",
    )
    parser.add_argument(
        "--smooth-percentile",
        type=float,
        default=0.0,
        help="Running percentile over the position angles; 0 is the running minimum.",
    )
    return parser.parse_args()


def extract_noise_mask(
    input_pattern: str,
    output_path: Path,
    min_frames_per_day: int = 6,
    smooth_angle_deg: float = 0.0,
    smooth_fraction: float = 0.0,
    smooth_percentile: float = 0.0,
) -> None:
    """Write one daily-min mask while retaining its detector WCS."""
    if min_frames_per_day < 1:
        raise ValueError("min_frames_per_day must be positive.")
    if smooth_angle_deg < 0 or not 0 <= smooth_fraction < 1:
        raise ValueError("Require smooth_angle_deg >= 0 and smooth_fraction in [0, 1).")
    if output_path.suffix.lower() not in {".fits", ".fit", ".fts"}:
        raise ValueError("output_path must use a FITS extension.")

    reference_files = sorted(Path(path) for path in glob.glob(input_pattern))
    reference_files = [path for path in reference_files if path.is_file()]
    if not reference_files:
        raise FileNotFoundError(f"No detector images matched: {input_pattern}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    array_path = output_path.with_suffix(".npy")
    plot_path = output_path.with_suffix(".png")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sunerf.data.coronagraph.compute_correction",
            "--type",
            "daily-min",
            "--input",
            input_pattern,
            "--output",
            str(array_path),
            "--plot-output",
            str(plot_path),
            "--min-frames-per-day",
            str(min_frames_per_day),
            "--smooth-angle-deg",
            str(smooth_angle_deg),
            "--smooth-fraction",
            str(smooth_fraction),
            "--smooth-percentile",
            str(smooth_percentile),
        ],
        check=True,
    )

    reference_map = Map(reference_files[0])
    mask = np.asarray(np.load(array_path), dtype=np.float32)
    if mask.shape != reference_map.data.shape:
        raise ValueError(
            f"Noise mask shape {mask.shape} does not match reference WCS shape "
            f"{reference_map.data.shape}."
        )
    Map(mask, reference_map.meta).save(output_path, overwrite=True)


def main() -> None:
    args = parse_args()
    extract_noise_mask(
        args.input, args.output, args.min_frames_per_day,
        args.smooth_angle_deg, args.smooth_fraction, args.smooth_percentile,
    )
    print(f"Daily-min detector mask: {args.output}")


if __name__ == "__main__":
    main()
