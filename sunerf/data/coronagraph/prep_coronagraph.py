#!/usr/bin/env python3
"""

Batch-preprocess FITS images.

For each FITS file matched by --data_path, the script:
1) Loads the file as a SunPy Map.
2) Computes helioprojective coordinates for each pixel.
3) Masks pixels inside the configured occulter/radius bounds.
4) Optionally clips data to configured minimum/maximum values.
5) Saves the result to --out_path with the same basename.

Example
-------
python prep_coronagraph.py --data_path '/path/to/data/*.fits' --out_path '/path/to/output/' --occ_min 5500
"""

import argparse
import multiprocessing
import os
from glob import glob

import numpy as np
from astropy import units as u
from astropy.io import fits
from sunpy.map import Map
from sunpy.sun import constants
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    add_common_prep_arguments,
    common_kwargs_from_args,
    parse_duration,
    sample_files_at_cadence,
)


def _load_map(file_path):
    """
    Load a Metis FITS file as a SunPy Map.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.

    Returns
    -------
    sunpy.map.Map
        SunPy map constructed from FITS data and header.
    """
    try:
        data = fits.getdata(file_path)
        header = fits.getheader(file_path)
        if 'rsun_ref' not in header:
            header['rsun_ref'] = constants.radius.to_value(u.m)  # Add solar radius if missing
    except Exception as e:
        raise RuntimeError(f"Error loading FITS file {file_path}: {e}")
    return Map(data, header)


def mask_radial_line(data, center, angle_deg, halfwidth_deg=2.0, r_min=0.0, r_max=np.inf, fill=np.nan):
    """
    Mask pixels near a radial line (through `center`) at a given position angle.

    Parameters
    ----------
    data : ndarray (H, W)
        Image.
    center : tuple[float, float]
        (cx, cy) in pixel coordinates, with x=column, y=row.
    angle_deg : float
        Line angle in degrees, measured CCW from +x axis (image columns).
        If you want solar-style PA (CCW from +y / north), see note below.
    halfwidth_deg : float
        Half-width of the masked angular band (degrees).
    r_min, r_max : float
        Optional radial limits (pixels) to only mask between these radii.
    fill : float
        Value to write into masked pixels (e.g., np.nan or 0).

    Returns
    -------
    out : ndarray
        Copy of `data` with masked pixels set to `fill`.
    mask : ndarray (bool)
        True where pixels were masked.
    """
    out = np.array(data, copy=True)
    cy, cx = center[1], center[0]  # keep user convention: (cx, cy)
    y, x = np.indices(out.shape)
    dx = x - cx
    dy = y - cy

    theta = np.arctan2(dy, dx)  # [-pi, pi], 0 along +x
    theta0 = np.deg2rad(angle_deg)

    # Smallest signed angular difference in [-pi, pi]
    dtheta = np.arctan2(np.sin(theta - theta0), np.cos(theta - theta0))

    r = np.hypot(dx, dy)

    mask = (np.abs(dtheta) <= np.deg2rad(halfwidth_deg)) & (r >= r_min) & (r <= r_max)
    out[mask] = fill
    return out, mask


class CoronagraphPrep:
    """
    Callable helper for multiprocessing conversion of coronagraph FITS files.

    Parameters
    ----------
    out_path : str
        Output directory.
    overwrite : bool, optional
        If False, existing outputs are skipped. Default is True.
    occ_min, occ_max : astropy.units.Quantity, optional
        Inner and outer radial bounds used for masking.
    """

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, max_radius=None, resize=None,
                 clip_min=None, clip_max=None, value_min=None, value_max=None, map_prep_func=None,
                 filter_bright_objects=False, bright_object_threshold=10.0):
        self.out_path = out_path
        self.overwrite = overwrite
        self.map_prep_func = map_prep_func
        self.map_preprocessor = MapPreprocessor(
            occ_min=occ_min,
            occ_max=occ_max,
            max_radius=max_radius,
            resize=resize,
            clip_min=clip_min,
            clip_max=clip_max,
            value_min=value_min,
            value_max=value_max,
            filter_bright_objects=filter_bright_objects,
            bright_object_threshold=bright_object_threshold,
        )

    def convert(self, file_path):
        """
        Load, preprocess, and save one coronagraph FITS file.

        Parameters
        ----------
        file_path : str
            Input FITS file path.

        Returns
        -------
        str
            Output file path.
        """
        out_path = os.path.join(self.out_path, os.path.basename(file_path))
        if os.path.exists(out_path) and not self.overwrite:
            print(f"File {out_path} already exists. Skipping.")
            return out_path

        s_map = _load_map(file_path)
        if self.map_prep_func is not None:
            s_map = self.map_prep_func(s_map)
        s_map = self.map_preprocessor.prepare_map(s_map)
        s_map.save(out_path, overwrite=True)
        return out_path


def main():
    """
    CLI entry point for batch preprocessing.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser, include_clip=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--cadence",
        type=parse_duration,
        default=None,
        help="Optional fixed sampling cadence like 15m, 1h, or 30s.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")
    if args.cadence is not None:
        original_count = len(files)
        files = sample_files_at_cadence(files, args.cadence)
        print(
            f"Cadence sampling kept {len(files)} of {original_count} files "
            f"at {args.cadence} spacing."
        )

    prepper = CoronagraphPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        **common_kwargs_from_args(args),
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        out_files = [
            f
            for f in tqdm(
                pool.imap(prepper.convert, files),
                total=len(files),
                desc="Preprocessing maps",
            )
        ]

    print(f"Preprocessed {len(out_files)} maps saved to {args.out_path}.")


if __name__ == "__main__":
    main()
