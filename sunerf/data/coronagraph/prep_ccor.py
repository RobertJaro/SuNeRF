#!/usr/bin/env python3
"""
Batch-preprocess CCOR FITS images.

For each FITS file matched by --data_path, the script:
1) Loads image data from extension 1.
2) Loads mask from extension 2 and sets masked pixels (mask != 0) to NaN.
3) Builds a SunPy map from extension 1 header.
4) Applies shared map preprocessing.
5) Saves the result to --out_path with the same basename.
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
)


def _header_flag_is_true(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        return normalized in {"T", "TRUE", "1", "Y", "YES"}
    return False


def _validate_ccor_quality(ext_header, primary_header, file_path):
    quality_keys = ("ISVIABLE", "ISNORMAL", "DCMPRS_Q", "IMGBLK_Q", "BORSGT_Q", "EARFOV_Q")
    failed = []
    for key in quality_keys:
        value = ext_header.get(key, primary_header.get(key))
        if not _header_flag_is_true(value):
            failed.append(f"{key}={value!r}")
    if failed:
        print(f"Skipping invalid CCOR file {file_path}: " + ", ".join(failed))
        return False
    return True


def _rotate_ccor_map(s_map, primary_header):
    """Undo any recorded 90-degree CCOR rotation through SunPy's map rotation."""
    nrot90 = s_map.meta.get("NROT90", primary_header.get("NROT90", 0))
    try:
        nrot90 = int(nrot90) % 4
    except (TypeError, ValueError):
        nrot90 = 0

    if nrot90 == 0:
        return s_map

    rotated_map = s_map.rotate(
        angle=-nrot90 * 90 * u.deg,
        order=0,
        missing=np.nan,
    )
    rotated_map.meta["NROT90"] = 0
    return rotated_map


def _load_ccor_map(file_path):
    """Load CCOR FITS and apply extension-2 mask to extension-1 data."""
    try:
        data = fits.getdata(file_path, 1)
        mask = fits.getdata(file_path, 2)
        header = fits.getheader(file_path, 1)
        primary_header = fits.getheader(file_path, 0)

        if not _validate_ccor_quality(header, primary_header, file_path):
            return None

        masked_data = np.array(data, dtype=float, copy=True)
        masked_data[np.array(mask) != 0] = np.nan

        if "rsun_ref" not in header:
            header["rsun_ref"] = constants.radius.to_value(u.m)
    except Exception as e:
        raise RuntimeError(f"Error loading CCOR FITS file {file_path}: {e}")

    s_map = Map(masked_data, header)
    return _rotate_ccor_map(s_map, primary_header)


class CCORPrep:
    """Callable helper for multiprocessing conversion of CCOR FITS files."""

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, max_radius=None, resize=None,
                 clip_min=None, clip_max=None, value_min=None, value_max=None,
                 filter_bright_objects=False, bright_object_threshold=10.0):
        self.out_path = out_path
        self.overwrite = overwrite
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
        out_path = os.path.join(self.out_path, os.path.basename(file_path))
        if os.path.exists(out_path) and not self.overwrite:
            return out_path

        s_map = _load_ccor_map(file_path)
        if s_map is None:
            return None
        s_map = self.map_preprocessor.prepare_map(s_map)

        s_map.save(out_path, overwrite=True)
        return out_path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for CCOR FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser, include_value_limits=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = CCORPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        **common_kwargs_from_args(args),
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(prepper.convert, files),
                total=len(files),
                desc="Preprocessing CCOR maps",
            )
        )
    out_files = [f for f in results if f is not None]
    skipped = len(results) - len(out_files)

    print(f"Preprocessed {len(out_files)} maps saved to {args.out_path}. Skipped invalid: {skipped}.")


if __name__ == "__main__":
    main()
