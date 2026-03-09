#!/usr/bin/env python3
"""
Batch-preprocess CCOR FITS images.

For each FITS file matched by --data_path, the script:
1) Loads image data from extension 1.
2) Loads mask from extension 2 and sets masked pixels (mask != 0) to NaN.
3) Builds a SunPy map from extension 1 header.
4) Applies shared coronagraph preprocessing (_prep_coronagraph_map).
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

from sunerf.data.coronagraph.prep_coronagraph import _prep_coronagraph_map


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
    quality_keys = ("ISVIABLE", "ISNORMAL", "DCMPRS_Q", "IMGBLK_Q")
    failed = []
    for key in quality_keys:
        value = ext_header.get(key, primary_header.get(key))
        if not _header_flag_is_true(value):
            failed.append(f"{key}={value!r}")
    if failed:
        print(f"Skipping invalid CCOR file {file_path}: " + ", ".join(failed))
        return False
    return True


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

    return Map(masked_data, header)


class CCORPrep:
    """Callable helper for multiprocessing conversion of CCOR FITS files."""

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, resize=None, clip_max=None):
        self.out_path = out_path
        self.overwrite = overwrite
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.resize = resize
        self.clip_max = clip_max

    def convert(self, file_path):
        out_path = os.path.join(self.out_path, os.path.basename(file_path))
        if os.path.exists(out_path) and not self.overwrite:
            return out_path

        s_map = _load_ccor_map(file_path)
        if s_map is None:
            return None
        s_map = _prep_coronagraph_map(s_map, occ_min=self.occ_min, occ_max=self.occ_max)

        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.clip_max is not None:
            s_map.data[:] = np.clip(s_map.data, a_max=self.clip_max, a_min=None)

        s_map.save(out_path, overwrite=True)
        return out_path


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", type=str, required=True, help="Glob pattern for CCOR FITS files.")
    p.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    p.add_argument(
        "--occ_min",
        type=float,
        default=None,
        help="Minimum occulter radius in arcseconds.",
    )
    p.add_argument(
        "--occ_max",
        type=float,
        default=None,
        help="Maximum occulter radius in arcseconds.",
    )
    p.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    p.add_argument(
        "--clip_max",
        type=float,
        default=None,
        help="Optional maximum value to clip data to.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize to (width height) in pixels.",
    )
    args = p.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = CCORPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        occ_min=args.occ_min * u.arcsec if args.occ_min is not None else None,
        occ_max=args.occ_max * u.arcsec if args.occ_max is not None else None,
        resize=args.resize,
        clip_max=args.clip_max,
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
