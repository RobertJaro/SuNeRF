#!/usr/bin/env python3
"""
Batch-preprocess PUNCH CAM FITS images.

For each FITS file matched by --data_path, the script:
1) Loads extension 1 as the primary image array.
2) Uses extension 1 header to construct a SunPy map.
3) Applies the shared coronagraph base preprocessing.
4) Saves the result to --out_path with the same basename.
"""

import argparse
import multiprocessing
import os
from glob import glob

import numpy as np
from astropy.io import fits
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    add_common_prep_arguments,
    common_kwargs_from_args,
)


class PunchCamPrep:
    """Callable helper for multiprocessing conversion of PUNCH CAM FITS files."""

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

        os.makedirs(self.out_path, exist_ok=True)

    @staticmethod
    def _load_punch_cam_map(file_path):
        """Load a PUNCH CAM FITS file from extension 1."""
        try:
            data = np.array(fits.getdata(file_path, 1), dtype=float, copy=True)
            header = fits.getheader(file_path, 1)
        except Exception as exc:
            raise RuntimeError(f"Error loading PUNCH CAM FITS file {file_path}: {exc}")

        return Map(data, header)

    def convert(self, file_path):
        """Load, preprocess, and save one CAM map."""
        out_path = os.path.join(self.out_path, os.path.basename(file_path))
        if os.path.exists(out_path) and not self.overwrite:
            return out_path

        try:
            cam_map = self._load_punch_cam_map(file_path)
            cam_map = self.map_preprocessor.prepare_map(cam_map)
            cam_map.save(out_path, overwrite=True)
        except Exception as exc:
            print(f"[{os.getpid()}] ERROR in {os.path.basename(file_path)}: {exc}", flush=True)
            raise
        return out_path


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for PUNCH CAM FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser, include_value_limits=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
    )
    args = parser.parse_args()

    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = PunchCamPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        **common_kwargs_from_args(args),
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        out_files = [
            out_file
            for out_file in tqdm(
                pool.imap_unordered(prepper.convert, files, chunksize=1),
                total=len(files),
                desc="Preprocessing PUNCH CAM maps",
            )
        ]

    print(f"Preprocessed {len(out_files)} files. Saved outputs to {args.out_path}.")


if __name__ == "__main__":
    main()
