#!/usr/bin/env python3
"""
Batch-preprocess PUNCH PAM FITS images.

For each FITS file matched by --data_path, the script:
1) Loads extension 1 and derives tB and pB from the three image planes.
2) Uses extension 1 header to construct SunPy maps.
3) Applies the shared coronagraph base preprocessing.
4) Saves results into separate folders:
   - <out_path>/tB/<basename>.fits
   - <out_path>/pB/<basename>.fits
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
    ensure_tb_pb_output_dirs,
)


class PunchPamPrep:
    """Callable helper for multiprocessing conversion of PUNCH PAM FITS files."""

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

        self.tb_out_path, self.pb_out_path = ensure_tb_pb_output_dirs(out_path)

    @staticmethod
    def _load_punch_pam_maps(file_path):
        """Load a PUNCH PAM FITS file and derive tB and pB from extension 1."""
        try:
            data = np.array(fits.getdata(file_path, 1), dtype=float, copy=True)
            header = fits.getheader(file_path, 1)
        except Exception as exc:
            raise RuntimeError(f"Error loading PUNCH PAM FITS file {file_path}: {exc}")

        tb = np.array(data[0], dtype=float, copy=True)
        pb = np.sqrt(np.square(data[1]) + np.square(data[2]))
        return Map(tb, header), Map(pb, header)

    def convert(self, file_path):
        """Load, preprocess, and save one PAM tB/pB pair."""
        basename = os.path.basename(file_path)
        tb_out = os.path.join(self.tb_out_path, basename)
        pb_out = os.path.join(self.pb_out_path, basename)
        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return tb_out, pb_out

        try:
            tb_map, pb_map = self._load_punch_pam_maps(file_path)
            tb_map = self.map_preprocessor.prepare_map(tb_map)
            pb_map = self.map_preprocessor.prepare_map(pb_map)
            tb_map.save(tb_out, overwrite=True)
            pb_map.save(pb_out, overwrite=True)
        except Exception as exc:
            print(f"[{os.getpid()}] ERROR in {basename}: {exc}", flush=True)
            raise
        return tb_out, pb_out


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for PUNCH PAM FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser, include_max_radius=True, include_value_limits=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = PunchPamPrep(
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
                desc="Preprocessing PUNCH PAM maps",
            )
        ]

    print(
        f"Preprocessed {len(out_files)} files. Saved tB to {os.path.join(args.out_path, 'tB')} and pB to "
        f"{os.path.join(args.out_path, 'pB')}."
    )


if __name__ == "__main__":
    main()
