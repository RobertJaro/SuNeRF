#!/usr/bin/env python3
"""
Batch-preprocess PUNCH FITS images.

For each FITS file matched by --data_path, the script:
1) Loads extension 1 and extracts tB and pB (index 0 and 1).
2) Uses extension 1 header to construct SunPy maps.
3) Applies shared map preprocessing.
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


class PunchPrep:
    """Callable helper for multiprocessing conversion of PUNCH FITS files."""

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
    def _load_punch_maps(file_path):
        """
        Load PUNCH FITS file and return tB and pB as SunPy maps.

        Expects:
        - extension 1 data layout: [tB, pB, pBp]
        - extension 1 header has map metadata/WCS
        """
        try:
            data = fits.getdata(file_path, 1)
            header = fits.getheader(file_path, 1)
            tb = np.array(data[0], dtype=float, copy=True)
            pb = np.array(data[1], dtype=float, copy=True)
        except Exception as e:
            raise RuntimeError(f"Error loading PUNCH FITS file {file_path}: {e}")

        return Map(tb, header), Map(pb, header)

    def convert(self, file_path):
        """Load, preprocess, and save tB/pB maps for one FITS file."""
        basename = os.path.basename(file_path)
        tb_out = os.path.join(self.tb_out_path, basename)
        pb_out = os.path.join(self.pb_out_path, basename)

        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return tb_out, pb_out

        try:
            tb_map, pb_map = self._load_punch_maps(file_path)
            tb_map = self.map_preprocessor.prepare_map(tb_map)
            pb_map = self.map_preprocessor.prepare_map(pb_map)

            tb_map.save(tb_out, overwrite=True)
            pb_map.save(pb_out, overwrite=True)
        except Exception as e:
            print(f"[{os.getpid()}] ERROR in {basename}: {e}", flush=True)
            raise
        return tb_out, pb_out


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for PUNCH FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser, include_clip=True)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = PunchPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        **common_kwargs_from_args(args),
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        out_files = [
            pair
            for pair in tqdm(
                pool.imap_unordered(prepper.convert, files, chunksize=1),
                total=len(files),
                desc="Preprocessing PUNCH maps",
            )
        ]

    print(
        f"Preprocessed {len(out_files)} files. Saved tB to {os.path.join(args.out_path, 'tB')} and pB to "
        f"{os.path.join(args.out_path, 'pB')}."
    )


if __name__ == "__main__":
    main()
