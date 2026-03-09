#!/usr/bin/env python3
"""
Batch-preprocess PUNCH FITS images.

For each FITS file matched by --data_path, the script:
1) Loads extension 1 and extracts tB and pB (index 0 and 1).
2) Uses extension 1 header to construct SunPy maps.
3) Applies the shared coronagraph base preprocessing (_prep_coronagraph_map).
4) Saves results into separate folders:
   - <out_path>/tB/<basename>.fits
   - <out_path>/pB/<basename>.fits
"""

import argparse
import multiprocessing
import os
from glob import glob

import numpy as np
from astropy import units as u
from astropy.io import fits
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.coronagraph.prep_coronagraph import _prep_coronagraph_map


class PunchPrep:
    """Callable helper for multiprocessing conversion of PUNCH FITS files."""

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, resize=None, clip_max=None):
        self.out_path = out_path
        self.overwrite = overwrite
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.resize = resize
        self.clip_max = clip_max

        self.tb_out_path = os.path.join(out_path, "tB")
        self.pb_out_path = os.path.join(out_path, "pB")
        os.makedirs(self.tb_out_path, exist_ok=True)
        os.makedirs(self.pb_out_path, exist_ok=True)

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

    def _prepare_map(self, s_map):
        s_map = _prep_coronagraph_map(s_map, occ_min=self.occ_min, occ_max=self.occ_max)
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.clip_max is not None:
            s_map.data[:] = np.clip(s_map.data, a_max=self.clip_max, a_min=None)
        return s_map

    def convert(self, file_path):
        """Load, preprocess, and save tB/pB maps for one FITS file."""
        basename = os.path.basename(file_path)
        tb_out = os.path.join(self.tb_out_path, basename)
        pb_out = os.path.join(self.pb_out_path, basename)

        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return tb_out, pb_out

        tb_map, pb_map = self._load_punch_maps(file_path)
        tb_map = self._prepare_map(tb_map)
        pb_map = self._prepare_map(pb_map)

        tb_map.save(tb_out, overwrite=True)
        pb_map.save(pb_out, overwrite=True)
        return tb_out, pb_out


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", type=str, required=True, help="Glob pattern for PUNCH FITS files.")
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

    prepper = PunchPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        occ_min=args.occ_min * u.arcsec if args.occ_min is not None else None,
        occ_max=args.occ_max * u.arcsec if args.occ_max is not None else None,
        resize=args.resize,
        clip_max=args.clip_max,
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        out_files = [
            pair
            for pair in tqdm(
                pool.imap(prepper.convert, files),
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
