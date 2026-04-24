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
from astropy import units as u
from astropy.io import fits
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.coronagraph.prep_coronagraph import _prep_coronagraph_map


class PunchPamPrep:
    """Callable helper for multiprocessing conversion of PUNCH PAM FITS files."""

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, max_radius=None, resize=None, value_min=None,
                 value_max=None):
        self.out_path = out_path
        self.overwrite = overwrite
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.max_radius = max_radius
        self.resize = resize
        self.value_min = value_min
        self.value_max = value_max

        self.tb_out_path = os.path.join(out_path, "tB")
        self.pb_out_path = os.path.join(out_path, "pB")
        os.makedirs(self.tb_out_path, exist_ok=True)
        os.makedirs(self.pb_out_path, exist_ok=True)

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

    def _prepare_map(self, s_map):
        s_map = _prep_coronagraph_map(s_map, occ_min=self.occ_min, occ_max=self.occ_max, max_radius=self.max_radius)
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.value_min is not None:
            s_map.data[s_map.data < self.value_min] = np.nan
        if self.value_max is not None:
            s_map.data[s_map.data > self.value_max] = np.nan
        return s_map

    def convert(self, file_path):
        """Load, preprocess, and save one PAM tB/pB pair."""
        basename = os.path.basename(file_path)
        tb_out = os.path.join(self.tb_out_path, basename)
        pb_out = os.path.join(self.pb_out_path, basename)
        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return tb_out, pb_out

        try:
            tb_map, pb_map = self._load_punch_pam_maps(file_path)
            tb_map = self._prepare_map(tb_map)
            pb_map = self._prepare_map(pb_map)
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
    parser.add_argument(
        "--occ_min",
        type=float,
        default=None,
        help="Minimum occulter radius in arcseconds.",
    )
    parser.add_argument(
        "--occ_max",
        type=float,
        default=None,
        help="Maximum occulter radius in arcseconds.",
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=None,
        help="Maximum projected radius in solar radii.",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument("--value_min", type=float, default=None, help="Optional minimum allowed data value.")
    parser.add_argument("--value_max", type=float, default=None, help="Optional maximum allowed data value.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize to (width height) in pixels.",
    )
    args = parser.parse_args()

    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")

    prepper = PunchPamPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        occ_min=args.occ_min * u.arcsec if args.occ_min is not None else None,
        occ_max=args.occ_max * u.arcsec if args.occ_max is not None else None,
        max_radius=args.max_radius * u.solRad if args.max_radius is not None else None,
        resize=args.resize,
        value_min=args.value_min,
        value_max=args.value_max,
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
