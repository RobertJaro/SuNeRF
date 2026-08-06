#!/usr/bin/env python3
"""
Batch-preprocess paired STEREO/COR tB and pB FITS images.

For each matched tB/pB pair:
1) Validate both files before any output is written.
2) Require SEB_PROG='NORMAL' in both headers.
3) Treat values <= 0 as invalid and skip the pair if either map has more than
   the configured NaN fraction.
4) Apply the shared coronagraph preprocessing to both maps.
5) Save outputs to:
   - <out_path>/tB/<basename>.fits
   - <out_path>/pB/<basename>.fits
"""

import argparse
import multiprocessing
import os
import re
from glob import glob

import numpy as np
from astropy import units as u
from astropy.io import fits
from sunpy.map import Map
from sunpy.sun import constants
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    add_cadence_argument,
    add_common_prep_arguments,
    common_kwargs_from_args,
    ensure_tb_pb_output_dirs,
    select_items_by_time,
    should_write_output,
)


PAIR_STEM_RE = re.compile(r"^(?P<prefix>\d{8}_\d{6})_1(?P<kind>[PB])(?P<suffix>.+)$")


def normalize_pair_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = PAIR_STEM_RE.match(stem)
    if match is None:
        return stem
    return f"{match.group('prefix')}_1X{match.group('suffix')}"


def collect_pairs(tb_path: str, pb_path: str):
    tb_files = {normalize_pair_stem(path): path for path in sorted(glob(tb_path))}
    pb_files = {normalize_pair_stem(path): path for path in sorted(glob(pb_path))}
    pair_keys = sorted(set(tb_files) & set(pb_files))
    if not pair_keys:
        raise RuntimeError(f"No matching FITS pairs found for {tb_path} and {pb_path}")
    return [(tb_files[key], pb_files[key]) for key in pair_keys]


def header_is_normal(header) -> bool:
    return str(header.get("SEB_PROG", "")).strip().upper() == "NORMAL"


def invalid_pixel_fraction(data) -> float:
    """Count NaN and non-positive pixels without modifying the input data."""
    if not data.size:
        return 1.0
    return float(np.count_nonzero(np.isnan(data) | (data <= 0)) / data.size)


def load_stereo_map(file_path: str):
    try:
        data = np.array(fits.getdata(file_path), dtype=float, copy=True)
        header = fits.getheader(file_path)
        if "rsun_ref" not in header:
            header["rsun_ref"] = constants.radius.to_value(u.m)
    except Exception as exc:
        raise RuntimeError(f"Error loading FITS file {file_path}: {exc}")
    return data, header, Map(data, header)


class StereoCorPrep:
    def __init__(self, out_path, overwrite=False, nan_threshold=0.5, **preprocess_kwargs):
        self.out_path = out_path
        self.overwrite = overwrite
        self.nan_threshold = nan_threshold
        self.map_preprocessor = MapPreprocessor(**preprocess_kwargs)

        self.tb_out_path, self.pb_out_path = ensure_tb_pb_output_dirs(out_path)

    def _validate_pair(self, tb_path: str, pb_path: str):
        tb_data, tb_header, tb_map = load_stereo_map(tb_path)
        pb_data, pb_header, pb_map = load_stereo_map(pb_path)

        reasons = []
        if not header_is_normal(tb_header):
            reasons.append(f"tB SEB_PROG={tb_header.get('SEB_PROG')!r}, required 'NORMAL'")
        if not header_is_normal(pb_header):
            reasons.append(f"pB SEB_PROG={pb_header.get('SEB_PROG')!r}, required 'NORMAL'")

        # Non-positive values count toward file rejection, but are not replaced
        # in the maps that continue through preprocessing.
        tb_nan_fraction = invalid_pixel_fraction(tb_data)
        pb_nan_fraction = invalid_pixel_fraction(pb_data)

        if tb_nan_fraction > self.nan_threshold:
            reasons.append(f"tB NaN={tb_nan_fraction:.3f}")
        if pb_nan_fraction > self.nan_threshold:
            reasons.append(f"pB NaN={pb_nan_fraction:.3f}")

        return reasons, tb_map, pb_map

    def convert(self, pair):
        tb_path, pb_path = pair
        tb_out = os.path.join(self.tb_out_path, os.path.basename(tb_path))
        pb_out = os.path.join(self.pb_out_path, os.path.basename(pb_path))
        write_tb = should_write_output(tb_out, self.overwrite)
        write_pb = should_write_output(pb_out, self.overwrite)

        # Validate every input pair before considering existing outputs so that
        # the observing-program requirement cannot be bypassed by --no-overwrite.
        reasons, tb_map, pb_map = self._validate_pair(tb_path, pb_path)
        if reasons:
            return {
                "status": "rejected",
                "tb_path": tb_path,
                "pb_path": pb_path,
                "reason": ", ".join(reasons),
            }

        if not write_tb and not write_pb:
            return {"status": "skipped_existing", "tb_out": tb_out, "pb_out": pb_out}

        try:
            tb_map = self.map_preprocessor.prepare_map(tb_map)
            pb_map = self.map_preprocessor.prepare_map(pb_map)
            if write_tb:
                tb_map.save(tb_out, overwrite=self.overwrite)
            if write_pb:
                pb_map.save(pb_out, overwrite=self.overwrite)
        except Exception as exc:
            print(
                f"[{os.getpid()}] ERROR in {os.path.basename(tb_path)} / {os.path.basename(pb_path)}: {exc}",
                flush=True,
            )
            raise

        return {"status": "processed", "tb_out": tb_out, "pb_out": pb_out}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tb_path", type=str, required=True, help="Glob pattern for STEREO/COR tB FITS files.")
    parser.add_argument("--pb_path", type=str, required=True, help="Glob pattern for STEREO/COR pB FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    add_common_prep_arguments(parser)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
    )
    add_cadence_argument(parser)
    parser.add_argument(
        "--nan_threshold",
        type=float,
        default=0.5,
        help="Reject a pair if either map exceeds this NaN fraction after values <= 0 are treated as invalid.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    pairs = collect_pairs(args.tb_path, args.pb_path)
    if args.start is not None or args.end is not None or args.cadence is not None:
        original_count = len(pairs)
        pairs = select_items_by_time(
            pairs,
            start=args.start,
            end=args.end,
            cadence=args.cadence,
        )
        print(f"Time selection kept {len(pairs)} of {original_count} pairs.")

    prepper = StereoCorPrep(
        args.out_path,
        overwrite=args.overwrite,
        **common_kwargs_from_args(args),
        nan_threshold=args.nan_threshold,
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(prepper.convert, pairs, chunksize=1),
                total=len(pairs),
                desc="Preprocessing STEREO/COR pairs",
            )
        )

    processed = [result for result in results if result["status"] == "processed"]
    skipped_existing = [result for result in results if result["status"] == "skipped_existing"]
    rejected = [result for result in results if result["status"] == "rejected"]

    for result in rejected:
        print(
            f"Rejected {result['tb_path']} and {result['pb_path']}: {result['reason']}"
        )

    print(f"Processed {len(processed)} pairs. Outputs saved to {args.out_path}.")
    print(f"Skipped existing {len(skipped_existing)} pairs.")
    print(f"Rejected {len(rejected)} invalid pairs.")


if __name__ == "__main__":
    main()
