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

from sunerf.data.coronagraph.prep_coronagraph import _get_observation_time, _prep_coronagraph_map, parse_duration


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


def sample_pairs_at_cadence(pairs, cadence):
    if cadence.total_seconds() <= 0:
        raise ValueError("Cadence must be positive.")

    timed_pairs = []
    for tb_path, pb_path in tqdm(pairs, desc="Loading observation times"):
        tb_time = _get_observation_time(tb_path)
        pb_time = _get_observation_time(pb_path)
        obs_time = min(tb_time, pb_time)
        timed_pairs.append((obs_time, (tb_path, pb_path)))
    timed_pairs.sort(key=lambda item: item[0])

    sampled = []
    next_time = timed_pairs[0][0]
    for obs_time, pair in timed_pairs:
        if obs_time < next_time:
            continue
        sampled.append(pair)
        next_time = obs_time + cadence

    return sampled


def header_is_normal(header) -> bool:
    return str(header.get("SEB_PROG", "")).strip().upper() == "NORMAL"


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
    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, resize=None, clip_max=None,
                 nan_threshold=0.5):
        self.out_path = out_path
        self.overwrite = overwrite
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.resize = resize
        self.clip_max = clip_max
        self.nan_threshold = nan_threshold

        self.tb_out_path = os.path.join(out_path, "tB")
        self.pb_out_path = os.path.join(out_path, "pB")
        os.makedirs(self.tb_out_path, exist_ok=True)
        os.makedirs(self.pb_out_path, exist_ok=True)

    def _prepare_map(self, s_map):
        s_map = _prep_coronagraph_map(s_map, occ_min=self.occ_min, occ_max=self.occ_max)
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.clip_max is not None:
            s_map.data[:] = np.clip(s_map.data, a_max=self.clip_max, a_min=None)
        return s_map

    def _validate_pair(self, tb_path: str, pb_path: str):
        tb_data, tb_header, tb_map = load_stereo_map(tb_path)
        pb_data, pb_header, pb_map = load_stereo_map(pb_path)

        reasons = []
        if not header_is_normal(tb_header):
            reasons.append("tB SEB_PROG!=NORMAL")
        if not header_is_normal(pb_header):
            reasons.append("pB SEB_PROG!=NORMAL")

        tb_invalid = np.array(tb_data, copy=True)
        pb_invalid = np.array(pb_data, copy=True)
        tb_invalid[tb_invalid <= 0] = np.nan
        pb_invalid[pb_invalid <= 0] = np.nan

        tb_nan_fraction = float(np.isnan(tb_invalid).mean()) if tb_invalid.size else 1.0
        pb_nan_fraction = float(np.isnan(pb_invalid).mean()) if pb_invalid.size else 1.0

        if tb_nan_fraction > self.nan_threshold:
            reasons.append(f"tB NaN={tb_nan_fraction:.3f}")
        if pb_nan_fraction > self.nan_threshold:
            reasons.append(f"pB NaN={pb_nan_fraction:.3f}")

        return reasons, tb_map, pb_map

    def convert(self, pair):
        tb_path, pb_path = pair
        tb_out = os.path.join(self.tb_out_path, os.path.basename(tb_path))
        pb_out = os.path.join(self.pb_out_path, os.path.basename(pb_path))

        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return {"status": "skipped_existing", "tb_out": tb_out, "pb_out": pb_out}

        reasons, tb_map, pb_map = self._validate_pair(tb_path, pb_path)
        if reasons:
            return {
                "status": "rejected",
                "tb_path": tb_path,
                "pb_path": pb_path,
                "reason": ", ".join(reasons),
            }

        try:
            tb_map = self._prepare_map(tb_map)
            pb_map = self._prepare_map(pb_map)
            tb_map.save(tb_out, overwrite=True)
            pb_map.save(pb_out, overwrite=True)
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
        "--no_overwrite",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=None,
        help="Optional maximum value to clip data to.",
    )
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
    parser.add_argument(
        "--cadence",
        type=parse_duration,
        default=None,
        help="Optional fixed sampling cadence like 15m, 1h, or 30s.",
    )
    parser.add_argument(
        "--nan_threshold",
        type=float,
        default=0.5,
        help="Reject a pair if either map exceeds this NaN fraction after values <= 0 are treated as invalid.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    pairs = collect_pairs(args.tb_path, args.pb_path)
    if args.cadence is not None:
        original_count = len(pairs)
        pairs = sample_pairs_at_cadence(pairs, args.cadence)
        print(
            f"Cadence sampling kept {len(pairs)} of {original_count} pairs "
            f"at {args.cadence} spacing."
        )

    prepper = StereoCorPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        occ_min=args.occ_min * u.arcsec if args.occ_min is not None else None,
        occ_max=args.occ_max * u.arcsec if args.occ_max is not None else None,
        resize=args.resize,
        clip_max=args.clip_max,
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
