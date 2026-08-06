#!/usr/bin/env python3
"""
Batch-preprocess SOHO/LASCO polarized pB and %P FITS images.

For each matched pB/%P pair:
1) Loads pB and percent-polarization maps.
2) Computes tB = pB / (%P / 100).
3) Applies the shared coronagraph preprocessing to tB and pB.
4) Saves outputs to:
   - <out_path>/tB/<basename>.fits
   - <out_path>/pB/<basename>.fits
"""

import argparse
import multiprocessing
import os
import re
from glob import glob
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates.ephemeris import get_horizons_coord
from sunpy.map import Map
from sunpy.sun import constants
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    add_cadence_argument,
    add_common_prep_arguments,
    common_kwargs_from_args,
    ensure_tb_pb_output_dirs,
    get_observation_time,
    select_items_by_time,
    should_write_output,
)


POLARIZED_STEM_RE = re.compile(r"^(?P<prefix>.+)-(?P<kind>pb|%p)-(?P<time>\d{8}_\d{4,6})$", re.IGNORECASE)
FITS_SUFFIXES = (".fits.gz", ".fts.gz", ".fits", ".fts")


def normalize_name(path: str) -> str:
    name = os.path.basename(path)
    previous = None
    while name != previous:
        previous = name
        name = unquote(name)
    return name


def fits_stem(path: str) -> str:
    name = normalize_name(path)
    for suffix in FITS_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return os.path.splitext(name)[0]


def normalize_pair_stem(path: str) -> str:
    stem = fits_stem(path)
    match = POLARIZED_STEM_RE.match(stem)
    if match is None:
        return stem
    return f"{match.group('prefix')}-X-{match.group('time')}"


def collect_pairs(pb_path: str, percent_path: str):
    pb_matches = sorted(glob(pb_path))
    percent_matches = sorted(glob(percent_path))

    pb_files = {normalize_pair_stem(path): path for path in pb_matches}
    percent_files = {normalize_pair_stem(path): path for path in percent_matches}
    pair_keys = sorted(set(pb_files) & set(percent_files))
    if not pair_keys:
        pb_only = sorted(set(pb_files) - set(percent_files))
        percent_only = sorted(set(percent_files) - set(pb_files))
        raise RuntimeError(
            f"No matching LASCO pB/%P pairs found.\n"
            f"pB-only keys: {pb_only[:10]}\n"
            f"%P-only keys: {percent_only[:10]}"
        )
    return [(pb_files[key], percent_files[key]) for key in pair_keys]


def add_soho_observer_metadata(pairs):
    patched_pairs = []
    obs_times = [get_observation_time(pb_path) for pb_path, _ in pairs]
    print(f"Querying Horizons for {len(obs_times)} SOHO observer positions.", flush=True)
    soho_positions = get_horizons_coord("SOHO", Time(obs_times))
    print("Finished Horizons SOHO observer query.", flush=True)
    for (pb_path, percent_path), soho in zip(pairs, soho_positions):
        observer = {
            "HGLN_OBS": soho.lon.to_value(u.deg),
            "HGLT_OBS": soho.lat.to_value(u.deg),
            "DSUN_OBS": soho.radius.to_value(u.m),
        }
        patched_pairs.append((pb_path, percent_path, observer))
    return patched_pairs


def add_soho_observer_metadata_to_files(files):
    patched_files = []
    obs_times = [get_observation_time(path) for path in files]
    print(f"Querying Horizons for {len(obs_times)} SOHO observer positions.", flush=True)
    soho_positions = get_horizons_coord("SOHO", Time(obs_times))
    print("Finished Horizons SOHO observer query.", flush=True)
    for path, soho in zip(files, soho_positions):
        observer = {
            "HGLN_OBS": soho.lon.to_value(u.deg),
            "HGLT_OBS": soho.lat.to_value(u.deg),
            "DSUN_OBS": soho.radius.to_value(u.m),
        }
        patched_files.append((path, observer))
    return patched_files


def load_lasco_maps(pb_path: str, percent_path: str, observer: dict):
    try:
        pb_data = np.array(fits.getdata(pb_path), dtype=np.float32, copy=True)
        percent_data = np.array(fits.getdata(percent_path), dtype=np.float32, copy=True)
        header = normalize_lasco_header(fits.getheader(pb_path), observer)
    except Exception as exc:
        raise RuntimeError(f"Error loading LASCO pair {pb_path} / {percent_path}: {exc}")

    tb_data = np.full_like(pb_data, np.nan, dtype=np.float32)
    np.divide(pb_data, percent_data / 100, out=tb_data, where=percent_data != 0)

    tb_header = header.copy()
    tb_header["POLAR"] = "Clear"
    tb_header["HISTORY"] = f"Computed tB = pB / (%P / 100) from {normalize_name(pb_path)} and {normalize_name(percent_path)}"
    return Map(tb_data, tb_header), Map(pb_data, header)


def load_clear_lasco_map(clear_path: str, observer: dict):
    try:
        data = np.array(fits.getdata(clear_path), dtype=np.float32, copy=True)
        header = normalize_lasco_header(fits.getheader(clear_path), observer)
    except Exception as exc:
        raise RuntimeError(f"Error loading LASCO clear file {clear_path}: {exc}")
    return Map(data, header)


def normalize_lasco_header(header, observer):
    header = header.copy()
    header.setdefault("CTYPE1", "HPLN-TAN")
    header.setdefault("CTYPE2", "HPLT-TAN")
    header.setdefault("CUNIT1", "arcsec")
    header.setdefault("CUNIT2", "arcsec")
    header.setdefault("CRVAL1", 0.0)
    header.setdefault("CRVAL2", 0.0)
    header.setdefault("RSUN_REF", constants.radius.to_value(u.m))
    header["HGLN_OBS"] = observer["HGLN_OBS"]
    header["HGLT_OBS"] = observer["HGLT_OBS"]
    header["DSUN_OBS"] = observer["DSUN_OBS"]
    return header


class LascoPrep:
    def __init__(self, out_path, overwrite=False, **preprocess_kwargs):
        self.overwrite = overwrite
        self.map_preprocessor = MapPreprocessor(**preprocess_kwargs)
        self.tb_out_path, self.pb_out_path = ensure_tb_pb_output_dirs(out_path)

    def convert(self, pair):
        pb_path, percent_path, observer = pair
        basename = f"{re.sub('pb', 'tB', fits_stem(pb_path), count=1, flags=re.IGNORECASE)}.fits"
        tb_out = os.path.join(self.tb_out_path, basename)
        pb_out = os.path.join(self.pb_out_path, f"{fits_stem(pb_path)}.fits")

        write_tb = should_write_output(tb_out, self.overwrite)
        write_pb = should_write_output(pb_out, self.overwrite)
        if not write_tb and not write_pb:
            return {"status": "skipped_existing", "tb_out": tb_out, "pb_out": pb_out}

        try:
            tb_map, pb_map = load_lasco_maps(pb_path, percent_path, observer)
            tb_map = self.map_preprocessor.prepare_map(tb_map)
            pb_map = self.map_preprocessor.prepare_map(pb_map)
            if write_tb:
                tb_map.save(tb_out, overwrite=self.overwrite)
            if write_pb:
                pb_map.save(pb_out, overwrite=self.overwrite)
        except Exception as exc:
            print(f"[{os.getpid()}] ERROR in {normalize_name(pb_path)} / {normalize_name(percent_path)}: {exc}",
                  flush=True)
            raise

        return {"status": "processed", "tb_out": tb_out, "pb_out": pb_out}


class LascoClearPrep:
    def __init__(self, out_path, overwrite=False, **preprocess_kwargs):
        self.overwrite = overwrite
        self.tb_out_path = os.path.join(out_path, "tB")
        os.makedirs(self.tb_out_path, exist_ok=True)
        self.map_preprocessor = MapPreprocessor(**preprocess_kwargs)

    def convert(self, item):
        clear_path, observer = item
        tb_out = os.path.join(self.tb_out_path, f"{fits_stem(clear_path)}.fits")
        if not should_write_output(tb_out, self.overwrite):
            return {"status": "skipped_existing", "tb_out": tb_out}

        try:
            tb_map = load_clear_lasco_map(clear_path, observer)
            tb_map = self.map_preprocessor.prepare_map(tb_map)
            tb_map.save(tb_out, overwrite=self.overwrite)
        except Exception as exc:
            print(f"[{os.getpid()}] ERROR in {normalize_name(clear_path)}: {exc}", flush=True)
            raise

        return {"status": "processed", "tb_out": tb_out}


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--pb_path", type=str, default=None, help="Glob pattern for LASCO pB FITS files.")
    parser.add_argument("--percent_path", type=str, default=None, help="Glob pattern for LASCO %P FITS files.")
    parser.add_argument("--clear_path", type=str, default=None, help="Glob pattern for LASCO clear level-1 FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed tB/pB maps.")
    add_common_prep_arguments(parser)
    parser.add_argument("--num_workers", type=int, default=32)
    add_cadence_argument(parser)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    if args.clear_path is not None:
        files = sorted(glob(args.clear_path))
        if not files:
            raise FileNotFoundError(f"No LASCO clear files matched: {args.clear_path}")
        if args.start is not None or args.end is not None or args.cadence is not None:
            original_count = len(files)
            files = select_items_by_time(
                files,
                start=args.start,
                end=args.end,
                cadence=args.cadence,
            )
            print(f"Time selection kept {len(files)} of {original_count} files.")
        files = add_soho_observer_metadata_to_files(files)
        prepper = LascoClearPrep(
            args.out_path,
            overwrite=args.overwrite,
            **common_kwargs_from_args(args),
        )
        with multiprocessing.Pool(args.num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(prepper.convert, files, chunksize=1),
                    total=len(files),
                    desc="Preprocessing LASCO clear files",
                )
            )
        processed = [result for result in results if result["status"] == "processed"]
        skipped_existing = [result for result in results if result["status"] == "skipped_existing"]
        print(f"Processed {len(processed)} clear files. Outputs saved to {args.out_path}.")
        print(f"Skipped existing {len(skipped_existing)} clear files.")
        return

    if args.pb_path is None or args.percent_path is None:
        raise SystemExit("Provide either --clear_path or both --pb_path and --percent_path.")

    pairs = collect_pairs(args.pb_path, args.percent_path)
    if args.start is not None or args.end is not None or args.cadence is not None:
        original_count = len(pairs)
        pairs = select_items_by_time(
            pairs,
            start=args.start,
            end=args.end,
            cadence=args.cadence,
        )
        print(f"Time selection kept {len(pairs)} of {original_count} pairs.")
    pairs = add_soho_observer_metadata(pairs)

    prepper = LascoPrep(
        args.out_path,
        overwrite=args.overwrite,
        **common_kwargs_from_args(args),
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(prepper.convert, pairs, chunksize=1),
                total=len(pairs),
                desc="Preprocessing LASCO pairs",
            )
        )

    processed = [result for result in results if result["status"] == "processed"]
    skipped_existing = [result for result in results if result["status"] == "skipped_existing"]
    print(f"Processed {len(processed)} pairs. Outputs saved to {args.out_path}.")
    print(f"Skipped existing {len(skipped_existing)} pairs.")


if __name__ == "__main__":
    main()
