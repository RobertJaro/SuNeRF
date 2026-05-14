#!/usr/bin/env python3
"""
Build tB/pB maps from PUNCH L1 polarization triplets (PM/PZ/PP).

The script:
1) Scans FITS files matched by --data_path.
2) Detects L1 polarization files via filename token P[MZP][1-4].
3) Builds nearest-time triplets (PM anchor + nearest PZ/PP) within tolerance.
4) Combines each triplet into Stokes-like components:
   - I = 2/3 * (PM + PZ + PP)
   - Q = 2/3 * (2*PZ - PM - PP)
   - U = 2/sqrt(3) * (PP - PM)
   - tB = I
   - pB = sqrt(Q^2 + U^2)
5) Optionally reprojects each input map to a TAN WCS before combining.
6) Applies shared map preprocessing.
7) Saves outputs to:
   - <out_path>/tB/<basename>.fits
   - <out_path>/pB/<basename>.fits
"""

import argparse
import datetime as dt
import multiprocessing
import os
import re
from bisect import bisect_left
from glob import glob

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from sunpy.map import Map
from sunpy.map.header_helper import make_fitswcs_header
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import (
    MapPreprocessor,
    add_common_prep_arguments,
    common_kwargs_from_args,
    ensure_tb_pb_output_dirs,
    parse_duration,
)

FILENAME_RE = re.compile(r"P(?P<pol>[MZP])(?P<sc>[1-4])_(?P<ts>\d{14})")


def parse_l1_polarization_file(file_path: str):
    name = os.path.basename(file_path)
    match = FILENAME_RE.search(name)
    if not match:
        return None
    pol = match.group("pol")
    sc = int(match.group("sc"))
    ts = dt.datetime.strptime(match.group("ts"), "%Y%m%d%H%M%S")
    return {"path": file_path, "pol": pol, "sc": sc, "ts": ts}


def build_triplets(entries, tolerance: dt.timedelta):
    entries_by_sc = {}
    for entry in entries:
        entries_by_sc.setdefault(entry["sc"], []).append(entry)

    tolerance_seconds = tolerance.total_seconds()
    triplets = []

    for sc in sorted(entries_by_sc):
        per_pol = {"M": [], "Z": [], "P": []}
        for entry in entries_by_sc[sc]:
            per_pol[entry["pol"]].append(entry)
        for pol in per_pol:
            per_pol[pol].sort(key=lambda x: x["ts"])

        times_by_pol = {pol: [e["ts"] for e in per_pol[pol]] for pol in per_pol}

        def nearest(anchor_ts: dt.datetime, pol: str):
            candidates = per_pol[pol]
            if not candidates:
                return None
            times = times_by_pol[pol]
            idx = bisect_left(times, anchor_ts)
            options = []
            if idx < len(candidates):
                options.append(candidates[idx])
            if idx > 0:
                options.append(candidates[idx - 1])
            if not options:
                return None
            best = min(options, key=lambda item: abs((item["ts"] - anchor_ts).total_seconds()))
            if abs((best["ts"] - anchor_ts).total_seconds()) > tolerance_seconds:
                return None
            return best

        for pm in per_pol["M"]:
            pz = nearest(pm["ts"], "Z")
            pp = nearest(pm["ts"], "P")
            if pz is None or pp is None:
                continue
            triplets.append((pm, pz, pp))

    return triplets


class PunchTripletPrep:
    def __init__(
        self,
        out_path,
        overwrite=True,
        occ_min=None,
        occ_max=None,
        max_radius=None,
        resize=None,
        clip_min=None,
        clip_max=None,
        value_min=None,
        value_max=None,
        filter_bright_objects=False,
        bright_object_threshold=10.0,
        reproject=False,
    ):
        self.out_path = out_path
        self.overwrite = overwrite
        self.reproject = reproject
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
    def _to_output_name(pm_basename: str):
        tb_name = re.sub(r"_PM([1-4])_", r"_TB\1_", pm_basename)
        pb_name = re.sub(r"_PM([1-4])_", r"_PB\1_", pm_basename)
        if tb_name == pm_basename:
            root, ext = os.path.splitext(pm_basename)
            tb_name = f"{root}_tB{ext}"
            pb_name = f"{root}_pB{ext}"
        return tb_name, pb_name

    def _load_punch_map(self, path):
        with fits.open(path) as hdul:
            data = hdul[1].data
            wcs_in = WCS(hdul[1].header, fobj=hdul)
            smap = Map(data, wcs_in)

        if not self.reproject:
            return smap

        target_header = make_fitswcs_header(
            smap.data.shape,
            smap.reference_coordinate,
            scale=u.Quantity(smap.scale),
            projection_code="TAN",
            instrument=smap.instrument,
            observatory=smap.observatory,
            wavelength=getattr(smap, "wavelength", None),
        )
        target_wcs = WCS(target_header)
        return smap.reproject_to(target_wcs)

    def convert(self, triplet):
        pm, pz, pp = triplet

        pm_basename = os.path.basename(pm["path"])
        tb_name, pb_name = self._to_output_name(pm_basename)
        tb_out = os.path.join(self.tb_out_path, tb_name)
        pb_out = os.path.join(self.pb_out_path, pb_name)

        if os.path.exists(tb_out) and os.path.exists(pb_out) and not self.overwrite:
            return tb_out, pb_out

        try:
            pm_map = self._load_punch_map(pm["path"])
            pz_map = self._load_punch_map(pz["path"])
            pp_map = self._load_punch_map(pp["path"])

            pm_data = np.asarray(pm_map.data, dtype=float)
            pz_data = np.asarray(pz_map.data, dtype=float)
            pp_data = np.asarray(pp_map.data, dtype=float)
            header = pm_map.meta

            pm_data[pm_data <= 0] = 0
            pz_data[pz_data <= 0] = 0
            pp_data[pp_data <= 0] = 0

            i_stokes = (2.0 / 3.0) * (pm_data + pz_data + pp_data)
            q_stokes = (2.0 / 3.0) * ((2.0 * pz_data) - pm_data - pp_data)
            u_stokes = (2.0 / np.sqrt(3.0)) * (pp_data - pm_data)

            tb = i_stokes
            pb = np.sqrt(q_stokes ** 2 + u_stokes ** 2)

            tb_map = self.map_preprocessor.prepare_map(Map(tb, header))
            pb_map = self.map_preprocessor.prepare_map(Map(pb, header))

            tb_map.save(tb_out, overwrite=True)
            pb_map.save(pb_out, overwrite=True)
        except Exception as exc:
            print(f"[{os.getpid()}] ERROR processing {pm_basename}: {exc}", flush=True)
            raise
        return tb_out, pb_out


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_path", type=str, required=True, help="Glob pattern for L1 PM/PZ/PP FITS files.")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed tB/pB maps.")
    parser.add_argument(
        "--pair_tolerance",
        type=parse_duration,
        default=parse_duration("3m"),
        help="Maximum timestamp separation when building PM/PZ/PP triplets.",
    )
    add_common_prep_arguments(parser, include_clip=True)
    parser.add_argument(
        "--reproject",
        action="store_true",
        help="Reproject each input map to a TAN WCS before combining. Disabled by default.",
    )
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

    parsed = [parse_l1_polarization_file(f) for f in files]
    entries = [item for item in parsed if item is not None]
    if not entries:
        raise RuntimeError("No L1 PM/PZ/PP files detected in --data_path matches.")

    triplets = build_triplets(entries, tolerance=args.pair_tolerance)
    if not triplets:
        raise RuntimeError("No valid PM/PZ/PP triplets found. Try increasing --pair_tolerance.")

    prepper = PunchTripletPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        **common_kwargs_from_args(args),
        reproject=args.reproject,
    )

    with multiprocessing.Pool(args.num_workers) as pool:
        out_files = [
            pair
            for pair in tqdm(
                pool.imap_unordered(prepper.convert, triplets, chunksize=1),
                total=len(triplets),
                desc="Combining PUNCH triplets",
            )
        ]

    print(
        f"Processed {len(out_files)} triplets. Saved tB to {os.path.join(args.out_path, 'tB')} and pB to "
        f"{os.path.join(args.out_path, 'pB')}."
    )


if __name__ == "__main__":
    main()
