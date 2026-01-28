#!/usr/bin/env python3
"""
Patch SOHO/LASCO FITS with correct observer metadata (SOHO position via HORIZONS),
for all files matched by a glob pattern, in parallel.

- Files that raise ANY error are SKIPPED (nothing written to out-dir)
- Successfully patched maps are saved with a suffix (default: _patched)
"""

import argparse
from pathlib import Path
import glob
import os
from multiprocessing import Pool

import astropy.units as u
import sunpy.map
from sunpy.coordinates.ephemeris import get_horizons_coord


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pattern", help="Glob pattern (quote it), e.g. '/path/*C2*.fits'")
    p.add_argument("--out-dir", default=None,
                   help="Output folder (default: alongside input files)")
    p.add_argument("--suffix", default="_patched",
                   help="Output filename suffix before extension")
    p.add_argument("-j", "--jobs", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1),
                   help="Number of worker processes")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite if output file exists")
    return p.parse_args()


def patch_lasco_observer(m: sunpy.map.Map) -> sunpy.map.Map:
    t = m.date  # uses DATE-OBS / DATE_OBS
    soho = get_horizons_coord("SOHO", t)  # Heliographic Stonyhurst

    meta = dict(m.meta)
    meta["hgln_obs"] = soho.lon.to_value(u.deg)
    meta["hglt_obs"] = soho.lat.to_value(u.deg)
    meta["dsun_obs"] = soho.radius.to_value(u.m)

    return sunpy.map.Map(m.data, meta)


def _work(item):
    """
    Return:
      ("OK", in_path, out_path)   -> saved successfully
      ("SKIP", in_path, reason)   -> nothing written
    """
    fp, out_dir, suffix, overwrite = item
    in_path = Path(fp)
    out_base = Path(out_dir) if out_dir else in_path.parent
    out_path = out_base / f"{in_path.stem}{suffix}{in_path.suffix}"

    if out_path.exists() and not overwrite:
        return ("SKIP", fp, f"exists (use --overwrite to replace)")

    try:
        m = sunpy.map.Map(str(in_path))
        m2 = patch_lasco_observer(m)

        # sanity check that observer metadata is valid
        _ = m2.observer_coordinate

        out_base.mkdir(parents=True, exist_ok=True)
        m2.save(out_path, overwrite=True)

        return ("OK", fp, str(out_path))

    except Exception as e:
        # NOTHING is written if we get here
        raise Exception(f'Invalid file {fp}')
        # return ("SKIP", fp, f"{type(e).__name__}: {e}")



def main():
    a = parse_args()
    files = sorted(glob.glob(a.pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {a.pattern}")

    items = [(fp, a.out_dir, a.suffix, a.overwrite) for fp in files]

    for item in items:
        status, fp, msg = _work(item)
        if status == "OK":
            print(f"[OK]   {fp} -> {msg}")
        else:
            print(f"[SKIP] {fp} -> {msg}")


if __name__ == "__main__":
    main()
