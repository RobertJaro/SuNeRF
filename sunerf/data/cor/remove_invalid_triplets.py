#!/usr/bin/env python3
"""
filter_polar_triplets.py

Remove COR2 polarization files that do not have a close-in-time matching triplet.

Logic
-----
For each POLAR=0 file (P000), find the closest POLAR=120 (P120) and POLAR=240 (P240)
by |DATE-OBS difference|. If either match is farther than `--max-dt` seconds, delete
the P000 file and also delete the "orphan" counterpart if it was within threshold
(so you don't keep incomplete pairs).

Default threshold: 300 s (5 minutes).

Example
-------
python filter_polar_triplets.py \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2/*.fts" \
  --max-dt 300
"""

import argparse
import glob
import os

from astropy.io.fits import getheader
from dateutil.parser import parse


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--glob", required=True, help="Glob pattern for input FITS/FTS files.")
    p.add_argument("--max-dt", type=float, default=300.0, help="Max allowed time difference in seconds.")
    p.add_argument("--dry-run", action="store_true", help="Print what would be deleted, but do not delete.")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    def rm(path):
        if args.dry_run:
            print("DRY-RUN delete:", path)
        else:
            os.remove(path)

    p000, p120, p240 = [], [], []
    for f in files:
        h = getheader(f)
        pol = h.get("POLAR")
        if pol not in (0, 120, 240):
            print(f'Removing non-polarization file: {f}')
            rm(f)
            continue
        d = {"file": f, "date": parse(h["DATE-OBS"])}
        (p000 if pol == 0 else p120 if pol == 120 else p240).append(d)

    if not p120 or not p240:
        raise RuntimeError("Need at least one P120 and one P240 file to match against P000 files.")



    for p0 in p000:
        c120 = min(p120, key=lambda x: abs(x["date"] - p0["date"]))
        c240 = min(p240, key=lambda x: abs(x["date"] - p0["date"]))

        dt120 = abs((c120["date"] - p0["date"]).total_seconds())
        dt240 = abs((c240["date"] - p0["date"]).total_seconds())

        ok120 = dt120 <= args.max_dt
        ok240 = dt240 <= args.max_dt

        if ok120 and ok240:
            continue

        if not ok120 and not ok240:
            print("Warning: no close P120/P240 match for:", p0["file"])
            rm(p0["file"])
        elif not ok120 and ok240:
            print("Warning: no close P120 match for:", p0["file"])
            rm(p0["file"])
            rm(c240["file"])
        elif not ok240 and ok120:
            print("Warning: no close P240 match for:", p0["file"])
            rm(p0["file"])
            rm(c120["file"])


if __name__ == "__main__":
    main()
