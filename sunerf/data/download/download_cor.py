#!/usr/bin/env python3
import argparse
from sunpy.net import Fido, attrs as a
import astropy.units as u
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO time, e.g. 2024-10-01 or 2024-10-01T00:00:00")
    p.add_argument("--end", required=True, help="ISO time, e.g. 2024-11-01 or 2024-11-01T00:00:00")
    p.add_argument("--detector", default="COR2", help="Detector to use, e.g. COR1, COR2")
    p.add_argument("--source", default="STEREO_A", help="Source to use, e.g. STEREO_A, STEREO_B")
    p.add_argument("--out", default="stereo", help="Output directory")
    p.add_argument("--sample-minutes", type=float, default=None, help="Optional cadence sampling")
    args = p.parse_args()

    q = [
        a.Time(args.start, args.end),
        a.Source(args.source),
        a.Instrument("SECCHI"),
        a.Detector(args.detector)
    ]

    res = Fido.search(*q)
    Fido.fetch(res, path=args.out, retry=True)

if __name__ == "__main__":
    main()
