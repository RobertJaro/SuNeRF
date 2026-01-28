#!/usr/bin/env python3
import argparse
from sunpy.net import Fido, attrs as a
import astropy.units as u
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO time, e.g. 2024-10-01 or 2024-10-01T00:00:00")
    p.add_argument("--end", required=True, help="ISO time, e.g. 2024-11-01 or 2024-11-01T00:00:00")
    p.add_argument("--detector", nargs="+", default=["COR1", "COR2"],
                   choices=["COR1", "COR2"], help="SECCHI detectors (default: both)")
    p.add_argument("--out", default="stereo", help="Output directory")
    p.add_argument("--sample-minutes", type=float, default=None, help="Optional cadence sampling")
    args = p.parse_args()

    for det in args.detector:
        q = [
            a.Time(args.start, args.end),
            a.Source("STEREO"),
            a.Instrument("SECCHI"),
            a.Detector(det)
        ]

        res = Fido.search(*q)
        Fido.fetch(res, path=os.path.join(args.out, det))

if __name__ == "__main__":
    main()
