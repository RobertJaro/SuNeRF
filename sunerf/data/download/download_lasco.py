#!/usr/bin/env python3
import argparse

from astropy.io.fits import getheader
from sunpy.net import Fido, attrs as a
import astropy.units as u
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO time, e.g. 2022-01-01 or 2022-01-01T00:00:00")
    p.add_argument("--end", required=True, help="ISO time, e.g. 2022-01-02 or 2022-01-02T00:00:00")
    p.add_argument("--detector", nargs="+", default=["C2", "C3"], choices=["C2", "C3"],
                   help="LASCO detector(s) to download")
    p.add_argument("--out", default="lasco", help="Output directory")
    p.add_argument("--sample-minutes", type=float, default=None, help="Optional cadence sampling")
    args = p.parse_args()

    for det in args.detector:
        q = [
            a.Time(args.start, args.end),
            a.Source("SOHO"),
            a.Instrument("LASCO"),
            a.Detector(det),
        ]
        if args.sample_minutes is not None:
            q.append(a.Sample(args.sample_minutes * u.min))

        res = Fido.search(*q)
        files = Fido.fetch(res, path=os.path.join(args.out, det))
        # remove polarized images - only tB
        for f in files:
            if getheader(f)['POLAR'] != 'Clear':
                print(f"Removing polarized image {f}")
                os.remove(f)

if __name__ == "__main__":
    main()
