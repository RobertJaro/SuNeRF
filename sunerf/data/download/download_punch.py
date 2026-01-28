#!/usr/bin/env python3
import argparse
from sunpy.net import Fido, attrs as a
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="ISO time, e.g. 2025-06-01 or 2025-06-01T00:00:00")
    p.add_argument("--end", required=True, help="ISO time, e.g. 2025-06-02 or 2025-06-02T00:00:00")
    p.add_argument("--detector", nargs="+", default=["NFI"],
                   choices=["NFI"], help="PUNCH detectors (currently only NFI here)")
    p.add_argument("--out", default="punch", help="Output directory")
    p.add_argument("--sample-minutes", type=float, default=None, help="Optional cadence sampling (not implemented)")
    args = p.parse_args()

    for det in args.detector:
        q = [
            a.Time(args.start, args.end),
            a.Source("PUNCH"),        # may or may not be required depending on VSO backend
            # a.Instrument("NFI"),      # key selector
        ]

        res = Fido.search(*q)
        print(res)

        # if len(res) == 0:
        #     print(f"No results for {det}")
        #     continue
        #
        # outdir = os.path.join(args.out, det)
        # os.makedirs(outdir, exist_ok=True)
        #
        # Fido.fetch(res, path=outdir)


if __name__ == "__main__":
    main()
