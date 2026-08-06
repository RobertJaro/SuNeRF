#!/usr/bin/env python3
import argparse
import os

from sunpy.net import Fido
from sunpy.net import attrs as a


DEFAULT_DATASETS = [
    "SOLO_COHO1HR_MERGED_MAG_PLASMA",
]


def main():
    parser = argparse.ArgumentParser(description="Download Solar Orbiter in-situ CDF files from CDAWeb via SunPy/Fido.")
    parser.add_argument("--start", required=True, help="Start time, e.g. 2025-09-01T00:00:00")
    parser.add_argument("--end", required=True, help="End time, e.g. 2025-10-01T00:00:00")
    parser.add_argument("--out", required=True, help="Output directory for raw Solar Orbiter CDF files")
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="CDAWeb dataset id to download. Can be repeated. Defaults to SOLO_COHO1HR_MERGED_MAG_PLASMA.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    datasets = args.datasets if args.datasets is not None else DEFAULT_DATASETS
    trange = a.Time(args.start, args.end)

    for dataset_id in datasets:
        print(f"Searching CDAWeb dataset {dataset_id} for {args.start} -> {args.end}")
        result = Fido.search(trange, a.cdaweb.Dataset(dataset_id))
        print(result)
        if len(result) == 0:
            print(f"No files found for {dataset_id}")
            continue
        print(f"Fetching {dataset_id}")
        Fido.fetch(result, path=os.path.join(args.out, "{file}"))


if __name__ == "__main__":
    main()
