#!/usr/bin/env python3
"""Download the PSI density cubes listed under one directory URL."""

import argparse
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the downloaded HDF4 density cubes.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Directory URL that lists the density cubes.",
    )
    parser.add_argument(
        "--accept",
        default="*.hdf",
        help="File-name pattern of the cubes to download, e.g. rho000050.hdf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to download the PSI HDF4 density cubes.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # --continue resumes partial cubes and leaves complete ones untouched.
    subprocess.run(
        [
            "wget",
            "--no-verbose",
            "--recursive",
            "--level=1",
            "--no-parent",
            "--no-directories",
            "--continue",
            "--accept",
            args.accept,
            "--directory-prefix",
            str(args.out_dir),
            args.url.rstrip("/") + "/",
        ],
        check=True,
    )
    print(f"PSI density cubes: {args.out_dir}")


if __name__ == "__main__":
    main()
