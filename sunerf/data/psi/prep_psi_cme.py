#!/usr/bin/env python3

import argparse
import glob
import os
import multiprocessing
from pathlib import Path

from astropy.io import fits
import numpy as np
import sunpy.map
import astropy.units as u
from tqdm import tqdm


def load_fixed_map(path):
    """Load FITS, sanitize header (tabs / non-ASCII), return SunPy Map."""

    # --- sanitize header ---
    with open(path, "rb") as f:
        blocks = []
        while True:
            b = f.read(2880)
            if not b:
                break
            blocks.append(b)
            if b"END" in b:
                break

    s = b"".join(blocks).decode("latin-1").replace("\t", " ").replace("\n", " ").replace("\r", " ")
    cards = [s[i:i+80] for i in range(0, len(s), 80)]
    cards = ["".join(c if 32 <= ord(c) <= 126 else " " for c in card) for card in cards]

    end = next(i for i, c in enumerate(cards) if c.startswith("END"))
    hdr = fits.Header.fromstring("".join(cards[:end+1]), sep="")
    header = dict(hdr)

    # --- fix header keywords ---
    header['DATE-OBS'] = header['DATE_OBS'].strip()

    # --- load data ---
    data = np.squeeze(fits.getdata(path)).astype(float)

    # mask invalid pixels
    data[data <= 0] = np.nan

    return sunpy.map.Map(data, header)


def process_file(in_file, out_file, resolution, overwrite):
    m = load_fixed_map(in_file)
    if resolution is not None:
        nx, ny = resolution
        m = m.resample((ny, nx) * u.pixel)
    m.save(out_file, overwrite=overwrite)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        required=True,
        help="Input FITS path or glob pattern (e.g. /data/**/*)",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files (default behavior is to skip existing files).",
    )
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=None,
        metavar=("NX", "NY"),
        help="Resize to target resolution",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for parallel processing.",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    has_glob = glob.has_magic(args.data_path)

    if has_glob:
        matches = [Path(p) for p in glob.glob(args.data_path, recursive=True)]
        input_files = sorted([p for p in matches if p.is_file()])
        if not input_files:
            raise FileNotFoundError(f"No files matched pattern: {args.data_path}")

        # Keep folder structure relative to the non-glob prefix.
        prefix_parts = []
        for part in data_path.parts:
            if glob.has_magic(part):
                break
            prefix_parts.append(part)
        input_root = Path(*prefix_parts) if prefix_parts else Path(".")
    else:
        if not data_path.is_file():
            raise FileNotFoundError(f"Input file not found: {data_path}")
        input_files = [data_path]
        input_root = data_path.parent

    tasks = []
    for in_file in input_files:
        relative = in_file.relative_to(input_root)
        out_file = output_path / relative if has_glob else output_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.exists() and not args.overwrite:
            continue
        tasks.append((str(in_file), str(out_file)))

    if not tasks:
        return

    workers = max(1, args.workers)
    pool_args = [(in_file, out_file, args.resolution, args.overwrite) for in_file, out_file in tasks]
    with multiprocessing.Pool(processes=workers) as pool:
        for _ in tqdm(pool.starmap(process_file, pool_args), total=len(pool_args), desc="Processing FITS", unit="file"):
            pass


if __name__ == "__main__":
    main()
