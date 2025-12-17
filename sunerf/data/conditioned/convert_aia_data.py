import argparse
import glob
import os.path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from sunerf.data.loader.base_loader import MapDataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AIA FITS maps to NPZ files for conditioned SuNeRF."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input glob pattern for FITS files (e.g. '/path/to/aia/*.193.*.fits')",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=16,
        help="Number of parallel processes (default: 16)",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(args.input))
    if len(files) == 0:
        raise RuntimeError(f"No files found for input pattern: {args.input}")

    loader = MapDataLoader(Rs_per_ds=1, reference_frame="helioprojective", add_hpc=True)

    with Pool(processes=args.nproc) as pool:
        for data in tqdm(pool.imap(loader.load, files), total=len(files)):
            out_file = os.path.join(
                args.output,
                f'{data["time"].strftime("%Y%m%d_%H%M%S")}.npz',
            )
            np.savez(out_file, image=data["image"], rays=data["rays"], hpc=data["hpc"])


if __name__ == "__main__":
    main()
