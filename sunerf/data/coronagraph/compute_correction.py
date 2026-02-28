import argparse
import glob

import numpy as np
from astropy.io import fits
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Compute percentile correction mask from FITS stack."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern to FITS files (e.g. '/path/to/*.fits')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=20.0,
        help="Percentile for correction mask (default: 20)",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise ValueError("No files matched --input pattern")

    stack = [fits.getdata(f) for f in tqdm(files, desc="Loading data")]
    stack = np.stack(stack, axis=0)

    mask = np.percentile(stack, args.percentile, axis=0)
    np.save(args.output, mask)


if __name__ == "__main__":
    main()