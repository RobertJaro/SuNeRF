#!/usr/bin/env python3
"""Download selected PSI observer and brightness-product sequences."""

import argparse
import shutil
from pathlib import Path

from sunerf.data.psi.download_psi_cme import download_with_wget
PRODUCTS = ("pb", "tb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for the downloaded source FITS files.",
    )
    parser.add_argument(
        "--observers",
        nargs="+",
        required=True,
        help="PSI observer identifiers to download.",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCTS,
        default=list(PRODUCTS),
        help="Brightness products to download.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Download URL template containing {observer} and {product}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloads instead of skipping them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if shutil.which("wget") is None:
        raise RuntimeError("wget is required to download the PSI FITS sequence.")

    observers = tuple(observer.upper() for observer in args.observers)
    for observer in observers:
        for product in args.products:
            download_with_wget(
                url=args.base_url.format(observer=observer, product=product),
                output_dir=args.out_dir / observer / product,
                overwrite=args.overwrite,
            )

    print(f"Downloaded PSI data: {args.out_dir}")


if __name__ == "__main__":
    main()
