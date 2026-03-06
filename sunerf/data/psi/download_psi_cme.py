import argparse
import shutil
import subprocess
from pathlib import Path

from tqdm.auto import tqdm

DEFAULT_BASE_URL = "https://www.predsci.com/~epalmerio/getpb/20211028/fakeC3/fits_{observer}/{product}/"
DEFAULT_OBSERVERS = ["L1", "L4", "L5"]
DEFAULT_PRODUCTS = ["pb", "tb"]


def download_with_wget(
        url: str, output_dir: Path, dry_run: bool = False, overwrite: bool = False
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "wget",
        "--no-verbose",
        "--recursive",
        "--no-parent",
        "--no-directories",
        "--reject",
        "index.html*",
        "--directory-prefix",
        str(output_dir),
    ]
    if not overwrite:
        cmd.append("--no-clobber")
    cmd.append(url)

    print(" ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PSI CME FITS files with wget.")
    parser.add_argument(
        "--observers",
        nargs="+",
        default=DEFAULT_OBSERVERS,
        help="Observer IDs to download (e.g. L1 L4 L5).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=DEFAULT_PRODUCTS,
        choices=DEFAULT_PRODUCTS,
        help="Products to download.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=(
            "URL template with placeholders {observer} and {product}. "
            "Default: https://www.predsci.com/~epalmerio/getpb/20211028/fakeC3/fits_{observer}/{product}/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print wget commands without downloading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default is to skip existing files).",
    )
    return parser.parse_args()


def main() -> None:
    if shutil.which("wget") is None:
        raise RuntimeError("wget is not installed or not found in PATH.")

    args = parse_args()
    observers = [observer.upper() for observer in args.observers]
    jobs = [(observer, product) for observer in observers for product in args.products]

    for observer, product in tqdm(jobs, desc="Download jobs", unit="job"):
        url = args.base_url.format(observer=observer, product=product)
        target_dir = args.out_dir / observer / product
        download_with_wget(
            url=url,
            output_dir=target_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
