#!/usr/bin/env python3
"""
Remove files whose filename stem (without extension) appears in a text file
of invalid identifiers. By default this deletes files; use --dry_run to
preview what would be removed.
"""

import argparse, glob
from pathlib import Path

def main():
    """
    Iterate over all files matched by --base_path and remove those whose
    stem matches any entry in --invalid_files.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Delete files whose filename stem matches any line in "
            "the invalid_files text list. Use --dry_run to preview."
        )
    )
    parser.add_argument(
        "--invalid_files",
        required=True,
        help="Text file with one invalid filename stem per line."
    )
    parser.add_argument(
        "--base_path",
        required=True,
        help='Glob pattern to search for files (e.g. "/path/**/*").'
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not delete files; only print what would be removed."
    )
    args = parser.parse_args()

    # Load invalid stems
    with open(args.invalid_files) as f:
        invalid = {
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        }

    for fname in glob.iglob(args.base_path, recursive=True):
        p = Path(fname)
        if not p.is_file():
            continue

        if p.stem in invalid:
            if args.dry_run:
                print(f"WOULD DELETE {p}")
            else:
                p.unlink()
                print(f"DELETED {p}")

if __name__ == "__main__":
    main()
