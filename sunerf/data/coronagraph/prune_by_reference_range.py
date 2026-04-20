#!/usr/bin/env python3
"""
Remove files outside the datetime range covered by a reference file set.

The script derives the inclusive [min, max] timestamp range from filenames in
the reference glob, then deletes files in one or more target globs whose
timestamps fall outside that range.
"""

import argparse
import glob
import re
from datetime import datetime
from pathlib import Path


TIMESTAMP_PATTERNS = (
    re.compile(r"(?P<ts>\d{8}T\d{6})"),
    re.compile(r"(?P<ts>\d{8}_\d{6})"),
    re.compile(r"(?P<ts>\d{14})"),
)


def extract_timestamp(path: Path) -> datetime | None:
    name = path.name
    for pattern in TIMESTAMP_PATTERNS:
        match = pattern.search(name)
        if match is None:
            continue
        stamp = match.group("ts")
        if "T" in stamp:
            return datetime.strptime(stamp, "%Y%m%dT%H%M%S")
        if "_" in stamp:
            return datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        return datetime.strptime(stamp, "%Y%m%d%H%M%S")
    return None


def iter_files(pattern: str):
    paths = [Path(path) for path in sorted(glob.glob(pattern))]
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    for path in paths:
        if path.is_file():
            yield path


def collect_range(reference_pattern: str) -> tuple[datetime, datetime]:
    timestamps = []
    skipped = 0
    for path in iter_files(reference_pattern):
        timestamp = extract_timestamp(path)
        if timestamp is None:
            skipped += 1
            continue
        timestamps.append(timestamp)

    if not timestamps:
        raise RuntimeError(f"No timestamps found in reference files: {reference_pattern}")

    if skipped:
        print(f"Skipped {skipped} reference files without a recognized timestamp pattern.")

    return min(timestamps), max(timestamps)


def prune_directory(pattern: str, start: datetime, end: datetime, dry_run: bool) -> tuple[int, int]:
    removed = 0
    skipped = 0

    for path in iter_files(pattern):
        timestamp = extract_timestamp(path)
        if timestamp is None:
            skipped += 1
            print(f"SKIP  {path} (no recognized timestamp)")
            continue

        if start <= timestamp <= end:
            continue

        removed += 1
        if dry_run:
            print(f"WOULD REMOVE {path}")
        else:
            path.unlink()
            print(f"REMOVED {path}")

    return removed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Remove files in target directories that fall outside the datetime range of a reference directory."
    )
    parser.add_argument(
        "--reference-path",
        default="/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/*",
        help="Glob used to derive the valid datetime range.",
    )
    parser.add_argument(
        "--target-path",
        action="append",
        default=None,
        help="Glob to prune. May be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be removed without deleting them.",
    )
    args = parser.parse_args()

    start, end = collect_range(args.reference_path)
    print(f"Reference range: {start.isoformat()} to {end.isoformat()}")

    target_paths = args.target_path or [
        "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*",
        "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*",
        "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*",
    ]

    for target in target_paths:
        removed, skipped = prune_directory(target, start, end, args.dry_run)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {removed} files from {target}. Skipped {skipped} without timestamps.")


if __name__ == "__main__":
    main()
