#!/usr/bin/env python3
import argparse
import datetime as dt
import re
from pathlib import Path
from shutil import copyfileobj
from typing import Iterator, List, Optional, Tuple

import fsspec

S3_BUCKET = "noaa-nesdis-swfo-ccor-1-pds"
DEFAULT_PREFIX = "SWFO/GOES-19/CCOR-1/ccor1-l1a_science"


def parse_iso_datetime(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use ISO format, e.g. 2025-09-01T00:00:00"
        ) from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_cadence(value: str) -> dt.timedelta:
    match = re.fullmatch(r"(?i)\s*(\d+)\s*([smhd])\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid cadence '{value}'. Use formats like 30m, 1h, 6h, 1d."
        )

    qty = int(match.group(1))
    unit = match.group(2).lower()
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return dt.timedelta(seconds=qty * seconds_per_unit)


def parse_timestamp_from_path(path: str) -> Optional[dt.datetime]:
    filename = Path(path).name

    # GOES convention: ..._sYYYYDDDHHMMSS_...
    match = re.search(r"_s(\d{4})(\d{3})(\d{6})_", filename)
    if match:
        year = int(match.group(1))
        day_of_year = int(match.group(2))
        hhmmss = match.group(3)
        hour, minute, second = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:])
        return dt.datetime(year, 1, 1, hour, minute, second) + dt.timedelta(
            days=day_of_year - 1
        )

    # Generic convention: ...YYYYMMDD_HHMMSS... (or YYYYMMDDTHHMMSS)
    match = re.search(r"(20\d{2})(\d{2})(\d{2})[T_]?(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute, second)

    return None


def iter_days(start: dt.datetime, end: dt.datetime) -> Iterator[dt.date]:
    day = start.date()
    last = end.date()
    while day <= last:
        yield day
        day += dt.timedelta(days=1)


def list_files(
    fs,
    product_prefix: str,
    start: dt.datetime,
    end: dt.datetime,
    extension: str,
) -> List[Tuple[str, Optional[dt.datetime]]]:
    found: List[str] = []
    for day in iter_days(start, end):
        pattern = f"{S3_BUCKET}/{product_prefix}/{day:%Y/%m/%d}/*{extension}"
        found.extend(fs.glob(pattern))

    files = sorted(set(found))
    timed = [(path, parse_timestamp_from_path(path)) for path in files]

    # Keep files that are either clearly in range or unparseable (to avoid dropping data).
    return [
        (path, ts)
        for path, ts in timed
        if ts is None or (start <= ts < end)
    ]


def sample_by_cadence(
    files: List[Tuple[str, Optional[dt.datetime]]],
    start: dt.datetime,
    end: dt.datetime,
    cadence: dt.timedelta,
) -> List[str]:
    parseable = sorted((p, ts) for p, ts in files if ts is not None)
    unparseable = [p for p, ts in files if ts is None]

    if not parseable:
        return sorted(unparseable)

    selected: List[str] = []
    idx = 0
    slot_start = start

    while slot_start < end:
        slot_end = min(slot_start + cadence, end)

        while idx < len(parseable) and parseable[idx][1] < slot_start:
            idx += 1

        if idx < len(parseable) and parseable[idx][1] < slot_end:
            selected.append(parseable[idx][0])
            while idx < len(parseable) and parseable[idx][1] < slot_end:
                idx += 1

        slot_start = slot_end

    return sorted(set(selected + unparseable))


def s3_to_local_path(s3_path: str, out_dir: Path) -> Path:
    return out_dir / Path(s3_path).name


def download_files(fs, files: List[str], out_dir: Path, overwrite: bool) -> Tuple[int, int]:
    downloaded = 0
    skipped = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_names = {}

    for i, s3_path in enumerate(files, start=1):
        local_path = s3_to_local_path(s3_path, out_dir)
        name = local_path.name

        if name in seen_names and seen_names[name] != s3_path:
            print(
                f"[{i}/{len(files)}] Warning: duplicate filename {name} from multiple S3 keys."
            )
        seen_names[name] = s3_path

        if local_path.exists() and not overwrite:
            skipped += 1
            print(f"[{i}/{len(files)}] Skipping existing: {local_path}")
            continue

        print(f"[{i}/{len(files)}] Downloading: {s3_path}")
        with fs.open(s3_path, "rb") as src, local_path.open("wb") as dst:
            copyfileobj(src, dst)
        downloaded += 1

    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download NOAA SWFO GOES-19 CCOR L1A science files with cadence sampling."
    )
    parser.add_argument("--start", required=True, type=parse_iso_datetime)
    parser.add_argument("--end", required=True, type=parse_iso_datetime)
    parser.add_argument(
        "--cadence",
        default="1h",
        type=parse_cadence,
        help="Sampling cadence (default: 1h). Examples: 30m, 1h, 6h.",
    )
    parser.add_argument(
        "--out",
        default="data/ccor",
        help="Local output directory (default: data/ccor).",
    )
    parser.add_argument(
        "--product-prefix",
        default=DEFAULT_PREFIX,
        help=f"S3 prefix under {S3_BUCKET} (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--ext",
        default=".fits",
        help="File extension filter (default: .fits).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite local files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files without downloading.",
    )
    args = parser.parse_args()

    if args.start >= args.end:
        raise SystemExit("Error: --start must be earlier than --end.")

    fs = fsspec.filesystem("s3", anon=True)
    files = list_files(
        fs=fs,
        product_prefix=args.product_prefix.strip("/"),
        start=args.start,
        end=args.end,
        extension=args.ext,
    )

    sampled_files = sample_by_cadence(
        files=files,
        start=args.start,
        end=args.end,
        cadence=args.cadence,
    )

    print(f"Found {len(files)} files in range; selected {len(sampled_files)} after cadence.")
    if args.dry_run:
        for path in sampled_files:
            print(path)
        return

    out_dir = Path(args.out)
    downloaded, skipped = download_files(
        fs=fs,
        files=sampled_files,
        out_dir=out_dir,
        overwrite=args.overwrite,
    )
    print(f"Done. Downloaded: {downloaded}, skipped existing: {skipped}.")


if __name__ == "__main__":
    main()
