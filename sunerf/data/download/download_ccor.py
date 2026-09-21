#!/usr/bin/env python3
import datetime as dt
import re
from pathlib import Path
from shutil import copyfileobj
from typing import Iterator, List, Optional, Tuple

import fsspec

from sunerf.data.download.core import (
    DownloadResult,
    add_common_arguments,
    parse_cadence,
    request_from_args,
)

S3_BUCKET = "noaa-nesdis-swfo-ccor-1-pds"
DEFAULT_PREFIX = "SWFO/GOES-19/CCOR-1/ccor1-l1a_science"


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
    cadence: Optional[dt.timedelta],
) -> List[str]:
    if cadence is None:
        return sorted({path for path, _ in files})

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


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download NOAA SWFO GOES-19 CCOR L1A science files with cadence sampling."
    )
    add_common_arguments(parser, default_output="data/ccor")
    parser.add_argument(
        "--cadence",
        default="1h",
        type=parse_cadence,
        help="Sampling cadence (default: 1h). Use 'none' to download all files.",
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
    return parser


def download(request, *, cadence, product_prefix=DEFAULT_PREFIX, extension=".fits"):

    fs = fsspec.filesystem("s3", anon=True)
    files = list_files(
        fs=fs,
        product_prefix=product_prefix.strip("/"),
        start=request.start,
        end=request.end,
        extension=extension,
    )

    sampled_files = sample_by_cadence(
        files=files,
        start=request.start,
        end=request.end,
        cadence=cadence,
    )

    print(f"Found {len(files)} files in range; selected {len(sampled_files)} after cadence.")
    if request.dry_run:
        for path in sampled_files:
            print(path)
        return DownloadResult(selected=len(sampled_files))

    downloaded, skipped = download_files(
        fs=fs,
        files=sampled_files,
        out_dir=request.output,
        overwrite=request.overwrite,
    )
    print(f"Done. Downloaded: {downloaded}, skipped existing: {skipped}.")
    return DownloadResult(
        selected=len(sampled_files), downloaded=downloaded, skipped=skipped
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        cadence=args.cadence,
        product_prefix=args.product_prefix,
        extension=args.ext,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
