#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin
from urllib.request import urlopen


ARCHIVES = {
    "level_1": {
        "url": "https://lasco-www.nrl.navy.mil/lz/level_1/",
        "subdir": "level_1",
    },
    "polarized": {
        "url": "https://lasco-www.nrl.navy.mil/lz/polarize/2010_03/vig/c2/",
        "subdir": "polarized",
    },
}
FITS_SUFFIXES = (".fts", ".fts.gz", ".fits", ".fits.gz")


@dataclass(frozen=True)
class ArchiveFile:
    url: str
    timestamp: Optional[dt.datetime]


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def parse_iso_datetime(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use ISO format, e.g. 2010-03-19T00:00:00"
        ) from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_timestamp_from_name(filename: str) -> Optional[dt.datetime]:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})[_T-]?(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute, second)

    match = re.search(r"(20\d{2})(\d{2})(\d{2})[_T-]?(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute)

    return None


def parse_listing_timestamp(href: str, html: str) -> Optional[dt.datetime]:
    escaped_href = re.escape(href)
    patterns = [
        rf'href=["\']{escaped_href}["\'][^>]*>.*?</a>\s+(\d{{4}}-\d{{2}}-\d{{2}})\s+(\d{{2}}:\d{{2}})',
        rf'href=["\']{escaped_href}["\'][^>]*>.*?</a>\s+(\d{{2}}-[A-Za-z]{{3}}-\d{{4}})\s+(\d{{2}}:\d{{2}})',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if not match:
            continue
        date_text, time_text = match.groups()
        for fmt in ("%Y-%m-%d %H:%M", "%d-%b-%Y %H:%M"):
            try:
                return dt.datetime.strptime(f"{date_text} {time_text}", fmt)
            except ValueError:
                pass
    return None


def parse_img_hdr(url: str, instrument: str) -> List[ArchiveFile]:
    with urlopen(urljoin(url, "img_hdr.txt")) as response:
        lines = response.read().decode("utf-8", errors="replace").splitlines()

    files = []
    for line in lines:
        columns = line.split()
        if len(columns) < 12:
            continue
        filename, date_text, time_text, detector = columns[:4]
        polar = columns[10]
        if detector.upper() != instrument.upper() or polar.lower() != "clear":
            continue
        timestamp = dt.datetime.strptime(f"{date_text} {time_text}", "%Y/%m/%d %H:%M:%S")
        files.append(ArchiveFile(urljoin(url, filename), timestamp))
    return files


def iter_days(start: dt.datetime, end: dt.datetime):
    day = start.date()
    last_day = (end - dt.timedelta(microseconds=1)).date()
    while day <= last_day:
        yield day
        day += dt.timedelta(days=1)


def list_archive_files(url: str) -> List[ArchiveFile]:
    with urlopen(url) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = LinkParser()
    parser.feed(html)

    files: List[ArchiveFile] = []
    for href in parser.links:
        name = Path(href).name
        if name.lower().endswith(FITS_SUFFIXES):
            timestamp = parse_timestamp_from_name(name) or parse_listing_timestamp(href, html)
            files.append(ArchiveFile(urljoin(url, href), timestamp))

    return sorted(set(files), key=lambda file: file.url)


def list_level_1_day_files(url: str, instrument: str) -> List[ArchiveFile]:
    files_by_name = {Path(file.url).name.removesuffix(".gz"): file.url for file in list_archive_files(url)}
    files = []
    for header_file in parse_img_hdr(url, instrument):
        name = Path(header_file.url).name
        file_url = files_by_name.get(name)
        if file_url is not None:
            files.append(ArchiveFile(file_url, header_file.timestamp))
    return sorted(files, key=lambda file: file.url)


def list_level_1_files(url: str, instrument: str, start: dt.datetime, end: dt.datetime) -> List[ArchiveFile]:
    files: List[ArchiveFile] = []
    for day in iter_days(start, end):
        day_url = urljoin(url, f"{day:%y%m%d}/{instrument.lower()}/")
        try:
            files.extend(list_level_1_day_files(day_url, instrument))
        except Exception as exc:
            print(f"Warning: could not list {day_url} ({exc})", flush=True)
    return sorted(files, key=lambda file: file.url)


def filter_by_time(files: List[ArchiveFile], start: dt.datetime, end: dt.datetime) -> List[str]:
    selected = []
    for file in files:
        if file.timestamp is None:
            print(f"Warning: could not parse timestamp, skipping {Path(file.url).name}", flush=True)
            continue
        if start <= file.timestamp < end:
            selected.append(file.url)
    return selected


def download_file(job):
    index, total, url, out_dir, overwrite = job
    local_path = Path(out_dir) / Path(url).name
    if local_path.exists() and not overwrite:
        return index, total, url, "skipped"

    tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
    try:
        subprocess.run(["wget", "-q", "-O", str(tmp_path), url], check=True)
        tmp_path.replace(local_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        return index, total, url, "failed"
    return index, total, url, "downloaded"


def download_files(urls: List[str], out_dir: Path, overwrite: bool, workers: int) -> Counter:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(i, len(urls), url, str(out_dir), overwrite) for i, url in enumerate(urls, start=1)]
    counts: Counter = Counter()

    print(f"Downloading {len(urls)} files to {out_dir} with {workers} workers.", flush=True)
    with Pool(processes=workers) as pool:
        for _, total, url, status in pool.imap_unordered(download_file, jobs):
            counts[status] += 1
            print(f"[{sum(counts.values())}/{total}] {status}: {Path(url).name}", flush=True)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Download prepped SOHO/LASCO C2 data from the NRL LASCO archive."
    )
    parser.add_argument("--start", required=True, type=parse_iso_datetime)
    parser.add_argument("--end", required=True, type=parse_iso_datetime)
    parser.add_argument(
        "--instrument",
        "--detector",
        dest="instrument",
        nargs="+",
        default=["C2"],
        choices=["C2"],
        help="LASCO instrument/detector to download (default: C2).",
    )
    parser.add_argument(
        "--product",
        nargs="+",
        choices=["level_1", "polarized"],
        default=["level_1", "polarized"],
        help="Product(s) to download (default: level_1 polarized).",
    )
    parser.add_argument(
        "--out",
        default="lasco",
        help="Output root. Files are written under <out>/<instrument>/<archive-product>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start >= args.end:
        raise SystemExit("Error: --start must be earlier than --end.")
    if args.workers < 1:
        raise SystemExit("Error: --workers must be at least 1.")

    totals: Counter = Counter()

    for instrument in args.instrument:
        for product in args.product:
            archive = ARCHIVES[product]
            files = (
                list_level_1_files(archive["url"], instrument, args.start, args.end)
                if product == "level_1"
                else list_archive_files(archive["url"])
            )
            selected_urls = filter_by_time(files, args.start, args.end)
            print(
                f"Found {len(files)} {instrument} {product} files; "
                f"selected {len(selected_urls)} from {args.start.isoformat()} to {args.end.isoformat()}.",
                flush=True,
            )

            if args.dry_run:
                for url in selected_urls:
                    print(url)
                continue

            out_dir = Path(args.out) / instrument.lower() / archive["subdir"]
            totals.update(download_files(
                urls=selected_urls,
                out_dir=out_dir,
                overwrite=args.overwrite,
                workers=args.workers,
            ))

    print(
        f"Done. Downloaded: {totals['downloaded']}, "
        f"skipped existing: {totals['skipped']}, failed: {totals['failed']}."
    )
    if totals["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
