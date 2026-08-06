#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
import time
import uuid
from bisect import bisect_left
from html.parser import HTMLParser
from multiprocessing import Pool
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urljoin
from urllib.request import urlopen

DEFAULT_ARCHIVE_ROOT = "https://umbra.nascom.nasa.gov/punch"
DEFAULT_L1_BASE_URL = "https://umbra.nascom.nasa.gov/punch/1"
DEFAULT_SUFFIX = "v0k.fits"
LEVEL_PRODUCTS = {
    "l2": ("PTM",),
    "l3": ("PAM", "CAM"),
}
DEFAULT_PRODUCT_BY_LEVEL = {
    "l2": "PTM",
    "l3": "PAM",
}
URLRETRIEVE_TRIES = 3
CURL_FALLBACK_TRIES = 2
RETRY_DELAY_SECONDS = 10
LIST_TIMEOUT_SECONDS = 30
DOWNLOAD_INACTIVITY_TIMEOUT_SECONDS = 60
MAX_DOWNLOAD_SECONDS = 1800
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
L1_INSTRUMENT_TO_SPACECRAFT = {
    "wfi1": 1,
    "wfi2": 2,
    "wfi3": 3,
    "nfi": 4,
}
L1_POLARIZATION_TYPES = ("PM", "PZ", "PP")


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
            f"Invalid datetime '{value}'. Use ISO format, e.g. 2025-09-01T00:00:00"
        ) from exc

    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_cadence(value: str) -> Optional[dt.timedelta]:
    if value.strip().lower() in {"none", "all"}:
        return None

    match = re.fullmatch(r"(?i)\s*(\d+)\s*([smhd])\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid cadence '{value}'. Use formats like 30m, 1h, 6h, 1d, or 'none'."
        )

    qty = int(match.group(1))
    unit = match.group(2).lower()
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return dt.timedelta(seconds=qty * seconds_per_unit)


def parse_timestamp_from_name(filename: str) -> Optional[dt.datetime]:
    # Common pattern: YYYYMMDD_HHMMSS or YYYYMMDDTHHMMSS
    match = re.search(r"(20\d{2})(\d{2})(\d{2})[T_]?(\d{2})(\d{2})(\d{2})", filename)
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return dt.datetime(year, month, day, hour, minute, second)

    # Fallback: YYYYDOYHHMMSS
    match = re.search(r"(20\d{2})(\d{3})(\d{6})", filename)
    if match:
        year = int(match.group(1))
        day_of_year = int(match.group(2))
        hhmmss = match.group(3)
        hour, minute, second = int(hhmmss[:2]), int(hhmmss[2:4]), int(hhmmss[4:])
        return dt.datetime(year, 1, 1, hour, minute, second) + dt.timedelta(
            days=day_of_year - 1
        )

    return None


def iter_days(start: dt.datetime, end: dt.datetime) -> Iterator[dt.date]:
    day = start.date()
    last = end.date()
    while day <= last:
        yield day
        day += dt.timedelta(days=1)


def iter_day_windows(
    start: dt.datetime, end: dt.datetime
) -> Iterator[Tuple[dt.datetime, dt.datetime]]:
    day = start.date()
    while day <= end.date():
        day_start = dt.datetime.combine(day, dt.time.min)
        day_end = day_start + dt.timedelta(days=1)
        window_start = max(start, day_start)
        window_end = min(end, day_end)
        if window_start < window_end:
            yield window_start, window_end
        day += dt.timedelta(days=1)


def list_day_files(day_url: str, extension: str) -> List[str]:
    with urlopen(day_url, timeout=LIST_TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = LinkParser()
    parser.feed(html)

    files: List[str] = []
    for href in parser.links:
        if href.startswith("?") or href.startswith("#") or href.endswith("/"):
            continue
        name = Path(href).name
        if name.endswith(extension):
            files.append(urljoin(day_url, href))
    return sorted(set(files))


def list_files(
    base_url: str,
    start: dt.datetime,
    end: dt.datetime,
    extension: str,
) -> List[Tuple[str, Optional[dt.datetime]]]:
    found: List[str] = []
    for day in iter_days(start, end):
        day_url = f"{base_url.rstrip('/')}/{day:%Y/%m/%d}/"
        try:
            found.extend(list_day_files(day_url=day_url, extension=extension))
        except Exception as exc:
            print(f"Warning: could not list {day_url} ({exc})")

    timed = [(url, parse_timestamp_from_name(Path(url).name)) for url in sorted(set(found))]
    return [(url, ts) for url, ts in timed if ts is None or (start <= ts < end)]


def list_l1_files(
    base_url: str,
    start: dt.datetime,
    end: dt.datetime,
    extension: str,
    spacecraft: int,
    include_clear: bool = False,
) -> List[Tuple[str, Optional[dt.datetime]]]:
    codes = [f"{kind}{spacecraft}" for kind in L1_POLARIZATION_TYPES]
    if include_clear:
        codes.append(f"CR{spacecraft}")

    found: List[str] = []
    for day in iter_days(start, end):
        for code in codes:
            day_url = f"{base_url.rstrip('/')}/{code}/{day:%Y/%m/%d}/"
            try:
                found.extend(list_day_files(day_url=day_url, extension=extension))
            except Exception as exc:
                print(f"Warning: could not list {day_url} ({exc})")

    timed = [(url, parse_timestamp_from_name(Path(url).name)) for url in sorted(set(found))]
    return [(url, ts) for url, ts in timed if ts is None or (start <= ts < end)]


def sample_by_cadence(
    files: List[Tuple[str, Optional[dt.datetime]]],
    start: dt.datetime,
    end: dt.datetime,
    cadence: Optional[dt.timedelta],
) -> List[str]:
    if cadence is None:
        return sorted({url for url, _ in files})

    parseable = sorted((u, ts) for u, ts in files if ts is not None)
    unparseable = [u for u, ts in files if ts is None]

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


def parse_l1_code(filename: str) -> Optional[Tuple[str, int]]:
    match = re.search(r"(PM|PZ|PP|CR)([1-4])", filename)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def filter_l1_files_for_instrument(
    files: List[Tuple[str, Optional[dt.datetime]]],
    instrument: str,
    include_clear: bool = False,
    pair_tolerance: dt.timedelta = dt.timedelta(minutes=3),
) -> List[Tuple[str, dt.datetime]]:
    spacecraft = L1_INSTRUMENT_TO_SPACECRAFT[instrument]
    by_type = {code: [] for code in ("PM", "PZ", "PP", "CR")}
    for url, ts in files:
        if ts is None:
            continue
        parsed = parse_l1_code(Path(url).name)
        if parsed is None:
            continue
        image_type, sc = parsed
        if sc != spacecraft:
            continue
        by_type[image_type].append((ts, url))

    for image_type in by_type:
        by_type[image_type].sort(key=lambda x: x[0])
    by_type_times = {image_type: [ts for ts, _ in by_type[image_type]] for image_type in by_type}

    tolerance_seconds = pair_tolerance.total_seconds()

    def nearest_pair(
        anchor_ts: dt.datetime, image_type: str
    ) -> Optional[Tuple[dt.datetime, str]]:
        candidates = by_type[image_type]
        if not candidates:
            return None
        times = by_type_times[image_type]
        idx = bisect_left(times, anchor_ts)
        options = []
        if idx < len(candidates):
            options.append(candidates[idx])
        if idx > 0:
            options.append(candidates[idx - 1])
        if not options:
            return None
        best = min(options, key=lambda item: abs((item[0] - anchor_ts).total_seconds()))
        if abs((best[0] - anchor_ts).total_seconds()) > tolerance_seconds:
            return None
        return best

    selected: List[Tuple[str, dt.datetime]] = []
    for pm_ts, pm_url in by_type["PM"]:
        pz = nearest_pair(pm_ts, "PZ")
        pp = nearest_pair(pm_ts, "PP")
        if pz is None or pp is None:
            continue
        cr = nearest_pair(pm_ts, "CR") if include_clear else None
        if include_clear and cr is None:
            continue

        selected.append((pm_url, pm_ts))
        selected.append((pz[1], pm_ts))
        selected.append((pp[1], pm_ts))
        if include_clear and cr is not None:
            selected.append((cr[1], pm_ts))

    return selected


def sample_l1_groups_by_cadence(
    files: List[Tuple[str, dt.datetime]],
    start: dt.datetime,
    end: dt.datetime,
    cadence: Optional[dt.timedelta],
) -> List[str]:
    if not files:
        return []
    if cadence is None:
        return sorted({url for url, _ in files})

    grouped = {}
    for url, ts in files:
        grouped.setdefault(ts, []).append(url)

    timestamps = sorted(grouped)
    selected: List[str] = []
    idx = 0
    slot_start = start
    while slot_start < end:
        slot_end = min(slot_start + cadence, end)
        while idx < len(timestamps) and timestamps[idx] < slot_start:
            idx += 1
        if idx < len(timestamps) and timestamps[idx] < slot_end:
            selected.extend(sorted(grouped[timestamps[idx]]))
            while idx < len(timestamps) and timestamps[idx] < slot_end:
                idx += 1
        slot_start = slot_end

    return sorted(set(selected))


def select_urls(
    level: str,
    instrument: Optional[str],
    base_url: str,
    start: dt.datetime,
    end: dt.datetime,
    extension: str,
    include_clear: bool,
    l1_pair_tolerance: dt.timedelta,
    cadence: Optional[dt.timedelta],
) -> Tuple[int, List[str]]:
    if level == "l1":
        spacecraft = L1_INSTRUMENT_TO_SPACECRAFT[instrument]
        files = list_l1_files(
            base_url=base_url,
            start=start,
            end=end,
            extension=extension,
            spacecraft=spacecraft,
            include_clear=include_clear,
        )
        filtered = filter_l1_files_for_instrument(
            files=files,
            instrument=instrument,
            include_clear=include_clear,
            pair_tolerance=l1_pair_tolerance,
        )
        sampled_urls = sample_l1_groups_by_cadence(
            files=filtered,
            start=start,
            end=end,
            cadence=cadence,
        )
    else:
        files = list_files(
            base_url=base_url,
            start=start,
            end=end,
            extension=extension,
        )
        sampled_urls = sample_by_cadence(
            files=files,
            start=start,
            end=end,
            cadence=cadence,
        )

    return len(files), sampled_urls


def download_one_file(
    url: str,
    local_path: Path,
    overwrite: bool,
    inactivity_timeout: int = DOWNLOAD_INACTIVITY_TIMEOUT_SECONDS,
    max_download_seconds: int = MAX_DOWNLOAD_SECONDS,
) -> Tuple[str, str]:
    if local_path.exists() and not overwrite:
        return "skipped", ""

    tmp_path = local_path.with_name(f".{local_path.name}.{uuid.uuid4().hex}.tmp")
    errors = []

    for attempt in range(1, URLRETRIEVE_TRIES + 1):
        try:
            started_at = time.monotonic()
            with urlopen(url, timeout=inactivity_timeout) as response:
                expected_size = response.headers.get("Content-Length")
                with tmp_path.open("wb") as destination:
                    _copy_response_with_deadline(
                        response=response,
                        destination=destination,
                        started_at=started_at,
                        max_download_seconds=max_download_seconds,
                    )
            if expected_size is not None and tmp_path.stat().st_size != int(expected_size):
                raise OSError(
                    f"incomplete download: expected {expected_size} bytes, "
                    f"received {tmp_path.stat().st_size}"
                )
            tmp_path.replace(local_path)
            return "downloaded", ""
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            errors.append(f"urllib attempt {attempt}: {exc}")
            if attempt < URLRETRIEVE_TRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    for attempt in range(1, CURL_FALLBACK_TRIES + 1):
        try:
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    str(inactivity_timeout),
                    "--speed-limit",
                    "1024",
                    "--speed-time",
                    str(inactivity_timeout),
                    "--max-time",
                    str(max_download_seconds),
                    "--output",
                    str(tmp_path),
                    url,
                ],
                check=True,
                timeout=max_download_seconds + inactivity_timeout,
            )
            tmp_path.replace(local_path)
            return "downloaded", ""
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            errors.append(f"curl fallback attempt {attempt}: {exc}")
            if attempt < CURL_FALLBACK_TRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    return "failed", "; ".join(errors)


def _copy_response_with_deadline(
    response: BinaryIO,
    destination: BinaryIO,
    started_at: float,
    max_download_seconds: int,
) -> None:
    while True:
        if time.monotonic() - started_at > max_download_seconds:
            raise TimeoutError(
                f"download exceeded {max_download_seconds} seconds"
            )
        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
        if not chunk:
            return
        destination.write(chunk)


def download_job(
    job: Tuple[int, int, str, str, bool, int, int]
) -> Tuple[int, int, str, str, str]:
    (
        i,
        total,
        url,
        local_path,
        overwrite,
        inactivity_timeout,
        max_download_seconds,
    ) = job
    print(f"[{i}/{total}] Starting: {url}", flush=True)
    status, error = download_one_file(
        url,
        Path(local_path),
        overwrite,
        inactivity_timeout=inactivity_timeout,
        max_download_seconds=max_download_seconds,
    )
    return i, total, url, status, error


def download_files(
    urls: List[str],
    out_dir: Path,
    overwrite: bool,
    workers: int,
    inactivity_timeout: int = DOWNLOAD_INACTIVITY_TIMEOUT_SECONDS,
    max_download_seconds: int = MAX_DOWNLOAD_SECONDS,
) -> Tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for tmp_path in out_dir.glob(".*.tmp"):
        tmp_path.unlink()
    seen_names: Dict[str, str] = {}
    jobs = []

    for i, url in enumerate(urls, start=1):
        local_path = out_dir / Path(url).name
        name = local_path.name
        if name in seen_names and seen_names[name] != url:
            print(f"[{i}/{len(urls)}] Warning: duplicate filename {name} from multiple URLs.")
        seen_names[name] = url

        jobs.append(
            (
                i,
                len(urls),
                url,
                str(local_path),
                overwrite,
                inactivity_timeout,
                max_download_seconds,
            )
        )

    downloaded = 0
    skipped = 0
    failed = 0
    print(f"Downloading with {workers} workers.", flush=True)
    with Pool(processes=workers) as pool:
        for i, total, url, status, error in pool.imap_unordered(download_job, jobs):
            if status == "skipped":
                skipped += 1
                print(
                    f"[{downloaded + skipped + failed}/{total}] Skipping existing: {out_dir / Path(url).name}",
                    flush=True,
                )
            elif status == "failed":
                failed += 1
                print(
                    f"[{downloaded + skipped + failed}/{total}] Failed: {url} ({error})",
                    flush=True,
                )
            else:
                downloaded += 1
                print(
                    f"[{downloaded + skipped + failed}/{total}] Downloaded: {url}",
                    flush=True,
                )

    return downloaded, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Download PUNCH archive files with cadence sampling."
    )
    parser.add_argument("--start", required=True, type=parse_iso_datetime)
    parser.add_argument("--end", required=True, type=parse_iso_datetime)
    parser.add_argument(
        "--level",
        choices=("l1", "l2", "l3"),
        default="l2",
        help="Data level to download (default: l2).",
    )
    parser.add_argument(
        "--product",
        choices=tuple(sorted({product for products in LEVEL_PRODUCTS.values() for product in products})),
        help="For L2/L3, product selector. Defaults by level (L2: PTM, L3: PAM).",
    )
    parser.add_argument(
        "--instrument",
        choices=tuple(L1_INSTRUMENT_TO_SPACECRAFT.keys()),
        help="L1 instrument selector. Required for --level l1.",
    )
    parser.add_argument(
        "--cadence",
        default=None,
        type=parse_cadence,
        help="Optional sampling cadence. Default: none (download all files). Examples: 30m, 1h, 6h, none.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Archive base URL. Defaults by level/product (L1 root, L2 PTM, L3 PAM/CAM).",
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="Filename suffix filter. Defaults by level.",
    )
    parser.add_argument(
        "--include-clear",
        action="store_true",
        help="For L1, also require/download clear-image CRn files with PMn/PZn/PPn.",
    )
    parser.add_argument(
        "--l1-pair-tolerance",
        default="3m",
        type=parse_cadence,
        help="For L1, max PM/PZ/PP timestamp separation when forming polarization sets (default: 3m).",
    )
    parser.add_argument(
        "--out",
        default="data/punch",
        help="Local output directory (default: data/punch).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--workers",
        default=10,
        type=int,
        help="Number of parallel downloads (default: 10).",
    )
    parser.add_argument(
        "--download-mode",
        choices=("day-by-day", "full"),
        default="day-by-day",
        help="Download one day at a time or all selected files at once (default: day-by-day).",
    )
    parser.add_argument(
        "--inactivity-timeout",
        default=DOWNLOAD_INACTIVITY_TIMEOUT_SECONDS,
        type=int,
        help="Abort a transfer after this many seconds without data (default: 60).",
    )
    parser.add_argument(
        "--max-download-seconds",
        default=MAX_DOWNLOAD_SECONDS,
        type=int,
        help="Maximum time allowed for one file attempt (default: 1800).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start >= args.end:
        raise SystemExit("Error: --start must be earlier than --end.")
    if args.workers < 1:
        raise SystemExit("Error: --workers must be at least 1.")
    if args.inactivity_timeout < 1:
        raise SystemExit("Error: --inactivity-timeout must be at least 1.")
    if args.max_download_seconds < args.inactivity_timeout:
        raise SystemExit(
            "Error: --max-download-seconds must be at least --inactivity-timeout."
        )
    if args.l1_pair_tolerance is None:
        raise SystemExit("Error: --l1-pair-tolerance must be a duration like 3m, not 'none'.")
    if args.level == "l1" and args.instrument is None:
        raise SystemExit("Error: --instrument is required for --level l1.")
    if args.level == "l1" and args.product is not None:
        raise SystemExit("Error: --product is only valid for --level l2 or --level l3.")
    if args.level in LEVEL_PRODUCTS:
        if args.product is None:
            product = DEFAULT_PRODUCT_BY_LEVEL[args.level]
        else:
            product = args.product.upper()
            if product not in LEVEL_PRODUCTS[args.level]:
                valid_products = ", ".join(LEVEL_PRODUCTS[args.level])
                raise SystemExit(
                    f"Error: --product {product} is not valid for --level {args.level}. "
                    f"Valid values: {valid_products}."
                )
    else:
        product = None

    base_url = (
        args.base_url
        if args.base_url is not None
        else (
            DEFAULT_L1_BASE_URL
            if args.level == "l1"
            else f"{DEFAULT_ARCHIVE_ROOT}/{args.level[1:]}/{product}"
        )
    )
    extension = (
        args.ext
        if args.ext is not None
        else DEFAULT_SUFFIX
    )

    downloaded = 0
    skipped = 0
    failed = 0

    if args.download_mode == "day-by-day":
        day_selections = []
        total_files = 0
        total_selected = 0
        for day_start, day_end in iter_day_windows(args.start, args.end):
            file_count, sampled_urls = select_urls(
                level=args.level,
                instrument=args.instrument,
                base_url=base_url,
                start=day_start,
                end=day_end,
                extension=extension,
                include_clear=args.include_clear,
                l1_pair_tolerance=args.l1_pair_tolerance,
                cadence=args.cadence,
            )
            total_files += file_count
            total_selected += len(sampled_urls)
            if sampled_urls:
                day_selections.append((day_start, day_end, sampled_urls))

        print(
            f"Found {total_files} files across {len(day_selections)} available days; "
            f"selected {total_selected}{'' if args.cadence is None else ' after cadence'}.",
            flush=True,
        )
        print("Available days:", flush=True)
        for day_start, day_end, day_urls in day_selections:
            print(
                f"  {day_start.date().isoformat()}: {len(day_urls)} files "
                f"({day_start.isoformat()} to {day_end.isoformat()})",
                flush=True,
            )

        if args.dry_run:
            for _, _, sampled_urls in day_selections:
                for url in sampled_urls:
                    print(url)
            return

        for day_start, day_end, sampled_urls in day_selections:
            print(
                f"Download day {day_start.date().isoformat()}: {len(sampled_urls)} files.",
                flush=True,
            )
            block_downloaded, block_skipped, block_failed = download_files(
                urls=sampled_urls,
                out_dir=Path(args.out),
                overwrite=args.overwrite,
                workers=args.workers,
                inactivity_timeout=args.inactivity_timeout,
                max_download_seconds=args.max_download_seconds,
            )
            downloaded += block_downloaded
            skipped += block_skipped
            failed += block_failed
    else:
        file_count, sampled_urls = select_urls(
            level=args.level,
            instrument=args.instrument,
            base_url=base_url,
            start=args.start,
            end=args.end,
            extension=extension,
            include_clear=args.include_clear,
            l1_pair_tolerance=args.l1_pair_tolerance,
            cadence=args.cadence,
        )
        if args.level == "l1":
            print(
                f"Found {file_count} candidate files in range for L1; "
                f"selected {len(sampled_urls)} URLs after instrument/full-polarization filtering"
                f"{'' if args.cadence is None else ' and cadence'}.",
                flush=True,
            )
        else:
            print(
                f"Found {file_count} files in range for {args.level.upper()} {product}; "
                f"selected {len(sampled_urls)}{'' if args.cadence is None else ' after cadence'}.",
                flush=True,
            )
        if args.dry_run:
            for url in sampled_urls:
                print(url)
            return

        downloaded, skipped, failed = download_files(
            urls=sampled_urls,
            out_dir=Path(args.out),
            overwrite=args.overwrite,
            workers=args.workers,
            inactivity_timeout=args.inactivity_timeout,
            max_download_seconds=args.max_download_seconds,
        )

    print(f"Done. Downloaded: {downloaded}, skipped existing: {skipped}, failed: {failed}.")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
