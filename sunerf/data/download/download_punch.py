#!/usr/bin/env python3
import argparse
import datetime as dt
import re
import subprocess
from bisect import bisect_left
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
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


def list_day_files(day_url: str, extension: str) -> List[str]:
    with urlopen(day_url) as response:
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


def download_files(urls: List[str], out_dir: Path, overwrite: bool) -> Tuple[int, int]:
    downloaded = 0
    skipped = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_names = {}

    for i, url in enumerate(urls, start=1):
        local_path = out_dir / Path(url).name
        name = local_path.name
        if name in seen_names and seen_names[name] != url:
            print(f"[{i}/{len(urls)}] Warning: duplicate filename {name} from multiple URLs.")
        seen_names[name] = url

        if local_path.exists() and not overwrite:
            skipped += 1
            print(f"[{i}/{len(urls)}] Skipping existing: {local_path}")
            continue

        print(f"[{i}/{len(urls)}] Downloading: {url}")
        subprocess.run(
            ["wget", "-nv", "-O", str(local_path), url],
            check=True,
        )
        downloaded += 1

    return downloaded, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download PUNCH archive files with cadence sampling using wget."
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.start >= args.end:
        raise SystemExit("Error: --start must be earlier than --end.")
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

    if args.level == "l1":
        spacecraft = L1_INSTRUMENT_TO_SPACECRAFT[args.instrument]
        files = list_l1_files(
            base_url=base_url,
            start=args.start,
            end=args.end,
            extension=extension,
            spacecraft=spacecraft,
            include_clear=args.include_clear,
        )
        filtered = filter_l1_files_for_instrument(
            files=files,
            instrument=args.instrument,
            include_clear=args.include_clear,
            pair_tolerance=args.l1_pair_tolerance,
        )
        sampled_urls = sample_l1_groups_by_cadence(
            files=filtered,
            start=args.start,
            end=args.end,
            cadence=args.cadence,
        )
    else:
        files = list_files(
            base_url=base_url,
            start=args.start,
            end=args.end,
            extension=extension,
        )
        sampled_urls = sample_by_cadence(
            files=files,
            start=args.start,
            end=args.end,
            cadence=args.cadence,
        )

    if args.level == "l1":
        print(
            f"Found {len(files)} candidate files in range for L1; "
            f"selected {len(sampled_urls)} URLs after instrument/full-polarization filtering"
            f"{'' if args.cadence is None else ' and cadence'}."
        )
    else:
        print(
            f"Found {len(files)} files in range for {args.level.upper()} {product}; "
            f"selected {len(sampled_urls)}{'' if args.cadence is None else ' after cadence'}."
        )
    if args.dry_run:
        for url in sampled_urls:
            print(url)
        return

    downloaded, skipped = download_files(
        urls=sampled_urls,
        out_dir=Path(args.out),
        overwrite=args.overwrite,
    )
    print(f"Done. Downloaded: {downloaded}, skipped existing: {skipped}.")


if __name__ == "__main__":
    main()
