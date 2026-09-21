"""Shared contracts and utilities for SuNeRF data downloaders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DownloadRequest:
    """Instrument-independent portion of every download request."""

    start: dt.datetime
    end: dt.datetime
    output: Path
    overwrite: bool = False
    dry_run: bool = False

    def __post_init__(self):
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")
        if not str(self.output):
            raise ValueError("output must be a non-empty path")


@dataclass(frozen=True)
class DownloadResult:
    """Common result returned by programmatic downloader entry points."""

    selected: int
    downloaded: int = 0
    skipped: int = 0
    files: tuple[str, ...] = ()


def combine_results(results: Iterable[DownloadResult]) -> DownloadResult:
    """Combine channel or dataset downloads into one uniform result."""
    items = tuple(results)
    return DownloadResult(
        selected=sum(item.selected for item in items),
        downloaded=sum(item.downloaded for item in items),
        skipped=sum(item.skipped for item in items),
        files=tuple(path for item in items for path in item.files),
    )


def parse_iso_datetime(value: str) -> dt.datetime:
    """Parse an ISO timestamp and normalize timezone-aware values to naive UTC."""
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime {value!r}; use ISO format such as 2025-09-01T00:00:00Z."
        ) from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def parse_cadence(value: str) -> dt.timedelta | None:
    """Parse a positive duration such as ``30m`` or ``6h``; ``all`` means none."""
    text = str(value).strip().lower()
    if text in {"all", "none"}:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd])", text)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"Invalid cadence {value!r}; use 30m, 1h, 6h, 1d, or 'all'."
        )
    amount = float(match.group(1))
    if not np.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError("cadence must be positive")
    seconds = amount * {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    return dt.timedelta(seconds=seconds)


def cadence_string(value: dt.timedelta | None) -> str | None:
    """Return a JSOC-compatible duration string."""
    if value is None:
        return None
    seconds = value.total_seconds()
    if seconds.is_integer():
        return f"{int(seconds)}s"
    return f"{seconds:g}s"


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: str | None = None,
) -> argparse.ArgumentParser:
    """Add the identical core CLI shared by every retained downloader."""
    parser.add_argument("--start", required=True, type=parse_iso_datetime)
    parser.add_argument("--end", required=True, type=parse_iso_datetime)
    parser.add_argument(
        "--output",
        required=default_output is None,
        default=default_output,
        type=Path,
        help="Directory receiving downloaded files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and list the selected records without downloading.",
    )
    return parser


def request_from_args(args: argparse.Namespace) -> DownloadRequest:
    try:
        return DownloadRequest(
            start=args.start,
            end=args.end,
            output=Path(args.output),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
    except ValueError as error:
        raise SystemExit(f"Error: {error}.") from error


def ensure_output(request: DownloadRequest) -> None:
    if not request.dry_run:
        request.output.mkdir(parents=True, exist_ok=True)


def result_count(response) -> int:
    """Count records in a SunPy UnifiedResponse or table-like response."""
    if hasattr(response, "colnames"):
        return len(response)
    try:
        return sum(len(block) for block in response)
    except TypeError:
        return len(response)


def fetch_fido(
    *responses,
    request: DownloadRequest,
    description: str,
) -> DownloadResult:
    """Fetch selected Fido records with uniform paths and error handling."""
    selected = sum(result_count(response) for response in responses)
    if request.dry_run:
        for response in responses:
            print(response)
        return DownloadResult(selected=selected)
    ensure_output(request)
    from sunpy.net import Fido

    downloaded = Fido.fetch(
        *responses,
        path=os.path.join(request.output, "{file}"),
        overwrite=request.overwrite,
    )
    errors = getattr(downloaded, "errors", ())
    if errors:
        messages = "; ".join(str(error.exception) for error in errors)
        raise RuntimeError(f"{description} failed: {messages}")
    files = tuple(map(str, downloaded))
    if len(files) != selected:
        raise RuntimeError(
            f"{description} returned {len(files)} files for {selected} selected records"
        )
    return DownloadResult(selected=selected, downloaded=len(files), files=files)


def nearest_indices(reference_times: Sequence, candidate_times: Sequence) -> np.ndarray:
    """Map each reference timestamp to the closest candidate timestamp."""
    reference = np.asarray(reference_times)
    candidates = np.asarray(candidate_times)
    if reference.size == 0:
        return np.empty(0, dtype=np.int64)
    if candidates.size == 0:
        raise ValueError("no candidate observations are available")
    return np.asarray(
        [int(np.argmin(np.abs(candidates - value))) for value in reference],
        dtype=np.int64,
    )


def unique_indices(indices: Iterable[int]) -> np.ndarray:
    """Deduplicate selected records without changing their temporal order."""
    return np.asarray(list(dict.fromkeys(int(index) for index in indices)), dtype=np.int64)


def cadence_indices(times: Sequence, start, end, cadence) -> np.ndarray:
    """Select observations nearest regular cadence slots in ``[start, end)``."""
    values = np.asarray(times)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    if cadence is None:
        return np.arange(values.size, dtype=np.int64)
    slots = []
    value = start
    while value < end:
        slots.append(value)
        value += cadence
    indices = nearest_indices(np.asarray(slots, dtype=object), values)
    return unique_indices(indices)
