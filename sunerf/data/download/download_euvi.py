"""Download time-matched STEREO/SECCHI EUVI channel sets from VSO.

Records already present are not fetched again: a level-0.5 file in the output
directory, or, with ``--prepared-dir``, the prepared product of its
SECCHI_PREP output (also one in ``rejected/``) whose raw file was deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path, PurePosixPath

import astropy.units as u
import numpy as np
from sunpy.net import Fido, attrs as a

from sunerf.data.euv.clear_sources import has_product
from sunerf.data.download.core import (
    DownloadRequest,
    DownloadResult,
    add_common_arguments,
    combine_results,
    fetch_fido,
    nearest_indices,
    parse_cadence,
    request_from_args,
    search_with_retry,
    split_request,
    unique_indices,
)


DEFAULT_CHANNELS = (171, 195, 284)
DEFAULT_SOURCES = ("STEREO_A", "STEREO_B")
# Largest time difference between a channel record and its anchor-channel record.
DEFAULT_MAX_OFFSET = dt.timedelta(minutes=10)


def _vso_search(*query):
    """Search VSO and reject empty responses so transient failures are retried."""
    response = Fido.search(*query)
    try:
        records = response["vso"]
    except (KeyError, IndexError) as error:
        raise RuntimeError("VSO returned no matching records") from error
    if len(records) == 0:
        raise RuntimeError("VSO returned no matching records")
    return records


def _times(records):
    return np.asarray(records["Start Time"].datetime)


def select_channel_sets(request, *, source, channels, cadence, max_offset=DEFAULT_MAX_OFFSET):
    """Records of every channel nearest to the anchor-channel times of one chunk.

    The anchor channel (284 when requested) sets the observation times. A
    channel with no record within ``max_offset`` of an anchor is left out of
    that set, and a channel without any record in the chunk is skipped with a
    message. A chunk without anchor records raises ``RuntimeError``.
    """
    time = a.Time(request.start, request.end)
    anchor_channel = 284 if 284 in channels else channels[0]
    anchor_query = [
        time,
        a.Instrument.secchi,
        a.Detector.euvi,
        a.Source(source),
        a.Wavelength(anchor_channel * u.AA),
    ]
    if cadence is not None:
        anchor_query.append(a.Sample(cadence.total_seconds() * u.s))
    anchors = search_with_retry(_vso_search, *anchor_query)
    anchor_times = _times(anchors)

    selected = []
    for channel in channels:
        try:
            records = search_with_retry(
                _vso_search,
                time,
                a.Instrument.secchi,
                a.Detector.euvi,
                a.Source(source),
                a.Wavelength(channel * u.AA),
            )
        except RuntimeError as error:
            print(f"{source} EUVI {channel}: no records in {request.start:%Y-%m-%d}.."
                  f"{request.end:%Y-%m-%d}, channel skipped ({error})")
            continue
        times = _times(records)
        indices = nearest_indices(anchor_times, times)
        offsets = np.abs(times[indices] - anchor_times)
        indices = unique_indices(indices[offsets <= max_offset])
        if len(indices):
            selected.append(records[indices])
    return tuple(selected)


def _is_present(fileid, output, prepared_dir) -> bool:
    """True when the raw file of a VSO ``fileid`` (its last path element, e.g.
    ``secchi/L0/a/img/euvi/20110206/20110206_001630_n4euA.fts``) or its prepared
    product exists."""
    name = PurePosixPath(str(fileid)).name
    if (Path(output) / name).exists():
        return True
    if prepared_dir is None:
        return False
    try:
        return has_product(name, prepared_dir, secchi_level_0=True)
    except ValueError:  # not a level-0.5 name: only the raw file can match
        return False


def missing_records(selected, request: DownloadRequest, prepared_dir=None):
    """Drop the records already present locally; returns (records, skipped)."""
    if request.overwrite:
        return selected, 0
    remaining, skipped = [], 0
    for records in selected:
        keep = np.array(
            [not _is_present(fileid, request.output, prepared_dir) for fileid in records["fileid"]],
            dtype=bool,
        )
        skipped += int((~keep).sum())
        if keep.any():
            remaining.append(records[keep])
    return tuple(remaining), skipped


def download(
    request: DownloadRequest,
    *,
    channels=DEFAULT_CHANNELS,
    sources=DEFAULT_SOURCES,
    cadence=None,
    chunk=dt.timedelta(days=7),
    max_offset=DEFAULT_MAX_OFFSET,
    prepared_dir=None,
):
    """Download channel sets; the range is queried in ``chunk`` pieces because the
    non-anchor channels are searched unsampled (nearest-time matching needs every
    record) and VSO responses grow with the range.

    A chunk without anchor records (a data gap, or an archive failure that
    outlasted the retries) is skipped and listed at the end; a rerun fills it,
    since finished files (raw, or prepared in ``prepared_dir``) are not
    downloaded again."""
    if not channels:
        raise ValueError("channels must not be empty")
    results = []
    gaps = []
    for part in split_request(request, chunk):
        for source in sources:
            try:
                selected = select_channel_sets(
                    part, source=source, channels=tuple(channels), cadence=cadence,
                    max_offset=max_offset,
                )
            except RuntimeError as error:
                gaps.append(f"{source} {part.start:%Y-%m-%d}..{part.end:%Y-%m-%d}: {error}")
                continue
            if not request.dry_run:
                selected, skipped = missing_records(selected, part, prepared_dir)
                if skipped:
                    print(f"{source} {part.start:%Y-%m-%d}..{part.end:%Y-%m-%d}: "
                          f"skipping {skipped} records already present")
                    results.append(DownloadResult(selected=skipped, skipped=skipped))
            if not selected:
                continue
            results.append(fetch_fido(
                *selected,
                request=part,
                description=f"EUVI download for {source} {part.start:%Y-%m-%d}..{part.end:%Y-%m-%d}",
            ))
    if gaps:
        print(f"{len(gaps)} chunks without EUVI records (data gap or archive failure):")
        for gap in gaps:
            print(f"  {gap}")
    return combine_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--chunk",
        type=parse_cadence,
        default=parse_cadence("7d"),
        help="Length of each archive query (e.g. 7d); 'all' queries the whole range at once.",
    )
    parser.add_argument(
        "--max-offset",
        type=parse_cadence,
        default=DEFAULT_MAX_OFFSET,
        help="Largest time difference between a channel and its anchor-channel record (e.g. 10m).",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=None,
        help="Directory of the prepared products; records with a product are not downloaded again.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        channels=args.channels,
        sources=args.sources,
        cadence=args.cadence,
        chunk=args.chunk,
        max_offset=args.max_offset,
        prepared_dir=args.prepared_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
