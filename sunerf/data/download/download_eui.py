"""Download time-matched Solar Orbiter/EUI FSI level-2 channel sets."""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    cadence_indices,
    fetch_fido,
    nearest_indices,
    parse_cadence,
    parse_iso_datetime,
    request_from_args,
)


DEFAULT_CHANNELS = (174, 304)


def _soar_records(response):
    try:
        records = response["soar"]
    except (KeyError, IndexError) as error:
        raise RuntimeError("SOAR returned no matching EUI records") from error
    if len(records) == 0:
        raise RuntimeError("SOAR returned no matching EUI records")
    return records


def _times(records):
    return np.asarray([parse_iso_datetime(str(value)) for value in records["Start time"]])


def select_channel_sets(request, *, channels, cadence, match_tolerance):
    try:
        from sunpy_soar import Product, SOOP
    except ImportError as error:
        raise ImportError(
            "EUI downloads require the optional sunpy-soar package."
        ) from error
    time = a.Time(request.start, request.end)
    records_by_channel = []
    for channel in channels:
        records_by_channel.append(_soar_records(Fido.search(
            time,
            a.Instrument.eui,
            Product(f"eui-fsi{int(channel)}-image"),
            a.Level(2),
            SOOP("none"),
        )))

    reference_times = _times(records_by_channel[0])
    reference_indices = cadence_indices(
        reference_times, request.start, request.end, cadence
    )
    reference_times = reference_times[reference_indices]
    selected_indices = [reference_indices]
    keep = np.ones(reference_times.shape, dtype=bool)
    for records in records_by_channel[1:]:
        candidate_times = _times(records)
        indices = nearest_indices(reference_times, candidate_times)
        offsets = np.asarray([
            abs(candidate_times[index] - time)
            for index, time in zip(indices, reference_times)
        ])
        keep &= offsets <= match_tolerance
        selected_indices.append(indices)
    if not keep.any():
        raise RuntimeError("No complete EUI channel sets satisfy the match tolerance")
    return tuple(
        records[indices[keep]]
        for records, indices in zip(records_by_channel, selected_indices)
    )


def download(
    request: DownloadRequest,
    *,
    channels=DEFAULT_CHANNELS,
    cadence=None,
    match_tolerance=dt.timedelta(minutes=1),
):
    selected = select_channel_sets(
        request,
        channels=tuple(channels),
        cadence=cadence,
        match_tolerance=match_tolerance,
    )
    return fetch_fido(*selected, request=request, description="EUI download")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument(
        "--match-tolerance",
        type=parse_cadence,
        default=parse_cadence("1m"),
        help="Maximum separation within a multi-channel observation set.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.match_tolerance is None:
        raise SystemExit("Error: --match-tolerance must be a duration, not 'all'.")
    download(
        request_from_args(args),
        channels=args.channels,
        cadence=args.cadence,
        match_tolerance=args.match_tolerance,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
