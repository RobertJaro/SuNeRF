"""Download time-matched STEREO/SECCHI EUVI channel sets from VSO."""

from __future__ import annotations

import argparse

import astropy.units as u
import numpy as np
from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    combine_results,
    fetch_fido,
    nearest_indices,
    parse_cadence,
    request_from_args,
    unique_indices,
)


DEFAULT_CHANNELS = (171, 195, 284)
DEFAULT_SOURCES = ("STEREO_A", "STEREO_B")


def _vso_records(response):
    try:
        records = response["vso"]
    except (KeyError, IndexError) as error:
        raise RuntimeError("VSO returned no matching records") from error
    if len(records) == 0:
        raise RuntimeError("VSO returned no matching records")
    return records


def _times(records):
    return np.asarray(records["Start Time"].datetime)


def select_channel_sets(request, *, source, channels, cadence):
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
    anchors = _vso_records(Fido.search(*anchor_query))
    anchor_times = _times(anchors)

    selected = []
    for channel in channels:
        records = _vso_records(Fido.search(
            time,
            a.Instrument.secchi,
            a.Detector.euvi,
            a.Source(source),
            a.Wavelength(channel * u.AA),
        ))
        indices = unique_indices(nearest_indices(anchor_times, _times(records)))
        selected.append(records[indices])
    return tuple(selected)


def download(
    request: DownloadRequest,
    *,
    channels=DEFAULT_CHANNELS,
    sources=DEFAULT_SOURCES,
    cadence=None,
):
    if not channels:
        raise ValueError("channels must not be empty")
    results = []
    for source in sources:
        selected = select_channel_sets(
            request, source=source, channels=tuple(channels), cadence=cadence
        )
        results.append(fetch_fido(
            *selected,
            request=request,
            description=f"EUVI download for {source}",
        ))
    return combine_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        channels=args.channels,
        sources=args.sources,
        cadence=args.cadence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
