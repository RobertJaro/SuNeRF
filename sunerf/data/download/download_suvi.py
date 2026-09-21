"""Download time-matched GOES/SUVI level-2 channel sets."""

from __future__ import annotations

import argparse

import astropy.units as u
import numpy as np
from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    cadence_indices,
    fetch_fido,
    nearest_indices,
    parse_cadence,
    request_from_args,
    unique_indices,
)


DEFAULT_CHANNELS = (94, 131, 171, 195, 284, 304)


def _records(response):
    if len(response) == 0 or len(response[0]) == 0:
        raise RuntimeError("Fido returned no matching SUVI records")
    return response[0]


def select_channel_sets(request, *, channels, cadence, satellite):
    time = a.Time(request.start, request.end)
    records_by_channel = [
        _records(Fido.search(
            time,
            a.Instrument("suvi"),
            a.Level.two,
            a.goes.SatelliteNumber(int(satellite)),
            a.Wavelength(int(channel) * u.AA),
        ))
        for channel in channels
    ]
    reference_times = np.asarray(records_by_channel[-1]["Start Time"])
    reference_indices = cadence_indices(
        reference_times, request.start, request.end, cadence
    )
    reference_times = reference_times[reference_indices]
    selected = [records_by_channel[-1][reference_indices]]
    for records in records_by_channel[:-1]:
        indices = unique_indices(nearest_indices(reference_times, records["Start Time"]))
        selected.append(records[indices])
    # Restore configured channel order after using the last channel as anchor.
    return tuple(selected[1:] + selected[:1])


def download(
    request: DownloadRequest,
    *,
    channels=DEFAULT_CHANNELS,
    cadence=None,
    satellite=16,
):
    selected = select_channel_sets(
        request,
        channels=tuple(channels),
        cadence=cadence,
        satellite=satellite,
    )
    return fetch_fido(*selected, request=request, description="SUVI download")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--satellite", type=int, default=16)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        channels=args.channels,
        cadence=args.cadence,
        satellite=args.satellite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
