"""Download STEREO/SECCHI COR observations from VSO."""

from __future__ import annotations

import argparse
import datetime as dt

import astropy.units as u
from astropy.time import Time
from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    fetch_fido,
    parse_cadence,
    request_from_args,
    split_request,
)

# VSO returns at most 4000 records per query; one day of COR2 is ~300.
QUERY_CHUNK = dt.timedelta(days=1)


def search_chunk(request: DownloadRequest, *, detector, source, cadence):
    query = [
        a.Time(request.start, request.end),
        a.Source(source),
        a.Instrument("SECCHI"),
        a.Detector(detector),
    ]
    if cadence is not None:
        query.append(a.Sample(cadence.total_seconds() * u.s))
    tables = []
    for table in Fido.search(*query):
        # Records overlapping a chunk boundary are returned by both chunks.
        start = Time(table["Start Time"])
        table = table[(start >= Time(request.start)) & (start < Time(request.end))]
        if len(table) > 0:
            tables.append(table)
    return tables


def download(request: DownloadRequest, *, detector="COR2", source="STEREO_A", cadence=None):
    tables = [
        table
        for chunk in split_request(request, QUERY_CHUNK)
        for table in search_chunk(chunk, detector=detector, source=source, cadence=cadence)
    ]
    if not tables:
        raise RuntimeError(f"No {source}/{detector} records found")
    return fetch_fido(*tables, request=request, description=f"{source}/{detector} download")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_output="stereo")
    parser.add_argument("--detector", default="COR2", choices=("COR1", "COR2"))
    parser.add_argument("--source", default="STEREO_A", choices=("STEREO_A", "STEREO_B"))
    parser.add_argument("--cadence", type=parse_cadence, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        detector=args.detector,
        source=args.source,
        cadence=args.cadence,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
