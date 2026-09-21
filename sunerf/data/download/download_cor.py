"""Download STEREO/SECCHI COR observations from VSO."""

from __future__ import annotations

import argparse

import astropy.units as u
from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    fetch_fido,
    parse_cadence,
    request_from_args,
)


def download(request: DownloadRequest, *, detector="COR2", source="STEREO_A", cadence=None):
    query = [
        a.Time(request.start, request.end),
        a.Source(source),
        a.Instrument("SECCHI"),
        a.Detector(detector),
    ]
    if cadence is not None:
        query.append(a.Sample(cadence.total_seconds() * u.s))
    result = Fido.search(*query)
    if sum(len(block) for block in result) == 0:
        raise RuntimeError(f"No {source}/{detector} records found")
    return fetch_fido(result, request=request, description=f"{source}/{detector} download")


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
