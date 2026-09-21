"""Download SDO/AIA level-1 EUV images from JSOC."""

from __future__ import annotations

import argparse

import drms

from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    cadence_string,
    combine_results,
    parse_cadence,
    request_from_args,
)
from sunerf.data.download.download_jsoc import download_jsoc_export


DEFAULT_CHANNELS = (94, 131, 171, 193, 211, 304, 335)


def aia_dataset(request: DownloadRequest, channel: int, cadence) -> str:
    duration = (request.end - request.start).total_seconds()
    time = request.start.isoformat("_", timespec="seconds")
    cadence_selector = "" if cadence is None else f"@{cadence_string(cadence)}"
    return (
        f"aia.lev1_euv_12s[{time} / {duration:g}s{cadence_selector}]"
        f"[{int(channel)}]{{image}}"
    )


def download(
    request: DownloadRequest,
    *,
    email: str,
    channels=DEFAULT_CHANNELS,
    cadence=None,
):
    client = None if request.dry_run else drms.Client(email=email)
    results = []
    for channel in channels:
        dataset = aia_dataset(request, channel, cadence)
        results.append(
            download_jsoc_export(
                dataset,
                request.output,
                client,
                dry_run=request.dry_run,
            )
        )
    return combine_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--email", required=True, help="JSOC-registered email address.")
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    request = request_from_args(args)
    download(request, email=args.email, channels=args.channels, cadence=args.cadence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
