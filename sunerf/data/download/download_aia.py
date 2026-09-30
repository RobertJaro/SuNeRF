"""Download SDO/AIA level-1 EUV images from JSOC, one export per day chunk."""

from __future__ import annotations

import argparse
import datetime as dt

import drms

from sunerf.data.download.core import (
    DownloadRequest,
    DownloadResult,
    add_common_arguments,
    cadence_string,
    combine_results,
    parse_cadence,
    request_from_args,
    split_request,
)


DEFAULT_CHANNELS = (94, 131, 171, 193, 211, 304, 335)


def aia_dataset(request: DownloadRequest, channels, cadence) -> str:
    """Build a record set covering every requested channel in a single export."""
    duration = (request.end - request.start).total_seconds()
    time = request.start.isoformat("_", timespec="seconds")
    cadence_selector = "" if cadence is None else f"@{cadence_string(cadence)}"
    wavelengths = ",".join(str(int(channel)) for channel in channels)
    return (
        f"aia.lev1_euv_12s[{time} / {duration:g}s{cadence_selector}]"
        f"[{wavelengths}]{{image}}"
    )


def pending_indices(export, request: DownloadRequest) -> list[int]:
    """Return the export records still missing locally, so reruns skip finished files.

    drms appends an ascending number when the target name is taken, so an
    unfiltered rerun silently duplicates every file instead of overwriting it.
    """
    pending = []
    for index, filename in enumerate(export.urls.filename):
        path = request.output / str(filename)
        if path.exists():
            if not request.overwrite:
                continue
            path.unlink()
        pending.append(index)
    return pending


def download(
    request: DownloadRequest,
    *,
    email: str,
    channels=DEFAULT_CHANNELS,
    cadence=None,
    chunk=dt.timedelta(days=7),
):
    client = None if request.dry_run else drms.Client(email=email)
    results = []
    for part in split_request(request, chunk):
        dataset = aia_dataset(part, channels, cadence)
        if request.dry_run:
            print(dataset)
            results.append(DownloadResult(selected=1))
            continue
        request.output.mkdir(parents=True, exist_ok=True)
        export = client.export(dataset, protocol="fits")
        while True:
            try:
                export.wait()
                break
            except TimeoutError:
                # JSOC status polls of a running export can exceed the drms
                # 60 s limit; the request id is kept, so just poll again.
                print(f"Status poll timed out for {export.id}, polling again")
        selected = len(export.urls)
        pending = pending_indices(export, request)
        skipped = selected - len(pending)
        if skipped:
            print(f"Skipping {skipped} already downloaded files of {dataset}")
        if not pending:
            results.append(DownloadResult(selected=selected, skipped=skipped))
            continue
        files = tuple(map(str, export.download(str(request.output), index=pending).download))
        results.append(
            DownloadResult(
                selected=selected, downloaded=len(files), skipped=skipped, files=files
            )
        )
    return combine_results(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--email", required=True, help="JSOC-registered email address.")
    parser.add_argument("--cadence", type=parse_cadence, default=parse_cadence("1h"))
    parser.add_argument("--channels", type=int, nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument(
        "--chunk",
        type=parse_cadence,
        default=parse_cadence("7d"),
        help="Length of each JSOC export (e.g. 7d); 'all' exports the range at once. "
        "All channels are exported together within a chunk.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(
        request_from_args(args),
        email=args.email,
        channels=args.channels,
        cadence=args.cadence,
        chunk=args.chunk,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
