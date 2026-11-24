"""Download SDO/AIA level-1 EUV images from JSOC, one export per day chunk.

Records already present are not downloaded again: a file in the output
directory, or, with ``--prepared-dir``, a prepared product (also one in
``rejected/``) whose raw file was deleted. A chunk whose records are all
present is not exported.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import time

import drms

from sunerf.data.euv.clear_sources import has_product
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
# drms error suffix of a request refused because another export is pending.
PENDING_EXPORT_STATUS = "[status=7]"
PENDING_EXPORT_WAIT_SECONDS = 60


SERIES = "aia.lev1_euv_12s"
SEGMENT = "image"


def aia_records(request: DownloadRequest, channels, cadence) -> str:
    """Record set of every requested channel, without the segment."""
    duration = (request.end - request.start).total_seconds()
    time = request.start.isoformat("_", timespec="seconds")
    cadence_selector = "" if cadence is None else f"@{cadence_string(cadence)}"
    wavelengths = ",".join(str(int(channel)) for channel in channels)
    return f"{SERIES}[{time} / {duration:g}s{cadence_selector}][{wavelengths}]"


def aia_dataset(request: DownloadRequest, channels, cadence) -> str:
    """Build a record set covering every requested channel in a single export."""
    return f"{aia_records(request, channels, cadence)}{{{SEGMENT}}}"


def export_filename(t_rec, wavelength) -> str:
    """JSOC export name of one record, e.g.
    ``aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.fits``."""
    return f"{SERIES}.{str(t_rec).replace(':', '')}.{int(wavelength)}.{SEGMENT}_lev1.fits"


def is_present(filename, request: DownloadRequest, prepared_dir=None) -> bool:
    """True when the raw file or its prepared product exists."""
    if (request.output / filename).exists():
        return True
    return prepared_dir is not None and has_product(filename, prepared_dir)


def chunk_complete(client, records, request: DownloadRequest, prepared_dir=None) -> bool:
    """True when every record of ``records`` is present locally, so the chunk
    needs no export. The expected names only decide whether to export: a
    failed query or a name mismatch costs the export, never a record."""
    try:
        keys = client.query(records, key="T_REC,WAVELNTH")
    except Exception as error:  # noqa: BLE001
        print(f"Query of {records} failed ({error}); exporting the chunk")
        return False
    return len(keys) > 0 and all(
        is_present(export_filename(t_rec, wavelength), request, prepared_dir)
        for t_rec, wavelength in zip(keys["T_REC"], keys["WAVELNTH"])
    )


def _export(client, dataset):
    """Submit one export and wait until it is finished.

    JSOC accepts one pending export per user: while another request of the
    same user is still processing (e.g. from a job that was cut off), a new one
    is refused with status 7, so wait and submit again.
    """
    while True:
        try:
            export = client.export(dataset, protocol="fits")
            while True:
                try:
                    export.wait()
                    return export
                except TimeoutError:
                    # JSOC status polls of a running export can exceed the drms
                    # 60 s limit; the request id is kept, so just poll again.
                    print(f"Status poll timed out for {export.id}, polling again")
        except drms.DrmsExportError as error:
            if PENDING_EXPORT_STATUS not in str(error):
                raise
            print(f"{error}; submitting again in {PENDING_EXPORT_WAIT_SECONDS} s")
            time.sleep(PENDING_EXPORT_WAIT_SECONDS)


def pending_indices(export, request: DownloadRequest, prepared_dir=None) -> list[int]:
    """Return the export records still missing locally, so reruns skip finished files.

    drms appends an ascending number when the target name is taken, so an
    unfiltered rerun silently duplicates every file instead of overwriting it.
    """
    pending = []
    for index, filename in enumerate(export.urls.filename):
        path = request.output / str(filename)
        if request.overwrite:
            if path.exists():
                path.unlink()
        elif is_present(str(filename), request, prepared_dir):
            continue
        pending.append(index)
    return pending


def download(
    request: DownloadRequest,
    *,
    email: str,
    channels=DEFAULT_CHANNELS,
    cadence=None,
    chunk=dt.timedelta(days=7),
    prepared_dir=None,
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
        if not request.overwrite and chunk_complete(
            client, aia_records(part, channels, cadence), request, prepared_dir
        ):
            print(f"Skipping {dataset}: every record is already present")
            results.append(DownloadResult(selected=0))
            continue
        export = _export(client, dataset)
        selected = len(export.urls)
        pending = pending_indices(export, request, prepared_dir)
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
        email=args.email,
        channels=args.channels,
        cadence=args.cadence,
        chunk=args.chunk,
        prepared_dir=args.prepared_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
