"""Download STEREO HI-1 Level-0 FITS files and the SSC browse images from the STEREO Science Center.

The Level-0 files carry the pointing for rendering the reconstruction on the HI-1 field of view;
the browse JPEGs are the standard quicklook processing of the Science Center, used for the
qualitative comparison (``sunerf.evaluation.cme.plot_hi1_comparison``).  Files go to
``<output>/L0`` and ``<output>/browse``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from urllib.request import urlopen

from sunerf.data.download.core import (
    DownloadRequest, DownloadResult, add_common_arguments, parse_cadence, request_from_args,
)

SSC = "https://stereo-ssc.nascom.nasa.gov"
# science images of HI-1 (s4h1): <stamp>_s4h1A.fts
FILE_PATTERN = re.compile(r"(\d{8}_\d{6})_s4h1([AB])\.fts")


def list_day(day: dt.date, spacecraft: str) -> list[str]:
    """Time stamps of the HI-1 science files of one day."""
    url = f"{SSC}/data/ins_data/secchi/L0/{spacecraft.lower()}/img/hi_1/{day:%Y%m%d}/"
    with urlopen(url) as response:
        html = response.read().decode("latin1")
    return sorted({match.group(1) for match in FILE_PATTERN.finditer(html) if match.group(2) == spacecraft})


def select_stamps(stamps: list[str], request: DownloadRequest, cadence: dt.timedelta | None) -> list[str]:
    """Stamps inside the request period, thinned to one per cadence."""
    selected, last = [], None
    for stamp in stamps:
        time = dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        if not request.start <= time <= request.end:
            continue
        if cadence is not None and last is not None and time - last < cadence:
            continue
        selected.append(stamp)
        last = time
    return selected


def fetch(url: str, path, overwrite: bool, dry_run: bool) -> bool:
    """Download ``url`` to ``path``; return True if a file was written."""
    if path.exists() and not overwrite:
        return False
    if dry_run:
        print(f"would download {url}")
        return True
    with urlopen(url) as response, open(path, "wb") as f:
        f.write(response.read())
    return True


def download(request: DownloadRequest, *, spacecraft="A", cadence=None) -> DownloadResult:
    l0_dir, browse_dir = request.output / "L0", request.output / "browse"
    l0_dir.mkdir(parents=True, exist_ok=True)
    browse_dir.mkdir(parents=True, exist_ok=True)
    label = {"A": "ahead", "B": "behind"}[spacecraft]
    files, downloaded, skipped = [], 0, 0
    day = request.start.date()
    while day <= request.end.date():
        for stamp in select_stamps(list_day(day, spacecraft), request, cadence):
            name = f"{stamp}_s4h1{spacecraft}"
            l0_url = f"{SSC}/data/ins_data/secchi/L0/{spacecraft.lower()}/img/hi_1/{day:%Y%m%d}/{name}.fts"
            browse_url = f"{SSC}/pub/browse/{day:%Y/%m/%d}/{label}/hi1/1024/{name}.jpg"
            for url, path in ((l0_url, l0_dir / f"{name}.fts"), (browse_url, browse_dir / f"{name}.jpg")):
                if fetch(url, path, request.overwrite, request.dry_run):
                    downloaded += 1
                else:
                    skipped += 1
                files.append(str(path))
        day += dt.timedelta(days=1)
    print(f"HI-1 {spacecraft}: {downloaded} files downloaded, {skipped} already present")
    return DownloadResult(selected=len(files), downloaded=downloaded, skipped=skipped, files=tuple(files))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_output="hi1")
    parser.add_argument("--spacecraft", default="A", choices=("A", "B"))
    parser.add_argument("--cadence", type=parse_cadence, default=None,
                        help="Keep one file per cadence (e.g. 4h); default: every file")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    download(request_from_args(args), spacecraft=args.spacecraft, cadence=args.cadence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
