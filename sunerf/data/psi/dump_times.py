#!/usr/bin/env python3
"""Tabulate the observation time of every PSI/MAS simulation dump.

The density cubes carry no time stamp.  PSI's synthetic coronagraph images of
the same run do: every FITS header holds the dump number (``SIM_DUMP``) and its
observation time (``DATE_OBS``).  This command reads those headers, from local
files or straight from the PSI web directory, and writes ``{dump: time}`` as
JSON.  ``render_psi_thomson`` places every cube at its tabulated time, so the
rendered series keeps the cadence of the simulation.

Only the header blocks are read; the image data are never transferred.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time

DEFAULT_URL = "https://www.predsci.com/~epalmerio/getpb/20211028/fakeC3/fits_L1/pb/"
_FITS_BLOCK = 2880
_MAX_HEADER_BLOCKS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fits-dir", type=Path, default=None,
                        help="Directory with one PSI synthetic image series (*.fts, *.fits).")
    source.add_argument("--url", default=DEFAULT_URL,
                        help="PSI web directory that lists one synthetic image series.")
    parser.add_argument("--out-file", type=Path, required=True, help="Output JSON table.")
    return parser.parse_args()


def parse_psi_header(raw: bytes) -> fits.Header:
    """Parse PSI header blocks, which hold tabs that astropy refuses to read."""
    text = raw.decode("latin-1")
    cards = [text[start:start + 80] for start in range(0, len(text), 80)]
    cards = ["".join(c if 32 <= ord(c) <= 126 else " " for c in card) for card in cards]
    end = next((index for index, card in enumerate(cards) if card.startswith("END")), None)
    if end is None:
        raise ValueError("PSI FITS header has no END card.")
    return fits.Header.fromstring("".join(cards[:end + 1]), sep="")


def _complete(raw: bytes) -> bool:
    return any(raw[start:start + 3] == b"END" for start in range(0, len(raw), 80))


def read_local_header(path: Path) -> fits.Header:
    raw = b""
    with open(path, "rb") as file:
        for _ in range(_MAX_HEADER_BLOCKS):
            raw += file.read(_FITS_BLOCK)
            if _complete(raw):
                return parse_psi_header(raw)
    raise ValueError(f"No FITS header end found in {path}.")


def read_remote_header(url: str) -> fits.Header:
    raw = b""
    for block in range(_MAX_HEADER_BLOCKS):
        start = block * _FITS_BLOCK
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{start + _FITS_BLOCK - 1}"}
        )
        with urllib.request.urlopen(request) as response:
            if response.status != 206:
                raise RuntimeError(f"{url} does not serve byte ranges; download the series "
                                   "and pass --fits-dir instead.")
            raw += response.read()
        if _complete(raw):
            return parse_psi_header(raw)
    raise ValueError(f"No FITS header end found in {url}.")


def list_remote_files(url: str) -> list[str]:
    url = url.rstrip("/") + "/"
    with urllib.request.urlopen(url) as response:
        listing = response.read().decode("utf-8", errors="replace")
    names = sorted(set(re.findall(r'href="([^"/?]+\.(?:fts|fits))"', listing)))
    if not names:
        raise FileNotFoundError(f"No FITS files listed under {url}.")
    return [url + name for name in names]


def dump_time(header: fits.Header) -> tuple[int, str]:
    date = header.get("DATE_OBS", header.get("DATE-OBS"))
    if date is None or "SIM_DUMP" not in header:
        raise KeyError("PSI header lacks SIM_DUMP or DATE_OBS.")
    return int(header["SIM_DUMP"]), Time(str(date).strip()).isot


def load_dump_times(path) -> dict[int, Time]:
    """Read the table written by this command as ``{dump: astropy Time}``."""
    table = json.loads(Path(path).read_text())["dump_times"]
    return {int(dump): Time(time) for dump, time in table.items()}


def cadence_summary(dump_times: dict[int, str]) -> dict:
    dumps = sorted(dump_times)
    if len(dumps) < 2:
        raise ValueError("At least two PSI dumps are needed to establish the cadence.")
    seconds =np.array([Time(dump_times[dump]).unix for dump in dumps])
    steps = np.diff(seconds) / np.diff(dumps)
    if np.any(steps <= 0):
        raise ValueError("PSI dump times do not increase with the dump number.")
    return {
        "n_dumps": len(dumps),
        "first": dump_times[dumps[0]],
        "last": dump_times[dumps[-1]],
        "cadence_seconds_min": float(steps.min()),
        "cadence_seconds_median": float(np.median(steps)),
        "cadence_seconds_max": float(steps.max()),
    }


def main() -> None:
    args = parse_args()
    if args.fits_dir is not None:
        files = sorted([*args.fits_dir.glob("*.fts"), *args.fits_dir.glob("*.fits")])
        if not files:
            raise FileNotFoundError(f"No FITS files found in {args.fits_dir}.")
        source, headers = str(args.fits_dir), (read_local_header(path) for path in files)
    else:
        source = args.url
        headers = (read_remote_header(url) for url in list_remote_files(args.url))

    dump_times = {}
    for header in headers:
        dump, time = dump_time(header)
        if dump_times.setdefault(dump, time) != time:
            raise ValueError(f"Dump {dump} has two observation times: {dump_times[dump]}, {time}.")
    dump_times = dict(sorted(dump_times.items()))

    summary = cadence_summary(dump_times)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_file.write_text(json.dumps(
        {"source": source, **summary, "dump_times": dump_times}, indent=2
    ))
    print(f"PSI dump times: {summary['n_dumps']} dumps, {summary['first']} -- {summary['last']}, "
          f"cadence {summary['cadence_seconds_min']:.0f}--{summary['cadence_seconds_max']:.0f} s "
          f"-> {args.out_file}")


if __name__ == "__main__":
    main()
