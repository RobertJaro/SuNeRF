"""Prepare the AIA degradation and master-pointing calibration tables."""

from argparse import ArgumentParser
from pathlib import Path

import astropy.units as u
from astropy.time import Time
from aiapy.calibrate.utils import get_correction_table, get_pointing_table


def prepare_tables(
    correction_source: Path,
    correction_path: Path,
    pointing_path: Path,
    start: Time,
    end: Time,
) -> None:
    if not correction_path.exists():
        get_correction_table(correction_source).write(correction_path, format="ascii.ecsv")
    if not pointing_path.exists():
        get_pointing_table("JSOC", time_range=(start - 12 * u.hour, end + 12 * u.hour)).write(
            pointing_path, format="ascii.ecsv"
        )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("correction_source", type=Path)
    parser.add_argument("correction_path", type=Path)
    parser.add_argument("pointing_path", type=Path)
    parser.add_argument("start", type=Time)
    parser.add_argument("end", type=Time)
    args = parser.parse_args()
    prepare_tables(
        args.correction_source,
        args.correction_path,
        args.pointing_path,
        args.start,
        args.end,
    )


if __name__ == "__main__":
    main()
