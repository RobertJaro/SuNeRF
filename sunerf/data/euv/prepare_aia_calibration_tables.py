"""Prepare the AIA degradation and master-pointing calibration tables.

The degradation table is the packaged SolarSoft V10 response table by default,
so preparation needs no SSW installation. The master-pointing table covers the
requested time range only and is always queried from JSOC.
"""

from argparse import ArgumentParser
from pathlib import Path

import astropy.units as u
from aiapy.calibrate.utils import get_correction_table, get_pointing_table
from astropy.time import Time

from sunerf.resources import is_builtin, resolve_artifact_path

# aiapy fetches the most recent table itself for these two keywords.
REMOTE_CORRECTION_SOURCES = ('SSW', 'JSOC')
PACKAGED_CORRECTION_SOURCE = 'builtin:aia_correction_table'


def resolve_correction_source(source: str) -> Path | str:
    """Return a remote keyword unchanged, or a verified local table path.

    A missing path must be rejected here: aiapy hands the value to
    ``astropy.io.ascii.read``, which then treats it as literal table content and
    fails with an unrelated ``AttributeError``.
    """
    if source.upper() in REMOTE_CORRECTION_SOURCES:
        return source.upper()
    if is_builtin(source):
        return resolve_artifact_path(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(
            f'AIA correction table {path} does not exist. Use '
            f'{PACKAGED_CORRECTION_SOURCE!r} for the packaged SolarSoft table, '
            f'or pass one of {REMOTE_CORRECTION_SOURCES} to fetch it.'
        )
    return path


def prepare_tables(
    correction_path: Path,
    pointing_path: Path,
    start: Time,
    end: Time,
    correction_source: Path | str = PACKAGED_CORRECTION_SOURCE,
) -> None:
    if not correction_path.exists():
        source = resolve_correction_source(str(correction_source))
        correction_path.parent.mkdir(parents=True, exist_ok=True)
        get_correction_table(source).write(correction_path, format="ascii.ecsv")
    if not pointing_path.exists():
        pointing_path.parent.mkdir(parents=True, exist_ok=True)
        get_pointing_table("JSOC", time_range=(start - 12 * u.hour, end + 12 * u.hour)).write(
            pointing_path, format="ascii.ecsv"
        )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("correction_path", type=Path)
    parser.add_argument("pointing_path", type=Path)
    parser.add_argument("start", type=Time)
    parser.add_argument("end", type=Time)
    parser.add_argument(
        "--correction-source",
        default=PACKAGED_CORRECTION_SOURCE,
        help=(
            "Packaged table (default), a path to an SSW "
            "aia_V*_response_table.txt, or 'SSW'/'JSOC' to fetch the latest."
        ),
    )
    args = parser.parse_args()
    prepare_tables(
        args.correction_path,
        args.pointing_path,
        args.start,
        args.end,
        correction_source=args.correction_source,
    )


if __name__ == "__main__":
    main()
