"""Delete raw or calibrated EUV inputs whose prepared product exists.

A source file is removed only when ``<prepared dir>/<name>.prepared.fits`` (or
the same name in its ``rejected/`` subdirectory, where the frame review moves
rejected products) exists. Prepared products are written atomically, so an
existing product is complete. Sources without a product (not prepared yet, or
skipped by the preparation) are kept and counted.

The product name is the source stem plus ``.prepared.fits``. STEREO/SECCHI
level-0.5 files are named ``YYYYMMDD_HHMMSS_<t>4euA.fts`` with an image-type
letter ``<t>`` (``n``, ``s``, ...) and SECCHI_PREP writes
``YYYYMMDD_HHMMSS_14euA.fts``; ``--secchi-level-0`` maps the first name to the
product of the second, so raw EUVI files are matched directly. A source whose
name cannot be mapped is kept and listed.

    python -m sunerf.data.euv.clear_sources "raw/aia/*.fits" --prepared-dir prepared/aia
    python -m sunerf.data.euv.clear_sources "raw/euvi_a/*.fts" --prepared-dir prepared/euvi_a --secchi-level-0
"""

import argparse
import glob
import os
from pathlib import Path

from sunerf.data.frame_review import REJECTED_DIRECTORY

# Character of a SECCHI file name that SECCHI_PREP replaces by '1': the
# image-type letter of a level-0.5 file ('n', 's', ...).
SECCHI_LEVEL_INDEX = 16


def prepared_name(source, secchi_level_0=False):
    stem = Path(source).stem
    if secchi_level_0:
        if len(stem) <= SECCHI_LEVEL_INDEX or not stem[SECCHI_LEVEL_INDEX].isalpha():
            raise ValueError(f'{source} is not a SECCHI level-0.5 file name (YYYYMMDD_HHMMSS_<letter>...)')
        stem = stem[:SECCHI_LEVEL_INDEX] + '1' + stem[SECCHI_LEVEL_INDEX + 1:]
    return f'{stem}.prepared.fits'


def has_product(source, prepared_dir, *, secchi_level_0=False):
    """True when the prepared product of ``source`` exists (also when moved to ``rejected/``)."""
    name = prepared_name(source, secchi_level_0)
    prepared_dir = Path(prepared_dir)
    return (prepared_dir / name).is_file() or (prepared_dir / REJECTED_DIRECTORY / name).is_file()


def clear_sources(pattern, prepared_dir, *, secchi_level_0=False, dry_run=False):
    """Remove the sources matching ``pattern`` that have a prepared product; returns (removed, kept)."""
    removed, kept, freed = 0, 0, 0
    unmatched = []
    for source in sorted(glob.glob(pattern)):
        try:
            prepared = has_product(source, prepared_dir, secchi_level_0=secchi_level_0)
        except ValueError:
            unmatched.append(source)
            prepared = False
        if prepared:
            freed += os.path.getsize(source)
            if not dry_run:
                os.remove(source)
            removed += 1
        else:
            kept += 1
    action = 'would remove' if dry_run else 'removed'
    print(f'{pattern}: {action} {removed} files ({freed / 1e9:.1f} GB) with a prepared product; '
          f'kept {kept} without one')
    if unmatched:
        print(f'{len(unmatched)} kept files have no recognized SECCHI level-0.5 name:')
        for source in unmatched:
            print(f'  {source}')
    return removed, kept


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pattern', help='glob of the source files (quote it)')
    parser.add_argument('--prepared-dir', required=True, help='directory of the prepared products')
    parser.add_argument('--secchi-level-0', action='store_true',
                        help='sources are SECCHI level-0.5 files, products come from their SECCHI_PREP output')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    clear_sources(args.pattern, args.prepared_dir, secchi_level_0=args.secchi_level_0, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
