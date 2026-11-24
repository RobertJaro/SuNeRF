"""Identify prepared PUNCH and COR2 brightness pairs across review directories."""
import re
from pathlib import Path

PAIR_STEM_RE = re.compile(r"^(?P<prefix>\d{8}_\d{6})_1(?P<kind>[PB])(?P<suffix>.+)$")


def normalize_pair_stem(path):
    stem = Path(path).stem
    return PAIR_STEM_RE.sub(r"\g<prefix>_1X\g<suffix>", stem)


def exclude_rejected_partners(tb_files, pb_files):
    """Omit a kept partner from paired products only when the other file was moved."""
    if pb_files is None:
        return tb_files, pb_files

    def usable(files, channel, other):
        skipped = set()
        for directory in {Path(path).parent for path in files}:
            if directory.name != channel:
                continue
            sibling = directory.parent / other
            for rejected in (sibling / 'rejected').glob('*'):
                if rejected.is_file() and rejected.name != 'rejected.log' and not (sibling / rejected.name).exists():
                    skipped.add((directory, normalize_pair_stem(rejected)))
        if not skipped:
            return files
        return [f for f in files if (Path(f).parent, normalize_pair_stem(f)) not in skipped]

    return usable(tb_files, 'tB', 'pB'), usable(pb_files, 'pB', 'tB')
