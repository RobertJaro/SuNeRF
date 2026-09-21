"""View-agnostic constants and path helpers for PSI experiments."""

from pathlib import Path


PRODUCTS = ("pb", "tb")
FRAME_ID = 50
FRAME_TOKEN = f"{FRAME_ID:03d}"
REF_DATE = "2021-10-28T15:30:00"
OUTPUT_SHAPE = (512, 512)


def channel_file(
    root: Path, view: str, product: str, frame_token: str = FRAME_TOKEN
) -> Path:
    """Return the unique fixed-frame FITS file, with a useful error otherwise."""
    matches = sorted((root / view / product).glob(f"*{frame_token}.fts"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {view}/{product} frame *{frame_token}.fts under {root}; "
            f"found {len(matches)}."
        )
    return matches[0]
