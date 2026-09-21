#!/usr/bin/env python3
"""Compatibility facade for the split PSI preparation commands."""

from sunerf.data.psi.degrade_psi import fill_nearest_finite, resample_detector_mask
from sunerf.data.psi.prepare_psi_clear import crop_to_fov, radial_arcsec, radial_rsun


def main() -> None:
    raise SystemExit(
        "PSI preparation is now split: run "
        "`python -m sunerf.data.psi.prepare_psi_clear --help` followed by "
        "`python -m sunerf.data.psi.degrade_psi --help`."
    )


if __name__ == "__main__":
    main()
