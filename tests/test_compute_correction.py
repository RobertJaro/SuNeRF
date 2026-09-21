import argparse

import numpy as np
import pytest

from sunerf.data.coronagraph.compute_correction import (
    compute_correction_mask,
    percentile_value,
)


def test_daily_percentile_is_computed_over_daily_medians():
    per_day_images = {
        "2026-04-01": [np.full((2, 2), value) for value in (0, 1, 1, 1, 1, 100)],
        "2026-04-02": [np.full((2, 2), 3) for _ in range(6)],
        "2026-04-03": [np.full((2, 2), 5) for _ in range(6)],
    }

    correction = compute_correction_mask(
        stack=np.empty((0, 2, 2)),
        per_day_images=per_day_images,
        correction_type="daily-percentile",
        percentile=50,
    )

    np.testing.assert_array_equal(correction, np.full((2, 2), 3))


def test_daily_min_accepts_lasco_four_frame_cadence():
    per_day_images = {
        "2010-03-15": [np.full((2, 2), value) for value in (1, 2, 3, 20)],
        "2010-03-16": [np.full((2, 2), value) for value in (4, 5, 6, 30)],
    }

    correction = compute_correction_mask(
        stack=np.empty((0, 2, 2)),
        per_day_images=per_day_images,
        correction_type="daily-min",
        min_frames_per_day=4,
    )

    np.testing.assert_array_equal(correction, np.full((2, 2), 2.5))


@pytest.mark.parametrize("value", ["-0.1", "100.1"])
def test_percentile_rejects_values_outside_closed_interval(value):
    with pytest.raises(argparse.ArgumentTypeError):
        percentile_value(value)
