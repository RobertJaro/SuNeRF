import numpy as np
import pytest

from sunerf.data.loader.thomson_instrument import _temporal_nanpercentile


def test_temporal_percentile_ignores_isolated_invalid_epochs():
    stack = np.array([[[1.0, np.nan]], [[3.0, 4.0]], [[5.0, 6.0]]])
    correction = _temporal_nanpercentile(stack, 50)
    np.testing.assert_allclose(correction, np.array([[[3.0, 5.0]]]))


@pytest.mark.parametrize("level", [-1, 101])
def test_temporal_percentile_validates_level(level):
    with pytest.raises(ValueError, match="between 0 and 100"):
        _temporal_nanpercentile(np.ones((2, 1, 1)), level)
