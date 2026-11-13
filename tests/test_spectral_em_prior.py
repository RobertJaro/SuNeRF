import numpy as np
import pytest
import torch

from sunerf.physics.euv import R_SUN_CM
from tests.test_plasma_alignment import _instruments, _module


def test_edge_weight_is_zero_inside_range_and_smooth_to_one_at_bounds(tmp_path):
    # Fixture grid spans log T 5..6; the prior is free between 5.25 and 5.75.
    module = _module(_instruments(tmp_path), regularization_log_T_range=[5.25, 5.75])
    log_T = torch.tensor([5.0, 5.125, 5.25, 5.5, 5.75, 5.875, 6.0])
    weight = module.unconstrained_temperature_weight(log_T)
    np.testing.assert_allclose(weight[[2, 3, 4]].numpy(), 0.0)
    np.testing.assert_allclose(weight[[0, 6]].numpy(), 1.0)
    np.testing.assert_allclose(weight[[1, 5]].numpy(), 0.5, rtol=1e-6)  # smoothstep midpoint
    # The ramp starts with zero slope at the edge: a tenth of the way out the
    # weight is 0.028, well below the linear 0.1.
    near_edge = module.unconstrained_temperature_weight(torch.tensor([5.775]))
    np.testing.assert_allclose(near_edge.item(), 0.028, rtol=1e-3)


def test_penalty_charges_edge_temperatures_in_column_units(tmp_path):
    module = _module(_instruments(tmp_path), regularization_log_T_range=[5.25, 5.75])
    em = torch.full((3, 4, 1), 1e16)  # (1e8 cm^-3)^2
    dl = torch.full((3, 4), R_SUN_CM / 4)  # one solar radius per ray
    log_T = torch.full((3, 4, 1), 5.5)
    assert module.compute_spectral_em_regularization(em, log_T, dl).item() == 0.0
    log_T[1] = 6.0  # at the bound: full weight on one of three rays
    penalty = module.compute_spectral_em_regularization(em, log_T, dl)
    np.testing.assert_allclose(penalty.item(), 1.0 / 3.0, rtol=1e-5)


def test_range_is_required_when_enabled_and_must_lie_inside_the_grid(tmp_path):
    config = _instruments(tmp_path)
    with pytest.raises(ValueError, match='regularization_log_T_range'):
        _module(config, lambda_config={'regularization': 1e-2})
    with pytest.raises(ValueError, match='regularization_log_T_range'):
        _module(config, regularization_log_T_range=[4.5, 5.5])
    with pytest.raises(ValueError, match='regularization_log_T_range'):
        _module(config, regularization_log_T_range=[5.5, 5.5])
    assert _module(config, lambda_config={'regularization': 1e-2},
                   regularization_log_T_range=[5.0, 6.0]).construction_spec['regularization_log_T_range'] == [5.0, 6.0]
