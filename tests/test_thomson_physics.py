import numpy as np
import torch
from torch import nn

from sunerf.model.thomson import ThomsonSuNeRFModule
from sunerf.physics.thomson import (
    LIMB_DARKENING_COEFF,
    R_SUN_CM,
    SIGMA_NE,
    electron_density_normalization_cm3,
)


def test_density_normalization_round_trips_mean_solar_brightness_prefactor():
    msb_norm = 1e-9
    Rs_per_ds = 100
    drho_cm3 = electron_density_normalization_cm3(msb_norm, Rs_per_ds)

    mean_disk_factor = 1.0 - LIMB_DARKENING_COEFF / 3.0
    thomson_factor = np.pi * SIGMA_NE / 2.0
    recovered_model_brightness = (
        thomson_factor
        * drho_cm3
        * Rs_per_ds
        * R_SUN_CM
        / (mean_disk_factor * msb_norm)
    )

    assert np.isclose(recovered_model_brightness, 1.0, rtol=1e-14)


def test_density_normalization_default_white_light_value():
    actual = electron_density_normalization_cm3(1e-9, 100)

    assert np.isclose(actual, 909.3228035477399, rtol=1e-14)


def _ballistic_module(acceleration_limit=1.0):
    module = ThomsonSuNeRFModule.__new__(ThomsonSuNeRFModule)
    nn.Module.__init__(module)
    module.ballistic_acceleration_limit = acceleration_limit
    return module


def _radial_module(tolerance_deg=30.0):
    module = ThomsonSuNeRFModule.__new__(ThomsonSuNeRFModule)
    nn.Module.__init__(module)
    module.radial_tolerance_deg = tolerance_deg
    module.radial_tolerance_cos = np.cos(np.deg2rad(tolerance_deg))
    return module


def test_radial_direction_loss_has_thirty_degree_dead_zone():
    angles_deg = torch.tensor([0.0, 20.0, 30.0, 45.0, 90.0, 180.0])
    angles_rad = torch.deg2rad(angles_deg)
    position = torch.tensor([[1.0, 0.0, 0.0]]).expand(angles_deg.numel(), -1)
    velocity = torch.stack(
        [torch.cos(angles_rad), torch.sin(angles_rad), torch.zeros_like(angles_rad)],
        dim=-1,
    )

    loss = _radial_module().compute_radial_direction_loss(position, velocity)

    torch.testing.assert_close(loss[:3], torch.zeros(3), atol=1e-7, rtol=0.0)
    assert torch.all(loss[3:] > 0)
    assert torch.all(loss[3:-1] < loss[4:])


def test_radial_direction_loss_penalizes_inward_radial_velocity():
    position = torch.tensor([[1.0, 0.0, 0.0]])
    outward = torch.tensor([[2.0, 0.0, 0.0]])
    inward = -outward

    module = _radial_module()

    torch.testing.assert_close(
        module.compute_radial_direction_loss(position, outward), torch.zeros(1)
    )
    assert module.compute_radial_direction_loss(position, inward).item() > 0


def test_ballistic_loss_penalizes_only_excess_acceleration_magnitude():
    query_points = torch.tensor(
        [[0.5, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    velocity = query_points[:, :3]
    velocity_jacobian = ThomsonSuNeRFModule.compute_physics_jacobians(
        query_points, v=velocity
    )['v']

    loss, terms = _ballistic_module().compute_ballistic_loss(velocity, velocity_jacobian)

    torch.testing.assert_close(loss, torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(terms['dv_dt'], torch.zeros(2))
    torch.testing.assert_close(terms['advective_acceleration'], torch.tensor([0.5, 2.0]))
    torch.testing.assert_close(terms['material_acceleration'], torch.tensor([0.5, 2.0]))
    torch.testing.assert_close(terms['acceleration_excess'], torch.tensor([0.0, 1.0]))


def test_ballistic_loss_includes_explicit_time_acceleration():
    query_points = torch.tensor(
        [[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 2.0]],
        requires_grad=True,
    )
    velocity = torch.stack(
        [2.0 * query_points[:, 3], torch.zeros(2), torch.zeros(2)], dim=-1
    )
    velocity_jacobian = ThomsonSuNeRFModule.compute_physics_jacobians(
        query_points, v=velocity
    )['v']

    loss, terms = _ballistic_module().compute_ballistic_loss(velocity, velocity_jacobian)

    torch.testing.assert_close(loss, torch.ones(2))
    torch.testing.assert_close(terms['dv_dt'], torch.full((2,), 2.0))
    torch.testing.assert_close(terms['advective_acceleration'], torch.zeros(2))
    torch.testing.assert_close(terms['material_acceleration'], torch.full((2,), 2.0))
    torch.testing.assert_close(terms['acceleration_excess'], torch.ones(2))
