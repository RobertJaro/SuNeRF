import numpy as np
import pytest
import torch

from sunerf.model.spherical_grid import SphericalGridPlasmaModel
from sunerf.train.coordinate_transformation import spherical_to_cartesian


def _model(log_density, log_temperature, *, time=(0.0, 2.0), log_T=(5.0, 6.0, 7.0)):
    return SphericalGridPlasmaModel(
        log_density=log_density,
        log_temperature=log_temperature,
        time=time,
        radius=(1.0, 3.0),
        latitude=(-0.5, 0.5),
        longitude=(0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
        log_T=log_T,
    )


def _cartesian_query(radius, latitude, longitude, time):
    spherical = torch.tensor([radius, latitude, longitude], dtype=torch.float32)
    xyz = spherical_to_cartesian(spherical, torch)
    return torch.cat([xyz, torch.tensor([time])])


def test_nonuniform_four_dimensional_interpolation():
    time = np.array([0.0, 2.0], dtype=np.float32)
    radius = np.array([1.0, 3.0], dtype=np.float32)
    latitude = np.array([-0.5, 0.5], dtype=np.float32)
    longitude = np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi], dtype=np.float32)
    tt, rr, aa, ll = np.meshgrid(time, radius, latitude, longitude, indexing="ij")
    log_density = tt + 2 * rr + 3 * aa + 0.25 * ll
    log_temperature = 5.5 + 0.1 * tt + 0.05 * rr
    model = _model(log_density, log_temperature)

    query = _cartesian_query(2.0, 0.0, 0.25 * np.pi, 1.0)
    fields = model._interpolate_fields(query[None])

    torch.testing.assert_close(fields[0, 0], torch.tensor(5.0 + 0.25 * 0.25 * np.pi))
    torch.testing.assert_close(fields[0, 1], torch.tensor(5.7))


def test_longitude_interpolation_is_periodic_at_seam():
    shape = (1, 2, 2, 4)
    longitude_values = np.array([0.0, -1.0, 0.0, 1.0], dtype=np.float32)
    log_density = np.broadcast_to(longitude_values, shape).copy()
    log_temperature = np.full(shape, 6.0, dtype=np.float32)
    model = _model(log_density, log_temperature, time=(0.0,))

    before_zero = _cartesian_query(2.0, 0.0, -0.25 * np.pi, 123.0)
    after_last = _cartesian_query(2.0, 0.0, 1.75 * np.pi, -123.0)
    values = model._interpolate_fields(torch.stack([before_zero, after_last]))

    torch.testing.assert_close(values[:, 0], torch.full((2,), 0.5), atol=1e-6, rtol=1e-6)


def test_field_returns_the_pointwise_plasma_state_without_temperature_binning():
    shape = (1, 2, 2, 4)
    model = _model(
        np.full(shape, np.log10(2.0), dtype=np.float32),
        np.full(shape, 7.5, dtype=np.float32),
        time=(0.0,),
    )

    output = model(_cartesian_query(2.0, 0.0, 0.25, 0.0)[None])

    # The renderer decides whether a temperature lies inside the response
    # support; the field itself never clips or bins the simulation state.
    assert set(output) == {"mean_log_T", "total_ne", "total_log_ne"}
    torch.testing.assert_close(output["mean_log_T"], torch.tensor([[7.5]]))
    torch.testing.assert_close(output["total_ne"], torch.tensor([[2.0]]))


def test_points_outside_spatial_grid_have_negligible_density():
    shape = (1, 2, 2, 4)
    model = _model(
        np.zeros(shape, dtype=np.float32),
        np.full(shape, 6.0, dtype=np.float32),
        time=(0.0,),
    )
    output = model(_cartesian_query(4.0, 0.0, 0.0, 0.0)[None])

    torch.testing.assert_close(output["total_log_ne"], torch.tensor([[-30.0]]))


def test_density_and_temperature_corners_are_interpolated_as_pairs():
    shape = (1, 2, 2, 4)
    log_density = np.zeros(shape, dtype=np.float32)
    log_temperature = np.full(shape, 6.0, dtype=np.float32)
    # The nearest interpolation corner has a density but no temperature.  It
    # must be excluded from both fields rather than contributing only density.
    log_density[0, 0, 0, 0] = 9.0
    log_temperature[0, 0, 0, 0] = np.nan
    model = _model(log_density, log_temperature, time=(0.0,))

    query = _cartesian_query(1.5, -0.25, 0.125 * np.pi, 0.0)
    fields = model._interpolate_fields(query[None])

    torch.testing.assert_close(fields, torch.tensor([[0.0, 6.0]]))


def test_invalid_grid_fill_cannot_create_observable_density():
    shape = (1, 2, 2, 4)
    with pytest.raises(ValueError, match="fill_log_density"):
        SphericalGridPlasmaModel(
            log_density=np.zeros(shape, dtype=np.float32),
            log_temperature=np.full(shape, 6.0, dtype=np.float32),
            time=(0.0,),
            radius=(1.0, 3.0),
            latitude=(-0.5, 0.5),
            longitude=(0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
            log_T=(5.0, 6.0, 7.0),
            fill_log_density=0.0,
        )
