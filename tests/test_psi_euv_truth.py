import numpy as np
import torch

from sunerf.evaluation.psi_euv_truth import compare, evaluation_points, query_model
from sunerf.model.spherical_grid import SphericalGridPlasmaModel


def _field(log_density, log_temperature):
    shape = (1, 2, 2, 4)
    return SphericalGridPlasmaModel(
        log_density=np.full(shape, log_density, dtype=np.float32),
        log_temperature=np.full(shape, log_temperature, dtype=np.float32),
        time=(0.0,),
        radius=(1.0, 1.5),
        latitude=(-1.5, 1.5),
        longitude=(0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi),
        log_T=(4.0, 8.0),
    )


def test_truth_comparison_reports_radially_binned_bias():
    radius = np.linspace(1.0, 1.5, 11)
    points = evaluation_points(radius, np.linspace(-1.0, 1.0, 5), np.linspace(0.0, 6.0, 7))
    truth = query_model(_field(8.0, 6.0), points)
    reconstruction = query_model(_field(8.2, 5.9), points)

    assert points.shape == (11, 5, 7, 4)
    torch.testing.assert_close(
        torch.linalg.norm(points[..., :3], dim=-1)[:, 0, 0],
        torch.as_tensor(radius, dtype=torch.float32),
    )
    report = compare(truth, reconstruction, radius, emitting_log_T=5.7)
    for statistics in report.values():
        assert np.isclose(statistics["log_ne_emitting"]["median_bias_dex"], 0.2, atol=1e-5)
        assert np.isclose(statistics["log_T_emitting"]["median_absolute_error_dex"], 0.1, atol=1e-5)
        assert statistics["cool_fraction"] == 0.0

    # Plasma colder than the emission cutoff is excluded from the emitting
    # statistics and reported as a cool fraction instead.
    cold = compare(query_model(_field(8.0, 5.0), points), reconstruction, radius, emitting_log_T=5.7)
    first = next(iter(cold.values()))
    assert first["log_ne_emitting"] is None and first["cool_fraction"] == 1.0
    assert first["log_ne_all"]["count"] > 0


def test_absorber_comparison_uses_one_equivalent_unit():
    from sunerf.evaluation.psi_euv_truth import compare_absorber

    radius = np.linspace(1.0, 1.5, 11)
    latitude = np.linspace(-1.0, 1.0, 5)
    truth = np.zeros((11, 5, 7))
    truth[:2] = 1.0e10  # limb layer below 1.02 R_sun
    reconstruction = 0.5 * truth

    report = compare_absorber(truth, reconstruction, radius, latitude)

    assert report["1.00-1.02"]["ratio"] == 0.5
    assert report["1.10-1.30"]["ratio"] is None  # nothing to recover there
