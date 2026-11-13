from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import torch

from sunerf.data.loader.insitu import InSituDataset
from sunerf.data.psi import synthetic_insitu
from sunerf.data.psi.density_cube import PSIDensityCube, PSIFieldCube


def _mesh():
    radius = np.linspace(1.0, 30.0, 30)
    theta = np.linspace(0.0, np.pi, 19)
    phi = np.linspace(0.0, 2.0 * np.pi, 37)[:-1]
    return radius, theta, phi


def _field(radius, theta, phi, function):
    p, t, r = np.meshgrid(phi, theta, radius, indexing="ij")
    return function(r, t, p)


def test_field_cube_interpolates_linear_fields_and_vanishes_outside():
    radius, theta, phi = _mesh()
    cube = PSIFieldCube(_field(radius, theta, phi, lambda r, t, p: 10.0 * r), radius, theta, phi, unit_scale=2.0)
    r = torch.tensor([5.5, 12.25, 31.0])
    value = cube(r, torch.full_like(r, 1.0), torch.full_like(r, 0.3))
    np.testing.assert_allclose(value.numpy(), [110.0, 245.0, 0.0], rtol=1e-5)


def test_sample_cubes_adds_corotation_to_tangential_velocity():
    radius, theta, phi = _mesh()
    density = PSIDensityCube(_field(radius, theta, phi, lambda r, t, p: 1.0 / r ** 2), radius, theta, phi)
    velocity = {
        "vr": PSIFieldCube(_field(radius, theta, phi, lambda r, t, p: 300.0 + 0 * r), radius, theta, phi),
        "vt": PSIFieldCube(_field(radius, theta, phi, lambda r, t, p: 5.0 + 0 * r), radius, theta, phi),
        "vp": PSIFieldCube(_field(radius, theta, phi, lambda r, t, p: -20.0 + 0 * r), radius, theta, phi),
    }
    r = np.array([10.0])
    n_e, rtn = synthetic_insitu.sample_cubes(density, velocity, r, np.array([np.pi / 2]), np.array([1.0]), "cpu")
    np.testing.assert_allclose(n_e, [1e6], rtol=1e-4)
    corotation = synthetic_insitu.CARRINGTON_OMEGA_RAD_S * 10.0 * synthetic_insitu.KM_PER_RSUN
    np.testing.assert_allclose(rtn, [[300.0, -20.0 + corotation, -5.0]], rtol=1e-5)
    assert 19.0 < corotation < 21.0


def test_sample_cubes_without_velocity_cubes_gives_missing_velocity():
    radius, theta, phi = _mesh()
    density = PSIDensityCube(_field(radius, theta, phi, lambda r, t, p: 1.0 + 0 * r), radius, theta, phi)
    _, rtn = synthetic_insitu.sample_cubes(density, None, np.array([3.0]), np.array([1.0]), np.array([1.0]), "cpu")
    assert np.all(np.isnan(rtn))


def _samples(n=48, with_velocity=True):
    start = datetime(2021, 10, 28, 15, 30)
    times = np.array([start + timedelta(minutes=5 * i) for i in range(n)])
    angle = np.linspace(0.0, 0.5, n)
    position = np.stack([15.0 * np.cos(angle), 15.0 * np.sin(angle), -0.5 * np.ones(n)], axis=-1)
    velocity = np.tile([400.0, 10.0, -5.0], (n, 1)) if with_velocity else np.full((n, 3), np.nan)
    return {
        "time": times, "dump": np.arange(1, n + 1), "position": position,
        "density": np.full(n, 2.0e4), "velocity": velocity,
        "theta": np.full(n, 1.6), "phi": angle,
    }


def test_degraded_product_keeps_truth_statistics_and_opens_gaps():
    truth = synthetic_insitu.clean_arrays(_samples(), "probe")
    args = SimpleNamespace(density_noise=0.15, velocity_noise_kms=25.0, gap_fraction=0.5,
                           gap_duration=timedelta(minutes=30))
    noisy = synthetic_insitu.degrade_arrays(truth, args, np.random.default_rng(1))

    assert np.all(truth["has_velocity"] == 1.0)
    gaps = noisy["has_velocity"] == 0.0
    assert 0 < gaps.sum() < gaps.size
    assert np.all(np.isnan(noisy["velocity_radial_kms"][gaps]))
    # Gaps are whole 30-min windows (6 samples at 5 min).
    assert all(len(set(window)) == 1 for window in gaps.reshape(-1, 6))
    log_ratio = np.log(noisy["density_cm3"] / truth["density_cm3"])
    assert 0.08 < log_ratio.std() < 0.25
    assert abs(np.nanmean(noisy["velocity_radial_kms"]) - 400.0) < 15.0
    np.testing.assert_array_equal(noisy["time_unix"], truth["time_unix"])


def test_product_loads_through_insitu_dataset(tmp_path):
    truth = synthetic_insitu.clean_arrays(_samples(with_velocity=False), "probe")
    path = tmp_path / "probe.npz"
    synthetic_insitu.write_product(path, truth, {"n_valid": 48})
    assert (tmp_path / "probe.json").exists()

    dataset = InSituDataset(str(path), Rs_per_ds=15.0, seconds_per_dt=86400.0,
                            ref_date=datetime(2021, 10, 8, 15, 30), work_directory=str(tmp_path),
                            drho_cm3=6000.0, batch_size=16, max_distance=30.2)
    batch = dataset[0]
    assert batch["query_points"].shape[-1] == 4
    np.testing.assert_allclose(batch["density"].numpy().mean(), 2.0e4 / 6000.0, rtol=1e-5)
    assert np.all(batch["has_velocity"].numpy() == 0.0)
    assert np.all(batch["has_density"].numpy() == 1.0)
