from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from sunerf.data.psi.spherical_grid import load_psi_grid, pair_psi_files
from sunerf.train.coordinate_transformation import spherical_to_cartesian


SOURCE_TEMPERATURE_SCALE_K = 2.807066716734894e7


def _write_psi_frame(root, frame_id, density):
    radius = np.array([1.1, 2.0], dtype=np.float32)
    colatitude = np.array([0.5 * np.pi - 0.5, 0.5 * np.pi + 0.5], dtype=np.float32)
    longitude = np.array([0.0, np.pi, 2.0 * np.pi], dtype=np.float32)
    shape = (radius.size, colatitude.size, longitude.size)
    values = {
        "rho": np.full(shape, density, dtype=np.float32),
        "t": np.full(shape, 1.0e6 / SOURCE_TEMPERATURE_SCALE_K, dtype=np.float32),
    }
    for field, data in values.items():
        path = Path(root) / field / f"{field}{frame_id:06d}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle["dim1"] = radius
            handle["dim2"] = colatitude
            handle["dim3"] = longitude
            handle["Data"] = data.T


def test_pair_psi_files_rejects_unpaired_frames(tmp_path):
    _write_psi_frame(tmp_path, 1813, density=1.0)
    (tmp_path / "t" / "t001813.h5").rename(tmp_path / "t" / "t001814.h5")

    with pytest.raises(ValueError, match="Unpaired PSI plasma frames"):
        pair_psi_files(tmp_path)


def test_load_psi_grid_interpolates_frames_and_drops_periodic_endpoint(tmp_path):
    _write_psi_frame(tmp_path, 1813, density=1.0)
    _write_psi_frame(tmp_path, 1815, density=100.0)
    grid = load_psi_grid(
        tmp_path,
        log_T=np.array([5.0, 6.0, 7.0], dtype=np.float32),
        density_unit_scale_cm3=1.0,
        temperature_unit_scale_K=SOURCE_TEMPERATURE_SCALE_K,
        reference_frame_id=1813,
        longitude_frame="carrington",
        seconds_per_dt=1.0,
        time_step_seconds=1.0,
        min_radius=1.0,
        max_radius=2.1,
    )

    assert grid.model.longitude.numel() == 2
    spherical = torch.tensor([1.5, 0.0, 0.25, 1.0], dtype=torch.float32)
    query = torch.cat([spherical_to_cartesian(spherical[:3], torch), spherical[3:]])
    output = grid.model(query[None])

    torch.testing.assert_close(output["total_log_ne"], torch.tensor([[1.0]]), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output["mean_log_T"], torch.tensor([[6.0]]), atol=1e-5, rtol=1e-5)
    provenance = grid.source_provenance
    assert provenance["schema"] == "sunerf.psi_grid_source.v1"
    assert provenance["source_unit_conversion"] == {
        "density_unit_scale_cm3": 1.0,
        "temperature_unit_scale_K": SOURCE_TEMPERATURE_SCALE_K,
        "density_output_unit": "cm-3",
        "temperature_output_unit": "K",
    }
    assert provenance["temporal_reference"]["reference_frame_id"] == 1813
    assert provenance["coordinate_convention"]["longitude_frame"] == "carrington"
    assert [frame["frame_id"] for frame in provenance["frames"]] == [1813, 1815]
    for frame in provenance["frames"]:
        assert len(frame["density"]["sha256"]) == 64
        assert len(frame["temperature"]["sha256"]) == 64
        assert Path(frame["density"]["path"]).is_absolute()
        assert Path(frame["temperature"]["path"]).is_absolute()


def test_load_psi_grid_requires_valid_explicit_source_conventions(tmp_path):
    _write_psi_frame(tmp_path, 1813, density=1.0)
    common = {
        "log_T": np.array([5.0, 6.0, 7.0], dtype=np.float32),
        "density_unit_scale_cm3": 1.0,
        "temperature_unit_scale_K": SOURCE_TEMPERATURE_SCALE_K,
        "reference_frame_id": 1813,
        "longitude_frame": "carrington",
    }
    with pytest.raises(ValueError, match="density_unit_scale_cm3"):
        load_psi_grid(tmp_path, **{**common, "density_unit_scale_cm3": 0.0})
    with pytest.raises(ValueError, match="longitude_frame"):
        load_psi_grid(tmp_path, **{**common, "longitude_frame": "unknown"})
