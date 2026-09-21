import copy
import numpy as np
import pytest
from datetime import datetime

from sunerf.data.psi.build_synthetic import (
    _load_response_for_synthesis,
    _response_renderer_config,
    _response_state_metadata,
    _synthetic_instrument_metadata,
    build_parser,
    build_synthetic_state,
)
from sunerf.response import RESPONSE_SCHEMA, ResponseArtifact
from sunerf.evaluation.loader import PlasmaSuNeRFLoader, SuNeRFLoader
from sunerf.model.spherical_grid import SphericalGridPlasmaModel
from types import SimpleNamespace
import torch


def _versioned_response(path):
    ResponseArtifact(
        channels=("B", "A"),
        log_temperature=np.array([5.0, 6.0]),
        response=np.array([[1.0, 2.0], [3.0, 4.0]]),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention="ne2",
        provenance={"builder": "test", "calibration_epoch": "2025-01-01T00:00:00Z"},
    ).save(path)


def test_direct_psi_requires_source_conventions_and_preserves_artifact_order(tmp_path):
    path = tmp_path / "response.npz"
    _versioned_response(path)
    args = build_parser().parse_args(
        [
            "--temperature-response-artifact", str(path),
            "--data-path", str(tmp_path / "grid"),
            "--out-path", str(tmp_path / "out"),
            "--source-density-scale-cm3", "1e8",
            "--source-temperature-scale-k", "2.807066716734894e7",
            "--reference-frame-id", "1813",
            "--longitude-frame", "carrington",
        ]
    )
    artifact = _load_response_for_synthesis(path, None)
    config = _response_renderer_config(artifact, path, Rs_per_ds=0.25)
    metadata = _response_state_metadata(artifact, path)

    assert args.density_unit_scale_cm3 == 1.0e8
    assert args.temperature_unit_scale_K == pytest.approx(2.807066716734894e7)
    assert args.reference_frame_id == 1813
    assert args.longitude_frame == "carrington"
    assert artifact.channels == ("B", "A")
    assert config["channels"] == ("B", "A")
    assert config["Rs_per_ds"] == 0.25
    assert config["artifact"] == str(path)
    assert metadata["schema"] == RESPONSE_SCHEMA
    assert metadata["channels"] == ("B", "A")
    assert metadata["response_unit"] == artifact.response_unit
    assert metadata["provenance"] == artifact.provenance

    instrument_metadata, data_metadata = _synthetic_instrument_metadata(
        artifact, path, "PSI"
    )
    channels = instrument_metadata["PSI"]["channels"]
    assert [channel["id"] for channel in channels] == ["B", "A"]
    assert [channel["response_channel_id"] for channel in channels] == ["B", "A"]
    assert [channel["cmap"] for channel in channels] == ["gray", "gray"]
    assert instrument_metadata["PSI"]["response"]["response_id"] == artifact.response_id
    assert instrument_metadata["PSI"]["response"]["sha256"]
    assert instrument_metadata["PSI"]["image_scaling"]["divisor"] == [1.0, 1.0]
    assert data_metadata["channel_ids"] == ("B", "A")
    assert data_metadata["response_ids"] == (artifact.response_id, artifact.response_id)

    # Exercise the artifact-driven loader routing without loading an executable
    # pickle: it must use these IDs rather than reconstructing gray 0..N labels.
    loader = object.__new__(SuNeRFLoader)
    loader.instrument_keys = ["PSI"]
    loader.config = {"PSI": {"instrument_key": "PSI", **data_metadata}}
    loader.rendering = SimpleNamespace(
        rendering_modules={"PSI": SimpleNamespace(temperature_response=torch.zeros(1, 2, 2))}
    )
    routed = loader._load_instrument_metadata(instrument_metadata)
    assert [channel["id"] for channel in routed["PSI"]["channels"]] == ["B", "A"]


def test_direct_psi_parser_rejects_implicit_source_conventions(tmp_path):
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--temperature-response-artifact", str(tmp_path / "response.npz"),
            "--data-path", str(tmp_path / "grid"),
            "--out-path", str(tmp_path / "out"),
        ])


def test_direct_psi_requires_explicit_composition_for_ne_nh_response(tmp_path):
    path = tmp_path / "response.npz"
    ResponseArtifact(
        channels=("A",),
        log_temperature=np.array([5.0, 6.0]),
        response=np.ones((1, 2)),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention="ne_nh",
        provenance={"builder": "test", "hydrogen_to_electron_ratio": 0.82},
    ).save(path)
    artifact = _load_response_for_synthesis(path, None)

    config = _response_renderer_config(artifact, path, Rs_per_ds=1.0)
    assert config["hydrogen_to_electron_ratio"] == 0.82
    matching_config = _response_renderer_config(
        artifact,
        path,
        Rs_per_ds=1.0,
        hydrogen_to_electron_ratio=0.82,
    )
    assert matching_config["hydrogen_to_electron_ratio"] == 0.82
    with pytest.raises(ValueError, match="does not match"):
        _response_renderer_config(
            artifact,
            path,
            Rs_per_ds=1.0,
            hydrogen_to_electron_ratio=0.83,
        )


def test_grid_artifact_is_weights_only_reconstructable_and_renders_finitely(
    tmp_path, monkeypatch
):
    response_path = tmp_path / "response.npz"
    _versioned_response(response_path)
    args = build_parser().parse_args([
        "--temperature-response-artifact", str(response_path),
        "--response-channels", "A",
        "--data-path", str(tmp_path / "grid"),
        "--out-path", str(tmp_path / "out"),
        "--instrument-key", "PSI",
        "--source-density-scale-cm3", "1e8",
        "--source-temperature-scale-k", "2.807066716734894e7",
        "--reference-frame-id", "1813",
        "--longitude-frame", "carrington",
        "--min-radius", "1.0",
        "--max-radius", "2.0",
        "--n-samples", "8",
        "--n-hierarchical-samples", "4",
        "--resolution", "4", "4",
    ])
    shape = (1, 2, 2, 2)
    model = SphericalGridPlasmaModel(
        log_density=np.full(shape, 8.0, dtype=np.float32),
        log_temperature=np.full(shape, 6.0, dtype=np.float32),
        time=np.array([0.0], dtype=np.float32),
        radius=np.array([1.0, 2.0], dtype=np.float32),
        latitude=np.array([-0.5 * np.pi, 0.5 * np.pi], dtype=np.float32),
        longitude=np.array([0.0, np.pi], dtype=np.float32),
        log_T=np.array([5.0, 6.0], dtype=np.float32),
    )
    fake_grid = SimpleNamespace(
        model=model,
        normalized_times=np.array([0.0], dtype=np.float32),
        observation_times=(datetime(2025, 1, 1),),
        ref_date=datetime(2025, 1, 1),
        source_provenance={
            "schema": "sunerf.psi_grid_source.v1",
            "source_root": str((tmp_path / "grid").resolve()),
            "frames": [{
                "frame_id": 1813,
                "density": {"path": "/input/rho001813.h5", "sha256": "a" * 64},
                "temperature": {"path": "/input/t001813.h5", "sha256": "b" * 64},
            }],
            "source_unit_conversion": {
                "density_unit_scale_cm3": 1.0e8,
                "temperature_unit_scale_K": 2.807066716734894e7,
                "density_output_unit": "cm-3",
                "temperature_output_unit": "K",
            },
            "temporal_reference": {
                "reference_frame_id": 1813,
                "reference_date": "2025-01-01T00:00:00",
                "time_step_seconds": 3600.0,
            },
            "coordinate_convention": {
                "radius_unit": "solar-radius",
                "latitude_unit": "rad",
                "longitude_unit": "rad",
                "longitude_frame": "carrington",
            },
        },
    )
    monkeypatch.setattr(
        "sunerf.data.psi.build_synthetic.load_psi_grid",
        lambda *unused_args, **unused_kwargs: fake_grid,
    )

    state = build_synthetic_state(args)
    artifact_path = tmp_path / "grid.snf"
    torch.save(state, artifact_path)
    restricted = torch.load(artifact_path, map_location="cpu", weights_only=True)

    assert restricted["artifact_type"] == "sunerf.plasma.grid"
    assert restricted["artifact_security"] == "weights_only"
    assert "rendering" not in restricted
    assert "values" in restricted["grid_state_dict"]
    assert restricted["synthetic_grid"]["frames"][0]["density"]["sha256"] == "a" * 64
    assert restricted["provenance"]["psi_grid_source"]["frames"][0][
        "temperature"
    ]["sha256"] == "b" * 64

    for suffix in ("temperature_response", "instrument_scaling_center"):
        state_key = next(
            key for key in restricted["rendering_state_dict"]
            if key.endswith(suffix)
        )
        tampered = copy.deepcopy(restricted)
        tampered["rendering_state_dict"][state_key] = (
            tampered["rendering_state_dict"][state_key] + 1
        )
        tampered_path = tmp_path / f"tampered-{suffix}.snf"
        torch.save(tampered, tampered_path)
        with pytest.raises(ValueError, match="immutable physics state"):
            PlasmaSuNeRFLoader(tampered_path, device="cpu")

    loader = PlasmaSuNeRFLoader(artifact_path, device="cpu")
    batch = {
        "PSI": {
            "rays": torch.tensor([[[0.0, 0.0, 215.0], [0.0, 0.0, -1.0]]]),
            "time": torch.zeros((1, 1)),
            "instrument": "PSI",
        }
    }
    rendered = loader.rendering(batch, shuffle=False)["model_out"]["PSI"]["image"]

    assert loader.loaded_safely is True
    assert torch.isfinite(rendered).all()
    assert torch.any(rendered > 0)
