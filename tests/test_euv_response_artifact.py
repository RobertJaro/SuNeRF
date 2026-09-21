import numpy as np
import pytest

from sunerf.response import ResponseArtifact, load_response_artifact


def _artifact(**overrides):
    values = {
        "channels": ("A", "B"),
        "log_temperature": np.array([5.0, 6.0, 7.0]),
        "response": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "response_unit": "cm5 DN s-1 pix-1",
        "emission_measure_convention": "ne2",
        "provenance": {"builder": "unit-test", "atomic_database": "test"},
    }
    values.update(overrides)
    return ResponseArtifact(**values)


def test_versioned_response_round_trip_preserves_order_and_metadata(tmp_path):
    path = tmp_path / "response.sunerf.npz"
    source = _artifact()
    source.save(path)

    loaded = load_response_artifact(path, channels=("B", "A"))

    assert loaded.channels == ("B", "A")
    assert loaded.response_unit == source.response_unit
    assert loaded.emission_measure_convention == "ne2"
    assert {
        key: value for key, value in loaded.provenance.items() if key != "response_id"
    } == {
        key: value for key, value in source.provenance.items() if key != "response_id"
    }
    assert loaded.response_id != source.response_id
    np.testing.assert_allclose(loaded.response, source.response[[1, 0]])


def test_response_channel_aliases_resolve_to_immutable_artifact_ids():
    artifact = _artifact(
        channels=('A94', 'A171'),
        response=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )

    selected = artifact.select_channels((171, 94))

    assert selected.channels == ('A171', 'A94')
    np.testing.assert_allclose(selected.response, artifact.response[[1, 0]])


def test_response_id_is_stable_and_sensitive_to_scientific_content(tmp_path):
    first = _artifact()
    second = _artifact()
    changed = first.updated(response=first.response + 1.0)

    assert first.response_id == second.response_id
    assert first.response_id.startswith("sha256:")
    assert first.verify_response_id()
    assert changed.response_id != first.response_id

    path = tmp_path / "response.npz"
    tampered_path = tmp_path / "tampered.npz"
    first.save(path)
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["response"] = payload["response"].copy()
    payload["response"][0, 0] += 1.0
    np.savez_compressed(tampered_path, **payload)

    with pytest.raises(ValueError, match="response_id verification failed"):
        load_response_artifact(tampered_path)


def test_response_artifact_write_is_atomic_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "response.npz"
    artifact = _artifact()
    artifact.save(path)
    original_bytes = path.read_bytes()

    def fail_after_partial_write(stream, **arrays):
        stream.write(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(np, "savez_compressed", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        artifact.save(path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".response.npz.*.tmp")) == []


def test_log_temperature_interpolation_is_zero_outside_source_support(tmp_path):
    path = tmp_path / "response.npz"
    _artifact(channels=("A",), response=np.array([[1.0, 3.0, 5.0]])).save(path)

    prepared = load_response_artifact(path, channels=("A",)).interpolate_temperature(
        np.array([4.0, 5.5, 7.5])
    )

    np.testing.assert_allclose(prepared.response, [[0.0, 2.0, 0.0]])


def test_unversioned_response_archive_is_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        temperature=10.0 ** np.array([5.0, 6.0, 7.0]),
        A=np.array([0.0, 2.0, 0.0]),
        B=np.array([0.0, 4.0, 0.0]),
    )

    with pytest.raises(ValueError, match="not a versioned SuNeRF"):
        load_response_artifact(path, channels=("B",))


def test_density_dependent_response_shape_is_validated():
    artifact = _artifact(
        log_density=np.array([8.0, 10.0]),
        response=np.ones((2, 2, 3)),
    )
    assert artifact.response.shape == (2, 2, 3)

    with pytest.raises(ValueError, match="response must have shape"):
        _artifact(log_density=np.array([8.0, 10.0]))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"channels": ("A", "A")}, "unique"),
        ({"log_temperature": np.array([5.0, 5.0, 7.0])}, "strictly increasing"),
        ({"response": np.array([[1.0, -1.0, 2.0], [1.0, 2.0, 3.0]])}, "non-negative"),
        ({"response_unit": "definitely not a unit ???"}, "invalid response_unit"),
        ({"response_unit": "1"}, "physical response_unit"),
        ({"provenance": {}}, "provenance"),
    ],
)
def test_response_validation_rejects_ambiguous_artifacts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _artifact(**overrides)
