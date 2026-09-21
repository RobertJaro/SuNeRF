from pathlib import Path

import pytest

from sunerf.response import pipeline


def test_workflow_paths_are_small_and_deterministic(tmp_path):
    paths = pipeline.pipeline_paths(tmp_path)
    assert paths["aia_sources"] == tmp_path / "instruments" / "aia"
    assert paths["throughputs"] == tmp_path / "throughputs"
    assert pipeline.resolve_throughput_paths(tmp_path) == {
        key: tmp_path / "throughputs" / filename
        for key, filename in pipeline.THROUGHPUT_FILENAMES.items()
    }
    assert pipeline.resolve_throughput_paths(tmp_path, ("aia", "euvi_a")) == {
        "aia": tmp_path / "throughputs" / "aia.throughput.npz",
        "euvi_a": tmp_path / "throughputs" / "euvi_a.throughput.npz",
    }


def test_prepare_downloads_then_exports(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        pipeline,
        "fetch_instrument_inputs",
        lambda root, *, instruments, force: calls.append(
            ("fetch", Path(root), instruments, force)
        ),
    )
    products = {"aia": tmp_path / "aia.npz"}
    monkeypatch.setattr(
        pipeline,
        "export_throughputs",
        lambda root, *, instruments: calls.append(
            ("export", Path(root), instruments)
        ) or products,
    )

    selected = ("aia", "euvi_a")
    assert pipeline.prepare_instruments(
        tmp_path, instruments=selected, force=True
    ) == products
    assert calls == [
        ("fetch", tmp_path, selected, True),
        ("export", tmp_path, selected),
    ]


def test_uniform_build_loads_atomic_grid_once_and_uses_one_fold(monkeypatch, tmp_path):
    spectral = object()
    throughputs = {"aia": object(), "eui_fsi": object()}
    calls = []

    class Artifact:
        def __init__(self, instrument):
            self.instrument = instrument

        def save(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.instrument, encoding="ascii")

    monkeypatch.setattr(
        pipeline,
        "load_spectral_emissivity",
        lambda path: calls.append(("spectral", Path(path))) or spectral,
    )

    def load_throughput(path):
        instrument = Path(path).name.split(".", 1)[0]
        value = throughputs[instrument]
        calls.append(("throughput", instrument))
        return value

    monkeypatch.setattr(pipeline, "load_instrument_throughput", load_throughput)

    def fold(received_spectral, received_throughput):
        assert received_spectral is spectral
        instrument = next(
            key for key, value in throughputs.items() if value is received_throughput
        )
        calls.append(("fold", instrument))
        return Artifact(instrument)

    monkeypatch.setattr(pipeline, "fold_temperature_response", fold)
    products = pipeline.build_responses(
        tmp_path / "atomic.npz",
        root=tmp_path,
        output_dir=tmp_path / "out",
        label="test",
        instruments=("aia", "eui_fsi"),
    )

    assert calls == [
        ("spectral", tmp_path / "atomic.npz"),
        ("throughput", "aia"),
        ("fold", "aia"),
        ("throughput", "eui_fsi"),
        ("fold", "eui_fsi"),
    ]
    assert products == {
        "aia": tmp_path / "out" / "aia_test.sunerf.npz",
        "eui_fsi": tmp_path / "out" / "eui_fsi_test.sunerf.npz",
    }
    assert all(path.is_file() for path in products.values())


def test_uniform_build_rejects_unsafe_label(tmp_path):
    with pytest.raises(ValueError, match="filename-safe"):
        pipeline.build_responses(
            tmp_path / "atomic.npz",
            root=tmp_path,
            label="bad/label",
        )
