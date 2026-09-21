from __future__ import annotations

from pathlib import Path

import numpy as np

from sunerf.response.builder import load_instrument_throughput
from sunerf.response.providers import secchi


def test_secchi_sra_sources_and_default_mode_are_pinned():
    assert secchi.EUVI_SOURCES["A"].sha256 == (
        "0883e6048088520abc0bed8924df01b141661ba7e63b328ef1e5701b0c43ce9f"
    )
    assert secchi.EUVI_SOURCES["A"].size_bytes == 2_677_664
    assert secchi.EUVI_SOURCES["B"].sha256 == (
        "7dd3920dd09102e230405d8e6904fb4cbe0773fa044351cbcc30f0a302063333"
    )
    assert secchi.EUVI_SOURCES["B"].size_bytes == 2_677_648
    modes = secchi.available_euvi_modes()
    assert modes["default_channels"] == (171, 195, 284)
    assert modes["default_filter"] == "S1"
    assert modes["recommended_secchi_prep_keyword"] == "/NORMAL_OFF"


def test_export_matches_secchi_prep_photonflux_formula(tmp_path, monkeypatch):
    wavelength = np.array([100.0, 200.0, 400.0])
    area = np.ones((4, 4, wavelength.size))
    area[0, 1] = [1.0, 2.0, 3.0]
    response = secchi.EUVISpectralResponseArea(
        spacecraft="A",
        wavelength_angstrom=wavelength,
        area_cm2=area,
        version="001",
        release_date="test",
    )
    source_path = tmp_path / secchi.EUVI_SOURCES["A"].filename
    monkeypatch.setattr(
        secchi,
        "require_source_paths",
        lambda sources, directory: {sources[0].key: source_path},
    )
    monkeypatch.setattr(secchi, "load_euvi_sra", lambda path, spacecraft: response)

    output = tmp_path / "euvi.throughput.npz"
    artifact = secchi.export_secchi_euvi_throughput(
        tmp_path,
        output,
        spacecraft="A",
        channels=(171,),
    )
    expected = (
        area[0, 1]
        * 171.0
        / wavelength
        * secchi.NATIVE_PIXEL_SOLID_ANGLE_SR["A"]
    )
    np.testing.assert_allclose(artifact.throughput[0], expected, rtol=0, atol=0)
    assert artifact.channels == ("171",)
    assert artifact.provenance["sensitivity_convention"] == "static_assumed"
    assert artifact.provenance["radiometry"]["secchi_prep_output"] == "PhotonFlux"
    assert artifact.provenance["radiometry"]["secchi_prep_normal_off"] is True
    assert artifact.provenance["radiometry"]["filter_by_channel"] == {"171": "S1"}
    assert load_instrument_throughput(output).content_sha256 == artifact.content_sha256


def test_open_normalized_export_divides_documented_scalar(tmp_path, monkeypatch):
    wavelength = np.array([100.0, 200.0])
    response = secchi.EUVISpectralResponseArea(
        spacecraft="A",
        wavelength_angstrom=wavelength,
        area_cm2=np.ones((4, 4, wavelength.size)),
        version="001",
        release_date="test",
    )
    monkeypatch.setattr(
        secchi,
        "require_source_paths",
        lambda sources, directory: {sources[0].key: Path(directory) / sources[0].filename},
    )
    monkeypatch.setattr(secchi, "load_euvi_sra", lambda path, spacecraft: response)
    normal_off = secchi.export_secchi_euvi_throughput(
        tmp_path,
        None,
        spacecraft="A",
        channels=(195,),
    )
    normalized = secchi.export_secchi_euvi_throughput(
        tmp_path,
        None,
        spacecraft="A",
        channels=(195,),
        secchi_prep_normalized_to_open=True,
    )
    np.testing.assert_allclose(normalized.throughput, normal_off.throughput / 0.49)
    assert normalized.provenance["radiometry"]["secchi_prep_normal_off"] is False
