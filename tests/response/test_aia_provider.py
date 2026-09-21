from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from sunerf.response.builder import (
    fold_temperature_response,
    load_instrument_throughput,
    load_spectral_emissivity,
)
from sunerf.response.providers import aia
from sunerf.response.providers.base import OptionalProviderDependencyError


GOLDEN_PEAKS = {
    "A94": (4.985879255997859e-12, 93.9000015258789),
    "A131": (1.8321672254752452e-11, 130.3000030517578),
    "A171": (2.6869977289615423e-11, 170.8000030517578),
    "A193": (1.831015904335711e-11, 193.3000030517578),
    "A211": (1.0265276902366597e-11, 210.5),
    # Genuine A131 crosstalk: equal effective area to the 335 A peak, but
    # 2.6 times the detector gain.
    "A335": (7.263628734821808e-13, 130.39999389648438),
}
A335_PRIMARY_PEAK = (2.771189e-13, 335.5)

requires_aia_assets = pytest.mark.skipif(
    "SUNERF_AIA_CALIBRATION_DIR" not in os.environ,
    reason="requires explicitly supplied, byte-pinned AIA calibration assets",
)


def test_aia_source_pins_are_immutable_and_complete():
    commit = "ef466df3faa4699ee1c8dec6b26f8449c8e41865"
    immutable_root = (
        "https://raw.githubusercontent.com/LM-SAL/backup-files/"
        f"{commit}/static/sdo/aia/response/"
    )
    assert aia.AIA_INSTRUMENT_SOURCE.url == (
        f"{immutable_root}aia_V8_all_fullinst.genx"
    )
    assert aia.AIA_INSTRUMENT_SOURCE.size_bytes == 5_426_568
    assert aia.AIA_INSTRUMENT_SOURCE.sha256 == (
        "3940648e6b02876c45a9893f40806bbcc50baa994ae3fa2d95148916988426dd"
    )
    assert aia.AIA_CORRECTION_SOURCE.url == (
        f"{immutable_root}aia_V10_20201119_190000_response_table.txt"
    )
    assert aia.AIA_CORRECTION_SOURCE.size_bytes == 17_400
    assert aia.AIA_CORRECTION_SOURCE.sha256 == (
        "0a3f2db39d05c44185f6fdeec928089fb55d1ce1e0a805145050c6356cbc6e98"
    )
    assert aia.AIA_THROUGHPUT_SOURCES == (
        aia.AIA_INSTRUMENT_SOURCE,
        aia.AIA_CORRECTION_SOURCE,
    )


def test_fetch_aia_sources_downloads_only_fold_inputs(monkeypatch, tmp_path):
    calls = {}

    def fake_fetch(sources, destination, *, force, timeout_seconds):
        calls["fetch"] = (sources, Path(destination), force, timeout_seconds)
        return {source.key: Path(destination) / source.filename for source in sources}

    monkeypatch.setattr(aia, "fetch_sources", fake_fetch)

    paths = aia.fetch_aia_sources(
        tmp_path,
        force=True,
        timeout_seconds=9.0,
    )

    expected = aia.AIA_THROUGHPUT_SOURCES
    assert calls["fetch"] == (expected, tmp_path, True, 9.0)
    assert tuple(paths) == tuple(source.key for source in expected)


def test_missing_aiapy_error_is_actionable(monkeypatch):
    monkeypatch.setattr(aia, "sys", SimpleNamespace(version_info=(3, 12, 0)))

    def missing(_package):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(aia.importlib.metadata, "version", missing)
    with pytest.raises(OptionalProviderDependencyError, match=r"aiapy==0\.12\.1"):
        aia._load_aiapy()


def test_export_aia_throughput_uses_only_pinned_local_inputs(monkeypatch, tmp_path):
    instrument_path = tmp_path / aia.AIA_INSTRUMENT_SOURCE.filename
    correction_path = tmp_path / aia.AIA_CORRECTION_SOURCE.filename
    source_paths = {
        "instrument": instrument_path,
        "correction_table": correction_path,
    }
    calls = {"channels": [], "responses": []}
    correction_table = object()
    grid = np.array([25.0, 100.0, 900.0]) * u.AA
    solid_angle = 8.461580394691914e-12 * u.sr / u.pix

    class FakeChannel:
        wavelength = grid
        plate_scale = solid_angle

        def __init__(self, wavelength, *, instrument_file):
            self.nominal = int(wavelength.to_value(u.AA))
            calls["channels"].append((self.nominal, Path(instrument_file)))

        def wavelength_response(self, **kwargs):
            calls["responses"].append((self.nominal, kwargs))
            scale = aia.AIA_CHANNEL_WAVELENGTHS.index(self.nominal) + 1
            return scale * np.array([0.25, 1.0, 0.5]) * u.cm**2 * u.DN / u.ph

    def fake_correction_table(path):
        calls["correction_path"] = Path(path)
        return correction_table

    monkeypatch.setattr(
        aia,
        "require_source_paths",
        lambda sources, directory: source_paths,
    )
    monkeypatch.setattr(
        aia,
        "_load_aiapy",
        lambda: (FakeChannel, fake_correction_table),
    )

    output = tmp_path / "aia.throughput.npz"
    artifact = aia.export_aia_throughput(tmp_path, output)

    assert output.is_file()
    assert artifact.channels == aia.AIA_CHANNELS
    # The 900 A node lies beyond the SolarSoft-validated fold range.
    np.testing.assert_array_equal(artifact.wavelength_angstrom, grid.value[:2])
    expected = np.stack(
        [
            scale * np.array([0.25, 1.0]) * solid_angle.value
            for scale in range(1, 7)
        ]
    )
    np.testing.assert_allclose(artifact.throughput, expected, rtol=0, atol=0)
    assert u.Unit(artifact.throughput_unit) == (
        u.cm**2 * u.DN * u.sr / (u.ph * u.pix)
    )
    assert artifact.calibration_epoch == aia.AIA_REFERENCE_EPOCH

    assert calls["correction_path"] == correction_path
    assert calls["channels"] == [
        (wavelength, instrument_path)
        for wavelength in aia.AIA_CHANNEL_WAVELENGTHS
    ]
    for nominal, settings in calls["responses"]:
        assert nominal in aia.AIA_CHANNEL_WAVELENGTHS
        assert settings["obstime"].utc.isot == "2010-03-24T00:00:00.000"
        assert settings["include_eve_correction"] is True
        assert settings["include_crosstalk"] is True
        assert settings["correction_table"] is correction_table
        assert settings["calibration_version"] == 10

    provenance = artifact.provenance
    assert provenance["sensitivity_convention"] == "reference_epoch"
    assert provenance["calibration"]["sha256"] == (
        aia.AIA_INSTRUMENT_SOURCE.sha256
    )
    assert provenance["degradation"]["sha256"] == (
        aia.AIA_CORRECTION_SOURCE.sha256
    )
    assert provenance["degradation"]["reference_epoch"] == (
        aia.AIA_REFERENCE_EPOCH
    )
    assert provenance["eve_normalization"]["absolute_reference_date"] == (
        aia.AIA_EVE_ABSOLUTE_REFERENCE_DATE
    )
    assert provenance["crosstalk"]["included"] is True
    assert provenance["wavelength_grid"]["maximum_angstrom"] == 100.0
    assert provenance["wavelength_grid"]["source_maximum_angstrom"] == 900.0
    assert provenance["radiometry"]["native_pixel_solid_angle_sr"] == (
        solid_angle.value
    )

    reloaded = load_instrument_throughput(output)
    assert reloaded.content_sha256 == artifact.content_sha256


@requires_aia_assets
def test_aia_official_asset_golden_peaks(tmp_path):
    artifact = aia.export_aia_throughput(
        os.environ["SUNERF_AIA_CALIBRATION_DIR"],
        tmp_path / "aia-golden.throughput.npz",
    )
    assert artifact.throughput.shape == (6, 3_881)
    assert artifact.wavelength_angstrom[0] == 25.0
    assert artifact.wavelength_angstrom[-1] == pytest.approx(
        aia.AIA_VALIDATED_WAVELENGTH_MAX_ANGSTROM, abs=0.05
    )
    for channel, row in zip(artifact.channels, artifact.throughput):
        peak_index = int(np.argmax(row))
        expected_value, expected_wavelength = GOLDEN_PEAKS[channel]
        assert row[peak_index] == pytest.approx(expected_value, rel=2e-14)
        assert artifact.wavelength_angstrom[peak_index] == expected_wavelength


@requires_aia_assets
def test_aia_throughput_peaks_at_nominal_wavelengths(tmp_path):
    artifact = aia.export_aia_throughput(
        os.environ["SUNERF_AIA_CALIBRATION_DIR"],
        tmp_path / "aia-peaks.throughput.npz",
    )
    wavelength = artifact.wavelength_angstrom
    for nominal, row in zip(aia.AIA_CHANNEL_WAVELENGTHS, artifact.throughput):
        # Only A335 carries crosstalk strong enough to outrank its own band.
        band = wavelength > 150.0 if nominal == 335 else np.ones_like(row, bool)
        peak = wavelength[band][np.argmax(row[band])]
        assert abs(peak - nominal) < 1.5
        in_band = np.abs(wavelength - nominal) < 25.0
        crosstalk = (np.abs(wavelength - 131.0) < 25.0) & (nominal == 335)
        total = np.trapezoid(row, wavelength)
        explained = np.trapezoid(row * (in_band | crosstalk), wavelength)
        assert explained / total > 0.9

    a335 = artifact.throughput[artifact.channels.index("A335")]
    primary = wavelength > 150.0
    expected_value, expected_wavelength = A335_PRIMARY_PEAK
    assert a335[primary].max() == pytest.approx(expected_value, rel=1e-6)
    assert wavelength[primary][np.argmax(a335[primary])] == expected_wavelength


@requires_aia_assets
def test_a335_response_peaks_at_fe_xvi(tmp_path):
    artifact = aia.export_aia_throughput(
        os.environ["SUNERF_AIA_CALIBRATION_DIR"],
        tmp_path / "aia-response.throughput.npz",
    )
    spectral = load_spectral_emissivity(
        "builtin:spectral/chianti_coronal_2021.spectral.npz"
    )
    response = fold_temperature_response(spectral, artifact, channels=["A335"])
    density = int(np.argmin(np.abs(response.log_density - 9.0)))
    curve = response.response[0, density]
    peak_log_temperature = response.log_temperature[np.argmax(curve)]
    assert 6.2 <= peak_log_temperature <= 6.6
    assert curve.max() == pytest.approx(3.8e-27, rel=0.05)
