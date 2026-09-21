import hashlib
from pathlib import Path

import numpy as np
import pytest

from sunerf.response.builder import load_instrument_throughput
from sunerf.response.providers import eui
from sunerf.response.providers.base import SourceFile


EXPECTED_SOURCE_PINS = {
    "fsi304_n4": (
        "FSI_Response_Merged_R0_AlMg4_V2.txt",
        21_957,
        "66f04d9291b8ae3fce6ab29b707c661511cfa15b614aa727c1818ef64ad83a7c",
    ),
    "fsi174_n25": (
        "FSI_Response_Merged_R0_AlZr25_V2.txt",
        21_958,
        "785ef1859bff35c788fae2be87caafa743b204d86e0d259abf65efb9fd3fc7ff",
    ),
    "fsi304_n26": (
        "FSI_Response_Merged_R0_AlMg26_V2.txt",
        21_958,
        "98fe2530e2ef9e54fd85bff8f5e826f859b1d3d93835f3dd90cddf8088ede386",
    ),
    "fsi174_n13": (
        "FSI_Response_Merged_R0_AlZr13_V2.txt",
        21_958,
        "4669b35606830964d5575c96f93e286a6f4e9780a6fbe5a65781c94e1ff5c5fd",
    ),
}

EXPECTED_FILTER_MODES = {
    "fsi304_n4": (304, "Magnesium_304_n4", 0),
    "fsi174_n25": (174, "Zirconium_174_n25", 50),
    "fsi304_n26": (304, "Magnesium_304_n26", 100),
    "fsi174_n13": (174, "Zirconium_174_n13", 150),
}


def _table_payload(
    mode,
    *,
    wavelength_nm=None,
    response=None,
    header=None,
) -> bytes:
    if wavelength_nm is None:
        wavelength_nm = eui.EUI_FSI_WAVELENGTH_NM
    if response is None:
        response = np.linspace(1e-22, 1e-12, wavelength_nm.size)
    if header is None:
        header = eui._expected_header(mode)
    lines = list(header)
    lines.extend(
        f"{wavelength:.4e} {value:.4e}"
        for wavelength, value in zip(wavelength_nm, response)
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def _write_test_sources(tmp_path, monkeypatch, mode_names):
    replacements = dict(eui.EUI_FSI_SOURCES)
    for mode_name in mode_names:
        mode = eui.EUI_FSI_FILTER_MODES[mode_name]
        original = eui.EUI_FSI_SOURCES[mode.source_key]
        payload = _table_payload(mode)
        (tmp_path / original.filename).write_bytes(payload)
        replacements[mode.source_key] = SourceFile(
            key=mode.source_key,
            filename=original.filename,
            url=f"https://example.invalid/{original.filename}",
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            version=original.version,
            description="test response table",
        )
    monkeypatch.setattr(eui, "EUI_FSI_SOURCES", replacements)


def test_public_calibration_sources_are_pinned_at_introduction_commit():
    assert eui.EUI_CALIBRATION_COMMIT == (
        "3619e9199ccce8421aa4c8b22c655741a4b8ab1c"
    )
    assert eui.EUI_CALIBRATION_SNAPSHOT == "2023-12-07"
    assert set(eui.EUI_FSI_SOURCES) == set(EXPECTED_SOURCE_PINS)
    for key, (filename, size_bytes, sha256) in EXPECTED_SOURCE_PINS.items():
        source = eui.EUI_FSI_SOURCES[key]
        assert (source.filename, source.size_bytes, source.sha256) == (
            filename,
            size_bytes,
            sha256,
        )
        assert eui.EUI_CALIBRATION_COMMIT in source.url
        assert "/-/raw/" in source.url


def test_all_filter_positions_are_exposed_but_hri_is_not_cross_snapshot_mixed():
    actual = {
        name: (mode.channel, mode.fits_filter, mode.fits_filtpos)
        for name, mode in eui.EUI_FSI_FILTER_MODES.items()
    }
    assert actual == EXPECTED_FILTER_MODES
    assert eui.EUI_DEFAULT_MODE_BY_CHANNEL == {
        174: "fsi174_n25",
        304: "fsi304_n4",
    }
    discovery = eui.available_eui_modes()
    assert set(discovery["filter_modes"]) == set(EXPECTED_FILTER_MODES)
    assert discovery["fits_header_keywords"] == {
        "filter": "FILTER",
        "filter_position": "FILTPOS",
    }
    assert discovery["hri_174"]["available"] is False


def test_default_export_records_units_static_sensitivity_and_fits_binding(
    tmp_path, monkeypatch
):
    _write_test_sources(tmp_path, monkeypatch, ("fsi174_n25", "fsi304_n4"))
    output_path = tmp_path / "eui.throughput.npz"

    artifact = eui.export_eui_throughput(tmp_path, output_path)

    assert artifact.channels == ("174", "304")
    np.testing.assert_array_equal(
        artifact.wavelength_angstrom,
        np.arange(10.0, 1001.0),
    )
    assert artifact.throughput.shape == (2, 991)
    assert np.isfinite(artifact.throughput).all()
    assert np.all(artifact.throughput >= 0)
    assert artifact.throughput_unit == "cm2 sr DN / (ph pix)"
    assert artifact.provenance["sensitivity_convention"] == "static_assumed"

    radiometry = artifact.provenance["radiometry"]
    assert radiometry["source_table_unit"] == "DN ph-1 cm2 sr"
    assert radiometry["native_pixel_solid_angle_sr"] == pytest.approx(
        4.6309190622000823e-10,
        rel=0,
        abs=0,
    )
    assert radiometry["filter_position_by_channel"] == {"174": 50, "304": 0}
    assert radiometry["fits_filter_by_channel"] == {
        "174": "Zirconium_174_n25",
        "304": "Magnesium_304_n4",
    }
    assert radiometry["filter_mapping"]["174"]["logical_mode"] == "fsi174_n25"
    assert radiometry["filter_mapping"]["304"]["logical_mode"] == "fsi304_n4"
    assert load_instrument_throughput(output_path).content_sha256 == (
        artifact.content_sha256
    )


def test_export_accepts_alternate_header_modes_without_duplicate_channels(
    tmp_path, monkeypatch
):
    _write_test_sources(tmp_path, monkeypatch, ("fsi174_n13", "fsi304_n26"))

    artifact = eui.export_eui_throughput(
        tmp_path,
        None,
        channels=("EUI_174", "304"),
        mode_by_channel={"174": "N13", 304: "#26"},
    )

    assert artifact.channels == ("174", "304")
    assert artifact.provenance["radiometry"]["filter_position_by_channel"] == {
        "174": 150,
        "304": 100,
    }
    with pytest.raises(ValueError, match="Unsupported EUI FSI filter mode"):
        eui.export_eui_throughput(
            tmp_path,
            None,
            channels=(174,),
            mode_by_channel={174: "n26"},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("grid", "wavelength grid must be exactly"),
        ("negative", "finite and non-negative"),
        ("nonfinite", "finite and non-negative"),
        ("header", "response header does not match"),
    ],
)
def test_table_reader_rejects_noncanonical_or_nonphysical_data(
    tmp_path, mutation, message
):
    mode = eui.EUI_FSI_FILTER_MODES["fsi174_n25"]
    wavelength = eui.EUI_FSI_WAVELENGTH_NM.copy()
    response = np.ones(wavelength.size)
    header = list(eui._expected_header(mode))
    if mutation == "grid":
        wavelength[10] += 0.01
    elif mutation == "negative":
        response[10] = -1.0
    elif mutation == "nonfinite":
        response[10] = np.nan
    else:
        header[1] = "# version=latest"
    path = Path(tmp_path) / "response.txt"
    path.write_bytes(
        _table_payload(
            mode,
            wavelength_nm=wavelength,
            response=response,
            header=header,
        )
    )

    with pytest.raises(ValueError, match=message):
        eui._read_fsi_response_table(path, mode)
