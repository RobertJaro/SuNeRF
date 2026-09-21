"""Pinned Solar Orbiter/EUI FSI wavelength-throughput exporter.

The four source tables are the public V2, on-axis, PTB-merged instrument
responses at the immutable ``euipublic`` commit where they were introduced.
They are calibration inputs, not a numbered EUI science-data release. An FSI image is
not identified completely by its nominal 174 or 304 Angstrom channel: each
channel has two physical filter-wheel positions.  Consequently an exported
artifact contains at most one response per nominal channel and records the
exact ``FILTER``/``FILTPOS`` binding needed to use it.

The pinned calibration commit contains no HRI-EUV response table. A table found
on another branch is deliberately not mixed into this provider.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sunerf.response.builder import InstrumentThroughput
from sunerf.response.providers.base import (
    SourceFile,
    fetch_sources,
    require_source_paths,
    verify_source,
)


EUI_CALIBRATION_COMMIT = "3619e9199ccce8421aa4c8b22c655741a4b8ab1c"
EUI_CALIBRATION_SNAPSHOT = "2023-12-07"
EUI_PROVIDER_NAME = "Solar Orbiter/EUI public calibration"
EUI_PROVIDER_VERSION = f"euipublic-{EUI_CALIBRATION_COMMIT[:8]}-fsi-v2"
EUI_CALIBRATION_VERSION = (
    f"V2/Center/Instrument/Merged/R0@{EUI_CALIBRATION_COMMIT[:8]}"
)

EUI_DATA_RELEASE_NOTES_URL = "https://www.sidc.be/EUI/data/latest_release_notes.html"
EUI_METADATA_STANDARD_URL = (
    "https://www.sidc.be/EUI/data/documents/"
    "SP_ROB_SOEUI_19002_MetadataStandard_1.8.pdf"
)
EUI_LAUNCH_REFERENCE_URL = (
    "https://www.esa.int/ESA_Multimedia/Videos/2020/02/Solar_Orbiter_liftoff"
)
_EUI_RESPONSE_TREE_URL = (
    "https://gitlab-as.oma.be/sidcpublic/euipublic/-/tree/"
    f"{EUI_CALIBRATION_COMMIT}/soloEUI/data/fsi/response/V2/Center/Instrument/Merged"
)
_EUI_RESPONSE_RAW_ROOT = (
    "https://gitlab-as.oma.be/sidcpublic/euipublic/-/raw/"
    f"{EUI_CALIBRATION_COMMIT}/soloEUI/data/fsi/response/V2/Center/Instrument/Merged"
)

# The public response generator multiplies the optics/detector response by
# these quantities before writing the pinned tables.  They are recorded here,
# not applied again.
EUI_FSI_NATIVE_PIXEL_SCALE_ARCSEC = 4.43873
EUI_FSI_NATIVE_PIXEL_SOLID_ANGLE_SR = 4.6309190622000823e-10
EUI_FSI_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE = 0.02
EUI_FSI_PUPIL_EDGE_CM = 0.275
EUI_FSI_PUPIL_AREA_CM2 = 0.19647951348359455
EUI_FSI_SOURCE_TABLE_UNIT = "DN ph-1 cm2 sr"
EUI_FSI_SOURCE_TABLE_HEADER_UNIT = "DN.ph$^{-1}$.cm$^2$.sr"
EUI_FSI_THROUGHPUT_UNIT = "cm2 sr DN / (ph pix)"

# Instrument response tables have no sensitivity epoch or time-dependent
# correction.  InstrumentThroughput requires a timestamp even for the
# static_assumed convention, so Solar Orbiter launch is used only as an
# explicitly declared beginning-of-life schema anchor.
EUI_CALIBRATION_EPOCH = "2020-02-10T04:03:00Z"

EUI_FSI_WAVELENGTH_NM = np.arange(10, 1001, dtype=np.float64) / 10.0
EUI_FSI_WAVELENGTH_ANGSTROM = np.arange(10, 1001, dtype=np.float64)


def _source(
    key: str,
    filename: str,
    sha256: str,
    size_bytes: int,
    description: str,
) -> SourceFile:
    return SourceFile(
        key=key,
        filename=filename,
        url=f"{_EUI_RESPONSE_RAW_ROOT}/{filename}",
        sha256=sha256,
        size_bytes=size_bytes,
        version=EUI_CALIBRATION_VERSION,
        description=description,
        reference_url=_EUI_RESPONSE_TREE_URL,
    )


EUI_FSI_SOURCES = {
    "fsi304_n4": _source(
        "fsi304_n4",
        "FSI_Response_Merged_R0_AlMg4_V2.txt",
        "66f04d9291b8ae3fce6ab29b707c661511cfa15b614aa727c1818ef64ad83a7c",
        21_957,
        "FSI 304 Angstrom filter #4 end-to-end response (FILTER Magnesium_304_n4)",
    ),
    "fsi174_n25": _source(
        "fsi174_n25",
        "FSI_Response_Merged_R0_AlZr25_V2.txt",
        "785ef1859bff35c788fae2be87caafa743b204d86e0d259abf65efb9fd3fc7ff",
        21_958,
        "FSI 174 Angstrom filter #25 end-to-end response (FILTER Zirconium_174_n25)",
    ),
    "fsi304_n26": _source(
        "fsi304_n26",
        "FSI_Response_Merged_R0_AlMg26_V2.txt",
        "98fe2530e2ef9e54fd85bff8f5e826f859b1d3d93835f3dd90cddf8088ede386",
        21_958,
        "FSI 304 Angstrom filter #26 end-to-end response (FILTER Magnesium_304_n26)",
    ),
    "fsi174_n13": _source(
        "fsi174_n13",
        "FSI_Response_Merged_R0_AlZr13_V2.txt",
        "4669b35606830964d5575c96f93e286a6f4e9780a6fbe5a65781c94e1ff5c5fd",
        21_958,
        "FSI 174 Angstrom filter #13 end-to-end response (FILTER Zirconium_174_n13)",
    ),
}
EUI_FSI_THROUGHPUT_SOURCES = tuple(EUI_FSI_SOURCES.values())


@dataclass(frozen=True)
class EUIFSIFilterMode:
    """One physical FSI filter-wheel mode for a nominal EUV channel."""

    logical_mode: str
    channel: int
    filter_serial: str
    fits_filter: str
    fits_filtpos: int
    source_key: str
    optics_response_filename: str

    def as_mapping(self) -> dict[str, object]:
        source = EUI_FSI_SOURCES[self.source_key]
        return {
            "logical_mode": self.logical_mode,
            "nominal_wavelength_angstrom": self.channel,
            "filter_serial": self.filter_serial,
            "fits_filter": self.fits_filter,
            "fits_filtpos": self.fits_filtpos,
            "calibration_source_key": self.source_key,
            "calibration_filename": source.filename,
        }


# FILTPOS values and FILTER strings are from the official EUI metadata
# standard.  Dictionary order follows increasing filter-wheel position.
EUI_FSI_FILTER_MODES = {
    "fsi304_n4": EUIFSIFilterMode(
        logical_mode="fsi304_n4",
        channel=304,
        filter_serial="n4",
        fits_filter="Magnesium_304_n4",
        fits_filtpos=0,
        source_key="fsi304_n4",
        optics_response_filename="FSI_Instrument_Merged_R0_AlMg4_V2.txt",
    ),
    "fsi174_n25": EUIFSIFilterMode(
        logical_mode="fsi174_n25",
        channel=174,
        filter_serial="n25",
        fits_filter="Zirconium_174_n25",
        fits_filtpos=50,
        source_key="fsi174_n25",
        optics_response_filename="FSI_Instrument_Merged_R0_AlZr25_V2.txt",
    ),
    "fsi304_n26": EUIFSIFilterMode(
        logical_mode="fsi304_n26",
        channel=304,
        filter_serial="n26",
        fits_filter="Magnesium_304_n26",
        fits_filtpos=100,
        source_key="fsi304_n26",
        optics_response_filename="FSI_Instrument_Merged_R0_AlMg26_V2.txt",
    ),
    "fsi174_n13": EUIFSIFilterMode(
        logical_mode="fsi174_n13",
        channel=174,
        filter_serial="n13",
        fits_filter="Zirconium_174_n13",
        fits_filtpos=150,
        source_key="fsi174_n13",
        optics_response_filename="FSI_Instrument_Merged_R0_AlZr13_V2.txt",
    ),
}

EUI_FSI_CHANNELS = (174, 304)
EUI_DEFAULT_MODE_BY_CHANNEL = {174: "fsi174_n25", 304: "fsi304_n4"}

EUI_HRI_174_RELEASE_COMPATIBLE = False
EUI_HRI_174_EXCLUSION_REASON = (
    "The pinned public calibration commit contains no HRI-EUV wavelength-response "
    "table; a table from another branch is not a compatible input."
)


@dataclass(frozen=True)
class EUIFSISpectralResponse:
    """Strictly validated contents of one pinned FSI response table."""

    mode: EUIFSIFilterMode
    wavelength_angstrom: np.ndarray
    response: np.ndarray


def _expected_header(mode: EUIFSIFilterMode) -> tuple[str, ...]:
    return (
        "# PTB=True",
        "# version=2",
        "# location=Center",
        "# detector=True",
        f"# Optics response: {mode.optics_response_filename}",
        f"# Wavelength (nm) {EUI_FSI_SOURCE_TABLE_HEADER_UNIT}",
    )


def _read_fsi_response_table(
    path: str | Path,
    mode: EUIFSIFilterMode,
) -> EUIFSISpectralResponse:
    """Read a response table after byte verification or in focused tests."""
    path = Path(path)
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"EUI FSI response table is not ASCII: {path}") from error
    if tuple(lines[:6]) != _expected_header(mode):
        raise ValueError(
            f"EUI FSI {mode.logical_mode} response header does not match "
            "V2 Center/Merged detector-response metadata"
        )
    if len(lines) != 6 + EUI_FSI_WAVELENGTH_NM.size:
        raise ValueError(
            f"EUI FSI {mode.logical_mode} response table must contain exactly "
            f"{EUI_FSI_WAVELENGTH_NM.size} data rows"
        )
    try:
        table = np.loadtxt(path, comments="#", dtype=np.float64)
    except ValueError as error:
        raise ValueError(f"Could not parse EUI FSI response table {path}") from error
    expected_shape = (EUI_FSI_WAVELENGTH_NM.size, 2)
    if table.shape != expected_shape:
        raise ValueError(
            f"EUI FSI {mode.logical_mode} response shape must be {expected_shape}; "
            f"received {table.shape}"
        )
    if not np.array_equal(table[:, 0], EUI_FSI_WAVELENGTH_NM):
        raise ValueError(
            f"EUI FSI {mode.logical_mode} wavelength grid must be exactly "
            "1.0--100.0 nm in 0.1 nm steps"
        )
    response = table[:, 1]
    if not np.isfinite(response).all() or np.any(response < 0):
        raise ValueError(
            f"EUI FSI {mode.logical_mode} response must be finite and non-negative"
        )
    if response.max() <= 0:
        raise ValueError(f"EUI FSI {mode.logical_mode} response has no positive support")
    return EUIFSISpectralResponse(
        mode=mode,
        wavelength_angstrom=EUI_FSI_WAVELENGTH_ANGSTROM.copy(),
        response=response.copy(),
    )


def _resolve_mode(value: str, *, channel: int | None = None) -> EUIFSIFilterMode:
    candidate = str(value).strip().casefold()
    for mode in EUI_FSI_FILTER_MODES.values():
        serial_number = mode.filter_serial[1:]
        aliases = {
            mode.logical_mode.casefold(),
            mode.filter_serial.casefold(),
            serial_number.casefold(),
            f"#{serial_number}".casefold(),
            mode.fits_filter.casefold(),
        }
        if candidate in aliases and (channel is None or mode.channel == channel):
            return mode
    context = "" if channel is None else f" for channel {channel}"
    raise ValueError(f"Unsupported EUI FSI filter mode {value!r}{context}")


def _normalize_channels(channels: Sequence[int | str]) -> tuple[int, ...]:
    normalized = []
    for value in channels:
        if isinstance(value, bool):
            raise ValueError("EUI FSI channels must be 174 and/or 304")
        text = str(value).strip()
        numeric_text = text[4:] if text.casefold().startswith("eui_") else text
        try:
            channel = int(numeric_text)
        except (TypeError, ValueError) as error:
            raise ValueError("EUI FSI channels must be 174 and/or 304") from error
        if text.casefold() not in {str(channel), f"eui_{channel}"}:
            raise ValueError("EUI FSI channels must be 174 and/or 304")
        normalized.append(channel)
    result = tuple(normalized)
    if not result or len(result) != len(set(result)):
        raise ValueError("channels must be a non-empty unique sequence")
    unsupported = sorted(set(result) - set(EUI_FSI_CHANNELS))
    if unsupported:
        raise ValueError(f"Unsupported EUI FSI channels {unsupported}")
    return result


def _select_modes(
    channels: tuple[int, ...],
    mode_by_channel: Mapping[int | str, str] | None,
) -> tuple[EUIFSIFilterMode, ...]:
    overrides: dict[int, str] = {}
    if mode_by_channel is not None:
        if not isinstance(mode_by_channel, Mapping):
            raise ValueError("mode_by_channel must be a mapping")
        for key, value in mode_by_channel.items():
            channel = _normalize_channels((key,))[0]
            if channel in overrides:
                raise ValueError(f"mode_by_channel repeats channel {channel}")
            overrides[channel] = value
        unexpected = sorted(set(overrides) - set(channels))
        if unexpected:
            raise ValueError(
                "mode_by_channel contains channels not selected for export: "
                f"{unexpected}"
            )
    return tuple(
        _resolve_mode(
            overrides.get(channel, EUI_DEFAULT_MODE_BY_CHANNEL[channel]),
            channel=channel,
        )
        for channel in channels
    )


def load_eui_fsi_response(
    path: str | Path,
    *,
    mode: str,
) -> EUIFSISpectralResponse:
    """Byte-verify and strictly parse one official FSI response table."""
    resolved_mode = _resolve_mode(mode)
    source = EUI_FSI_SOURCES[resolved_mode.source_key]
    verified_path = verify_source(path, source)
    return _read_fsi_response_table(verified_path, resolved_mode)


def fetch_eui_sources(
    destination_dir: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> Mapping[str, Path]:
    """Fetch all four byte-pinned public FSI filter response tables."""
    return fetch_sources(
        EUI_FSI_THROUGHPUT_SOURCES,
        destination_dir,
        force=force,
        timeout_seconds=timeout_seconds,
    )


def _calibration_bundle_sha256(modes: Sequence[EUIFSIFilterMode]) -> str:
    inputs = {
        mode.logical_mode: {
            "filename": EUI_FSI_SOURCES[mode.source_key].filename,
            "sha256": EUI_FSI_SOURCES[mode.source_key].sha256,
        }
        for mode in modes
    }
    payload = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_eui_throughput(
    source_dir: str | Path,
    output_path: str | Path | None,
    *,
    channels: Sequence[int | str] = EUI_FSI_CHANNELS,
    mode_by_channel: Mapping[int | str, str] | None = None,
) -> InstrumentThroughput:
    """Export a canonical FSI 174/304 throughput artifact.

    The defaults are filter #25 (``FILTPOS=50``) for FSI 174 and filter #4
    (``FILTPOS=0``) for FSI 304.  Use ``mode_by_channel={174: "n13"}`` or
    ``mode_by_channel={304: "n26"}`` when the input FITS headers select the
    alternate physical filter.  One artifact cannot contain both modes for the
    same nominal channel because runtime channel identifiers remain ``174`` and
    ``304``.
    """
    channels = _normalize_channels(channels)
    modes = _select_modes(channels, mode_by_channel)
    sources = tuple(EUI_FSI_SOURCES[mode.source_key] for mode in modes)
    paths = require_source_paths(sources, source_dir)

    responses = [
        _read_fsi_response_table(paths[mode.source_key], mode)
        for mode in modes
    ]
    for response in responses[1:]:
        if not np.array_equal(
            response.wavelength_angstrom,
            responses[0].wavelength_angstrom,
        ):
            raise ValueError("Selected EUI FSI response tables do not share an exact grid")

    channel_names = tuple(str(channel) for channel in channels)
    filter_mapping = {
        str(mode.channel): mode.as_mapping()
        for mode in modes
    }
    source_provenance = {
        mode.logical_mode: EUI_FSI_SOURCES[mode.source_key].as_provenance()
        for mode in modes
    }
    provenance = {
        "instrument": {
            "name": "Solar Orbiter/EUI/FSI",
            "spacecraft": "Solar Orbiter",
            "detector": "FSI",
        },
        "provider": {
            "name": EUI_PROVIDER_NAME,
            "version": EUI_PROVIDER_VERSION,
        },
        "calibration": {
            "name": "EUI FSI V2 Center PTB-merged instrument-response tables",
            "version": EUI_CALIBRATION_VERSION,
            "sha256": _calibration_bundle_sha256(modes),
            "sha256_scope": (
                "SHA-256 of canonical JSON mapping each selected logical mode "
                "to its source filename and byte SHA-256"
            ),
            "source_commit": EUI_CALIBRATION_COMMIT,
            "snapshot_date": EUI_CALIBRATION_SNAPSHOT,
            "sources": source_provenance,
        },
        "sensitivity_convention": "static_assumed",
        "static_sensitivity_assumption": {
            "applied": True,
            "time_dependent_degradation_correction_included": False,
            "response_state": "static preflight PTB-merged model",
            "reason": (
                "The pinned wavelength-response files encode no sensitivity epoch "
                "or time-dependent degradation correction."
            ),
        },
        "calibration_epoch_basis": {
            "epoch": EUI_CALIBRATION_EPOCH,
            "role": "beginning-of-life schema anchor only",
            "basis": "Solar Orbiter launch; not an epoch encoded by the response files",
            "reference_url": EUI_LAUNCH_REFERENCE_URL,
        },
        "radiometry": {
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": EUI_FSI_NATIVE_PIXEL_SOLID_ANGLE_SR,
            "native_pixel_solid_angle_relative_tolerance": (
                EUI_FSI_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE
            ),
            "native_pixel_plate_scale_arcsec": EUI_FSI_NATIVE_PIXEL_SCALE_ARCSEC,
            "native_pixel_definition": "unbinned 3072 x 3072 FSI detector pixel",
            "source_table_unit": EUI_FSI_SOURCE_TABLE_UNIT,
            "source_table_header_unit": EUI_FSI_SOURCE_TABLE_HEADER_UNIT,
            "artifact_unit_pixel_bookkeeping": (
                "pix-1 records that each source-table value already contains one "
                "native-pixel solid angle"
            ),
            "included_components": {
                "ptb_merged_optics_response": True,
                "detector_response": True,
                "entrance_pupil_area": True,
                "native_pixel_solid_angle": True,
            },
            "entrance_pupil_area_cm2": EUI_FSI_PUPIL_AREA_CM2,
            "fits_header_keywords": {
                "filter": "FILTER",
                "filter_position": "FILTPOS",
            },
            "filter_mapping": filter_mapping,
            "filter_position_by_channel": {
                str(mode.channel): mode.fits_filtpos for mode in modes
            },
            "fits_filter_by_channel": {
                str(mode.channel): mode.fits_filter for mode in modes
            },
            "metadata_standard_url": EUI_METADATA_STANDARD_URL,
        },
        "hri_174": {
            "included": False,
            "release_compatible": EUI_HRI_174_RELEASE_COMPATIBLE,
            "reason": EUI_HRI_174_EXCLUSION_REASON,
        },
        "wavelength_grid": {
            "coordinate": "vacuum_angstrom",
            "minimum_angstrom": 10.0,
            "maximum_angstrom": 1000.0,
            "step_angstrom": 1.0,
            "count": 991,
        },
        "references": {
            "science_data_release_notes": EUI_DATA_RELEASE_NOTES_URL,
            "metadata_standard": EUI_METADATA_STANDARD_URL,
            "immutable_response_tree": _EUI_RESPONSE_TREE_URL,
        },
    }
    artifact = InstrumentThroughput(
        channels=channel_names,
        wavelength_angstrom=responses[0].wavelength_angstrom,
        throughput=np.stack([response.response for response in responses]),
        throughput_unit=EUI_FSI_THROUGHPUT_UNIT,
        calibration_epoch=EUI_CALIBRATION_EPOCH,
        provenance=provenance,
    )
    if output_path is not None:
        artifact.save(output_path)
    return artifact


def available_eui_modes() -> dict[str, object]:
    """Describe every pinned FSI table and the safe artifact defaults."""
    return {
        "instrument": "Solar Orbiter/EUI/FSI",
        "calibration_snapshot": EUI_CALIBRATION_SNAPSHOT,
        "calibration_commit": EUI_CALIBRATION_COMMIT,
        "channels": EUI_FSI_CHANNELS,
        "default_mode_by_channel": {
            str(channel): mode
            for channel, mode in EUI_DEFAULT_MODE_BY_CHANNEL.items()
        },
        "filter_modes": {
            name: {
                **mode.as_mapping(),
                "source": EUI_FSI_SOURCES[mode.source_key].as_provenance(),
            }
            for name, mode in EUI_FSI_FILTER_MODES.items()
        },
        "fits_header_keywords": {
            "filter": "FILTER",
            "filter_position": "FILTPOS",
        },
        "sensitivity_convention": "static_assumed",
        "hri_174": {
            "available": EUI_HRI_174_RELEASE_COMPATIBLE,
            "reason": EUI_HRI_174_EXCLUSION_REASON,
        },
    }


__all__ = [
    "EUI_CALIBRATION_EPOCH",
    "EUI_CALIBRATION_COMMIT",
    "EUI_CALIBRATION_SNAPSHOT",
    "EUI_DATA_RELEASE_NOTES_URL",
    "EUI_CALIBRATION_VERSION",
    "EUI_DEFAULT_MODE_BY_CHANNEL",
    "EUI_FSI_CHANNELS",
    "EUI_FSI_FILTER_MODES",
    "EUI_FSI_NATIVE_PIXEL_SCALE_ARCSEC",
    "EUI_FSI_NATIVE_PIXEL_SOLID_ANGLE_SR",
    "EUI_FSI_PUPIL_AREA_CM2",
    "EUI_FSI_SOURCE_TABLE_UNIT",
    "EUI_FSI_SOURCES",
    "EUI_FSI_THROUGHPUT_SOURCES",
    "EUI_HRI_174_RELEASE_COMPATIBLE",
    "EUI_HRI_174_EXCLUSION_REASON",
    "EUI_PROVIDER_NAME",
    "EUI_PROVIDER_VERSION",
    "EUIFSIFilterMode",
    "EUIFSISpectralResponse",
    "available_eui_modes",
    "export_eui_throughput",
    "fetch_eui_sources",
    "load_eui_fsi_response",
]
