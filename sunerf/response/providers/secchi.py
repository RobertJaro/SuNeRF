"""Official STEREO/SECCHI EUVI wavelength-throughput provider.

The SolarSoft SRA products contain the end-to-end effective area for every
spacecraft, wavelength quadrant, and filter-wheel position.  This exporter
matches the historical ``SECCHI_PREP`` ``PhotonFlux`` convention explicitly;
it does not use the bundled, old-CHIANTI SRE temperature responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.io import readsav

from sunerf.response.builder import InstrumentThroughput
from sunerf.response.providers.base import (
    SourceFile,
    fetch_sources,
    require_source_paths,
)


PROVIDER_NAME = "SolarSoft/SECCHI EUVI SRA"
PROVIDER_VERSION = "001"
CALIBRATION_EPOCH = "2008-02-19T00:00:00Z"
REFERENCE_DIRECTORY = (
    "https://soho.nascom.nasa.gov/solarsoft/stereo/secchi/calibration/"
    "euvi_response/"
)

EUVI_SOURCES = {
    "A": SourceFile(
        key="euvi_a_sra",
        filename="ahead_sra_001.geny",
        url=f"{REFERENCE_DIRECTORY}ahead_sra_001.geny",
        sha256="0883e6048088520abc0bed8924df01b141661ba7e63b328ef1e5701b0c43ce9f",
        size_bytes=2_677_664,
        version="001",
        description="STEREO-A/SECCHI EUVI end-to-end spectral response area",
        reference_url=REFERENCE_DIRECTORY,
    ),
    "B": SourceFile(
        key="euvi_b_sra",
        filename="behind_sra_001.geny",
        url=f"{REFERENCE_DIRECTORY}behind_sra_001.geny",
        sha256="7dd3920dd09102e230405d8e6904fb4cbe0773fa044351cbcc30f0a302063333",
        size_bytes=2_677_648,
        version="001",
        description="STEREO-B/SECCHI EUVI end-to-end spectral response area",
        reference_url=REFERENCE_DIRECTORY,
    ),
}

CHANNELS = (171, 195, 284, 304)
FILTERS = ("OPEN", "S1", "S2", "DBL")

# EUVI_GET_NORMAL values.  SECCHI_PREP divides the calibrated signal by this
# scalar unless /NORMAL_OFF is supplied.  The scalar does not turn a filtered
# bandpass into the OPEN bandpass; the actual SRA filter curve remains required.
OPEN_NORMALIZATION = {
    171: {"OPEN": 1.0, "S1": 0.49, "S2": 0.49, "DBL": 0.41},
    195: {"OPEN": 1.0, "S1": 0.49, "S2": 0.49, "DBL": 0.41},
    284: {"OPEN": 1.0, "S1": 0.33, "S2": 0.33, "DBL": 0.24},
    304: {"OPEN": 1.0, "S1": 0.29, "S2": 0.29, "DBL": 0.22},
}

# Native physical CCD pixels, not the possibly binned WCS pixels in an output
# image.  Values come from the SECCHI/EUVI calibration and measurement-data
# document plate scales of 1.5882 and 1.5904 arcsec/pixel.
NATIVE_PIXEL_SOLID_ANGLE_SR = {
    "A": 5.92870876398425e-11,
    "B": 5.9451452242248e-11,
}


@dataclass(frozen=True)
class EUVISpectralResponseArea:
    spacecraft: str
    wavelength_angstrom: np.ndarray
    area_cm2: np.ndarray
    version: str
    release_date: str


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii").strip()
    return str(value).strip()


def _labels(values) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_text(item) for item in row) for row in np.asarray(values))


def load_euvi_sra(path: str | Path, *, spacecraft: str) -> EUVISpectralResponseArea:
    """Read and strictly validate one official IDL SAVE/GENY SRA product."""
    spacecraft = spacecraft.upper()
    if spacecraft not in EUVI_SOURCES:
        raise ValueError("spacecraft must be 'A' or 'B'")
    source = EUVI_SOURCES[spacecraft]
    path = require_source_paths((source,), Path(path).parent)[source.key]
    archive = readsav(path, python_dict=True, verbose=False)
    if set(archive) != {"p0"}:
        raise ValueError(f"Unexpected SECCHI SRA variables in {path}: {sorted(archive)}")
    records = archive["p0"]
    if np.asarray(records).shape != (1,):
        raise ValueError("SECCHI SRA P0 must contain exactly one record")
    record = records[0]
    required = {
        "VERSION", "DATE", "STEREO", "LAMBDA", "AREA", "WAVELNTH", "FILTER"
    }
    names = set(record.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"SECCHI SRA record is missing fields {sorted(required - names)}")
    if _text(record["VERSION"]) != PROVIDER_VERSION:
        raise ValueError(f"Unsupported SECCHI SRA version {_text(record['VERSION'])!r}")
    if _text(record["STEREO"]).upper() != spacecraft:
        raise ValueError(
            f"SECCHI SRA spacecraft {_text(record['STEREO'])!r} does not match {spacecraft}"
        )

    wavelength = np.asarray(record["LAMBDA"], dtype=np.float64)
    area = np.asarray(record["AREA"], dtype=np.float64)
    if wavelength.shape != (20_000,):
        raise ValueError(f"SECCHI SRA wavelength shape must be (20000,), got {wavelength.shape}")
    if not np.isfinite(wavelength).all() or np.any(np.diff(wavelength) <= 0):
        raise ValueError("SECCHI SRA wavelength axis must be finite and strictly increasing")
    if area.shape != (4, 4, wavelength.size) or not np.isfinite(area).all():
        raise ValueError(
            "SECCHI SRA AREA must be a finite (4,4,20000) array, "
            f"got {area.shape}"
        )
    expected_channels = tuple((str(channel),) * 4 for channel in CHANNELS)
    if _labels(record["WAVELNTH"]) != expected_channels:
        raise ValueError("SECCHI SRA WAVELNTH labels do not match the documented layout")
    expected_filters = (FILTERS,) * 4
    if _labels(record["FILTER"]) != expected_filters:
        raise ValueError("SECCHI SRA FILTER labels do not match the documented layout")

    # The released Behind file contains a tiny negative interpolation artifact
    # only in the 284-A curves.  Reject any changed pattern before clipping it.
    negative = area < 0
    if spacecraft == "A" and np.any(negative):
        raise ValueError("STEREO-A SRA unexpectedly contains negative effective area")
    if spacecraft == "B":
        expected_negative = np.zeros_like(negative)
        expected_negative[2] = area[2] < 0
        if (
            int(negative.sum()) != 460
            or not np.array_equal(negative, expected_negative)
            or not np.isclose(area.min(), -5.664088803314371e-06, rtol=1e-7, atol=0)
        ):
            raise ValueError(
                "STEREO-B negative-area pattern differs from the pinned SRA release"
            )
        area = np.clip(area, 0.0, None)

    return EUVISpectralResponseArea(
        spacecraft=spacecraft,
        wavelength_angstrom=wavelength,
        area_cm2=area,
        version=_text(record["VERSION"]),
        release_date=_text(record["DATE"]),
    )


def fetch_secchi_sources(
    destination_dir: str | Path,
    *,
    spacecraft: Sequence[str] = ("A", "B"),
    force: bool = False,
) -> Mapping[str, Path]:
    """Fetch the byte-pinned A/B SRA files."""
    selected = tuple(EUVI_SOURCES[item.upper()] for item in spacecraft)
    return fetch_sources(selected, destination_dir, force=force)


def export_secchi_euvi_throughput(
    source_dir: str | Path,
    output_path: str | Path | None,
    *,
    spacecraft: str,
    channels: Sequence[int] = (171, 195, 284),
    filter_by_channel: Mapping[int, str] | None = None,
    secchi_prep_normalized_to_open: bool = False,
) -> InstrumentThroughput:
    """Export throughput matching SECCHI_PREP ``PhotonFlux`` images.

    ``secchi_prep_normalized_to_open=False`` corresponds to running
    ``SECCHI_PREP,...,/NORMAL_OFF`` and is the recommended release convention.
    The wavelength-dependent ``lambda_nominal/lambda`` factor matches
    ``GET_CALFAC``'s conversion from energy-weighted DN to photon-equivalent
    counts at each channel's nominal wavelength.
    """
    spacecraft = spacecraft.upper()
    if spacecraft not in EUVI_SOURCES:
        raise ValueError("spacecraft must be 'A' or 'B'")
    channels = tuple(int(channel) for channel in channels)
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("channels must be a non-empty unique sequence")
    unknown_channels = sorted(set(channels) - set(CHANNELS))
    if unknown_channels:
        raise ValueError(f"Unsupported EUVI channels {unknown_channels}")
    selected_filters = {channel: "S1" for channel in channels}
    if filter_by_channel is not None:
        supplied_keys = {int(key) for key in filter_by_channel}
        if supplied_keys != set(channels):
            raise ValueError("filter_by_channel must specify exactly the selected channels")
        selected_filters = {
            int(channel): str(value).upper()
            for channel, value in filter_by_channel.items()
        }
    invalid_filters = {
        channel: filter_name
        for channel, filter_name in selected_filters.items()
        if filter_name not in FILTERS
    }
    if invalid_filters:
        raise ValueError(f"Unsupported EUVI filter selections {invalid_filters}")

    source = EUVI_SOURCES[spacecraft]
    path = require_source_paths((source,), source_dir)[source.key]
    sra = load_euvi_sra(path, spacecraft=spacecraft)
    solid_angle = NATIVE_PIXEL_SOLID_ANGLE_SR[spacecraft]
    curves = []
    normalization = {}
    for channel in channels:
        channel_index = CHANNELS.index(channel)
        filter_name = selected_filters[channel]
        filter_index = FILTERS.index(filter_name)
        scalar = (
            OPEN_NORMALIZATION[channel][filter_name]
            if secchi_prep_normalized_to_open
            else 1.0
        )
        normalization[channel] = scalar
        curve = sra.area_cm2[channel_index, filter_index].copy()
        curve *= channel / sra.wavelength_angstrom
        curve *= solid_angle / scalar
        curves.append(curve)

    calibration = source.as_provenance()
    calibration.update(
        {
            "name": source.filename,
            "version": source.version,
            "sha256": source.sha256,
        }
    )
    # SourceFile calls this field ``url`` while response provenance elsewhere
    # calls it ``source_url``. Retain just the latter canonical spelling.
    calibration["source_url"] = calibration.pop("url")
    calibration.pop("filename", None)
    artifact = InstrumentThroughput(
        channels=tuple(str(channel) for channel in channels),
        wavelength_angstrom=sra.wavelength_angstrom,
        throughput=np.stack(curves),
        throughput_unit="cm2 sr pix-1",
        calibration_epoch=CALIBRATION_EPOCH,
        provenance={
            "instrument": {
                "name": f"STEREO-{spacecraft}/SECCHI/EUVI",
                "spacecraft": spacecraft,
            },
            "provider": {"name": PROVIDER_NAME, "version": PROVIDER_VERSION},
            "calibration": calibration,
            "sensitivity_convention": "static_assumed",
            "static_sensitivity_assumption": {
                "applied": True,
                "reason": (
                    "The official SRA and SECCHI GET_CALFAC provide no "
                    "time-dependent EUVI degradation model."
                ),
            },
            "radiometry": {
                "measurement_semantics": "per_native_pixel",
                "native_pixel_solid_angle_sr": solid_angle,
                "native_pixel_solid_angle_relative_tolerance": 0.02,
                "secchi_prep_output": "PhotonFlux",
                "secchi_prep_normal_off": not secchi_prep_normalized_to_open,
                "photon_equivalent_factor": "lambda_nominal/lambda",
                "filter_by_channel": {
                    str(channel): selected_filters[channel] for channel in channels
                },
                "open_normalization_divisor": {
                    str(channel): normalization[channel] for channel in channels
                },
                "pixel_definition": "native physical CCD pixel after IPSUM correction",
            },
            "data_repairs": (
                {
                    "policy": "clip_pinned_negative_interpolation_artifacts_to_zero",
                    "sample_count": 460,
                    "minimum_cm2": -5.664088803314371e-06,
                    "affected_channel": 284,
                }
                if spacecraft == "B"
                else {"policy": "none", "sample_count": 0}
            ),
        },
    )
    if output_path is not None:
        artifact.save(output_path)
    return artifact


def available_euvi_modes() -> dict[str, object]:
    """Describe the provider surface for registry/CLI discovery."""
    return {
        "spacecraft": tuple(EUVI_SOURCES),
        "channels": CHANNELS,
        "filters": FILTERS,
        "default_channels": (171, 195, 284),
        "default_filter": "S1",
        "recommended_secchi_prep_keyword": "/NORMAL_OFF",
        "sensitivity_convention": "static_assumed",
    }
