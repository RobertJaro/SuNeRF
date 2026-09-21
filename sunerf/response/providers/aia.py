"""Pinned SDO/AIA wavelength-throughput exporter.

The exporter deliberately separates network access from calibration.  Fetching
is handled by :func:`fetch_aia_sources`; :func:`export_aia_throughput` accepts
only byte-verified local SolarSoft inputs and disables aiapy's implicit data
downloads by passing the instrument file and correction table explicitly.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys

import numpy as np
from astropy import units as u
from astropy.time import Time

from sunerf.response.builder import InstrumentThroughput
from sunerf.response.providers.base import (
    OptionalProviderDependencyError,
    SourceFile,
    fetch_sources,
    require_source_paths,
)


AIAPY_VERSION = "0.12.1"
AIA_PROVIDER_VERSION = "aiapy-0.12.1-aia-v8-v10-reference"
AIA_REFERENCE_EPOCH = "2010-03-24T00:00:00Z"
AIA_EVE_ABSOLUTE_REFERENCE_DATE = "2010-05-01"
AIA_CALIBRATION_VERSION = 10
AIA_CHANNEL_WAVELENGTHS = (94, 131, 171, 193, 211, 335)
AIA_CHANNELS = tuple(f"A{channel}" for channel in AIA_CHANNEL_WAVELENGTHS)
AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE = 0.02
# SolarSoft folds the AIA temperature responses over 25-413 Angstrom only
# (aia_bp_make_emiss.pro defaults).  The instrument file continues to 900
# Angstrom with extrapolated component efficiencies whose Al-filter leak was
# never validated against EVE; folding it with optically thin O III-O V and
# Ne VIII lines at 550-800 Angstrom moves the A335 response maximum from
# log T 6.4 (Fe XVI 335.4) to log T 5.3.
AIA_VALIDATED_WAVELENGTH_MAX_ANGSTROM = 413.0

_LMSAL_BACKUP_COMMIT = "ef466df3faa4699ee1c8dec6b26f8449c8e41865"
_LMSAL_BACKUP_ROOT = (
    "https://raw.githubusercontent.com/LM-SAL/backup-files/"
    f"{_LMSAL_BACKUP_COMMIT}/static/sdo/aia/response"
)
_SSW_RESPONSE_ROOT = "https://hesperia.gsfc.nasa.gov/ssw/sdo/aia/response"

AIA_INSTRUMENT_SOURCE = SourceFile(
    key="instrument",
    filename="aia_V8_all_fullinst.genx",
    url=f"{_LMSAL_BACKUP_ROOT}/aia_V8_all_fullinst.genx",
    sha256="3940648e6b02876c45a9893f40806bbcc50baa994ae3fa2d95148916988426dd",
    size_bytes=5_426_568,
    version="V8",
    description=(
        "AIA EUV component efficiencies, effective areas, detector gain, "
        "crosstalk inputs, and plate scale; byte-identical for AIA V9/V10"
    ),
    reference_url=f"{_SSW_RESPONSE_ROOT}/V10_release_notes.txt",
)

AIA_CORRECTION_SOURCE = SourceFile(
    key="correction_table",
    filename="aia_V10_20201119_190000_response_table.txt",
    url=f"{_LMSAL_BACKUP_ROOT}/aia_V10_20201119_190000_response_table.txt",
    sha256="0a3f2db39d05c44185f6fdeec928089fb55d1ce1e0a805145050c6356cbc6e98",
    size_bytes=17_400,
    version="V10-20201119_190000",
    description="AIA V10 EVE/FISM degradation and effective-area correction table",
    reference_url=f"{_SSW_RESPONSE_ROOT}/V10_release_notes.txt",
)

AIA_THROUGHPUT_SOURCES = (AIA_INSTRUMENT_SOURCE, AIA_CORRECTION_SOURCE)


def fetch_aia_sources(
    raw_dir: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
):
    """Fetch and verify the immutable AIA wavelength-response inputs."""
    return fetch_sources(
        AIA_THROUGHPUT_SOURCES,
        raw_dir,
        force=force,
        timeout_seconds=timeout_seconds,
    )


def _load_aiapy():
    if sys.version_info < (3, 12):
        raise OptionalProviderDependencyError(
            "The pinned AIA response exporter requires Python >=3.12 and "
            f"aiapy=={AIAPY_VERSION}. Create the isolated response environment "
            "and install `sunerf[euv-prep]`."
        )
    try:
        installed_version = importlib.metadata.version("aiapy")
    except importlib.metadata.PackageNotFoundError as error:
        raise OptionalProviderDependencyError(
            "The AIA response exporter requires the optional dependency "
            f"aiapy=={AIAPY_VERSION}. Install `sunerf[euv-prep]` in a "
            "Python >=3.12 environment."
        ) from error
    if installed_version != AIAPY_VERSION:
        raise OptionalProviderDependencyError(
            "The AIA response exporter is validated only with "
            f"aiapy=={AIAPY_VERSION}; found aiapy=={installed_version}."
        )
    try:
        from aiapy.calibrate.utils import get_correction_table
        from aiapy.response import Channel
    except ImportError as error:  # pragma: no cover - broken installations
        raise OptionalProviderDependencyError(
            f"Could not import the required aiapy=={AIAPY_VERSION} response APIs."
        ) from error
    return Channel, get_correction_table


def _source_provenance(source: SourceFile) -> dict[str, object]:
    value = source.as_provenance()
    # Keep the required calibration fields at the section root while retaining
    # the complete immutable source description.
    return {
        "name": source.filename,
        "version": source.version,
        "sha256": source.sha256,
        "source": value,
    }


def export_aia_throughput(raw_dir: str | Path, output: str | Path) -> InstrumentThroughput:
    """Export the reference-epoch throughput for the six modeled AIA EUV bands.

    Images prepared by :func:`aiapy.calibrate.correct_degradation` are divided
    by sensitivity relative to 2010-03-24.  This exporter consequently
    evaluates the wavelength response at that exact epoch.  EVE absolute
    normalization, shared-telescope crosstalk, detector gain, and native pixel
    solid angle are all included explicitly.

    The exported grid ends at ``AIA_VALIDATED_WAVELENGTH_MAX_ANGSTROM`` to match
    the SolarSoft temperature-response convention.  The A131 peak in the A335
    row (and A94 in A304) is genuine shared-telescope crosstalk; it exceeds the
    335 Angstrom peak in DN only because the detector gain scales with photon
    energy.
    """
    paths = require_source_paths(AIA_THROUGHPUT_SOURCES, raw_dir)
    Channel, get_correction_table = _load_aiapy()
    correction_table = get_correction_table(paths["correction_table"])
    reference_time = Time(AIA_REFERENCE_EPOCH, scale="utc")

    output_unit = u.cm**2 * u.DN * u.sr / (u.ph * u.pix)
    wavelength_angstrom: np.ndarray | None = None
    throughput_rows = []
    native_pixel_solid_angle_sr: float | None = None

    for nominal_wavelength in AIA_CHANNEL_WAVELENGTHS:
        channel = Channel(
            nominal_wavelength * u.AA,
            instrument_file=paths["instrument"],
        )
        channel_wavelength = np.asarray(
            channel.wavelength.to_value(u.AA), dtype=np.float64
        )
        if channel_wavelength.ndim != 1 or channel_wavelength.size < 2:
            raise ValueError(
                f"AIA {nominal_wavelength} wavelength grid is not one-dimensional"
            )
        if not np.isfinite(channel_wavelength).all() or np.any(
            np.diff(channel_wavelength) <= 0
        ):
            raise ValueError(
                f"AIA {nominal_wavelength} wavelength grid is not finite and increasing"
            )
        if wavelength_angstrom is None:
            wavelength_angstrom = channel_wavelength
        elif not np.array_equal(channel_wavelength, wavelength_angstrom):
            raise ValueError("AIA channels do not share an identical wavelength grid")
        # Half a native 0.1 Angstrom step absorbs the float32 grid rounding.
        validated = channel_wavelength <= AIA_VALIDATED_WAVELENGTH_MAX_ANGSTROM + 0.05
        if validated.sum() < 2:
            raise ValueError(
                f"AIA {nominal_wavelength} wavelength grid has no validated support"
            )

        solid_angle = channel.plate_scale.to_value(u.sr / u.pix)
        solid_angle = float(np.asarray(solid_angle).item())
        if not np.isfinite(solid_angle) or solid_angle <= 0:
            raise ValueError(f"AIA {nominal_wavelength} plate scale is invalid")
        if native_pixel_solid_angle_sr is None:
            native_pixel_solid_angle_sr = solid_angle
        elif not np.isclose(
            solid_angle, native_pixel_solid_angle_sr, rtol=0, atol=0
        ):
            raise ValueError(
                "AIA channels do not share an identical native pixel solid angle"
            )

        wavelength_response = channel.wavelength_response(
            obstime=reference_time,
            include_eve_correction=True,
            include_crosstalk=True,
            correction_table=correction_table,
            calibration_version=AIA_CALIBRATION_VERSION,
        )
        values = np.asarray(
            (wavelength_response * channel.plate_scale).to_value(output_unit),
            dtype=np.float64,
        )
        if values.shape != channel_wavelength.shape:
            raise ValueError(
                f"AIA {nominal_wavelength} throughput shape {values.shape} does not "
                f"match wavelength shape {channel_wavelength.shape}"
            )
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(
                f"AIA {nominal_wavelength} throughput is not finite and non-negative"
            )
        throughput_rows.append(values[validated])

    assert wavelength_angstrom is not None  # fixed non-empty channel list
    assert native_pixel_solid_angle_sr is not None
    source_maximum_angstrom = float(wavelength_angstrom[-1])
    wavelength_angstrom = wavelength_angstrom[validated]
    provenance = {
        "instrument": {
            "name": "SDO/AIA",
            "spacecraft": "SDO",
            "detector": "AIA",
        },
        "provider": {"name": "aiapy", "version": AIAPY_VERSION},
        "calibration": _source_provenance(AIA_INSTRUMENT_SOURCE),
        "degradation": {
            **_source_provenance(AIA_CORRECTION_SOURCE),
            "algorithm": "aiapy.calibrate.degradation",
            "calibration_version": AIA_CALIBRATION_VERSION,
            "reference_epoch": AIA_REFERENCE_EPOCH,
            "factor_at_reference_epoch": 1.0,
        },
        "eve_normalization": {
            "included": True,
            "algorithm": "aiapy.response.Channel.eve_correction",
            "absolute_reference_date": AIA_EVE_ABSOLUTE_REFERENCE_DATE,
        },
        "crosstalk": {
            "included": True,
            "channel_pairs": [["A94", "A304"], ["A131", "A335"]],
        },
        "detector_gain": {"included": True, "output": "DN per detected photon"},
        "channels": list(AIA_CHANNELS),
        "wavelength_grid": {
            "coordinate": "vacuum_angstrom",
            "minimum_angstrom": float(wavelength_angstrom[0]),
            "maximum_angstrom": float(wavelength_angstrom[-1]),
            "count": int(wavelength_angstrom.size),
            "source_maximum_angstrom": source_maximum_angstrom,
            "truncation": "solarsoft_aia_bp_make_emiss_default_range",
        },
        "sensitivity_convention": "reference_epoch",
        "radiometry": {
            "measurement_semantics": "per_native_pixel",
            "native_pixel_solid_angle_sr": native_pixel_solid_angle_sr,
            "native_pixel_solid_angle_relative_tolerance": (
                AIA_NATIVE_PIXEL_SOLID_ANGLE_RELATIVE_TOLERANCE
            ),
        },
    }
    artifact = InstrumentThroughput(
        channels=AIA_CHANNELS,
        wavelength_angstrom=wavelength_angstrom,
        throughput=np.stack(throughput_rows, axis=0),
        throughput_unit=output_unit.to_string(),
        calibration_epoch=AIA_REFERENCE_EPOCH,
        provenance=provenance,
    )
    artifact.save(output)
    return artifact


__all__ = [
    "AIA_CALIBRATION_VERSION",
    "AIA_CHANNELS",
    "AIA_CHANNEL_WAVELENGTHS",
    "AIA_CORRECTION_SOURCE",
    "AIA_INSTRUMENT_SOURCE",
    "AIA_PROVIDER_VERSION",
    "AIA_REFERENCE_EPOCH",
    "AIA_THROUGHPUT_SOURCES",
    "AIA_VALIDATED_WAVELENGTH_MAX_ANGSTROM",
    "export_aia_throughput",
    "fetch_aia_sources",
]
