"""Provider-neutral construction of EUV temperature-response artifacts.

This module deliberately does not run CHIANTI or download calibration data.
Instead, it defines two small validated interchange formats:

* a wavelength-resolved spectral emissivity grid produced by any CHIANTI
  backend, and
* an ordered wavelength-resolved instrument-throughput table.

Folding those two inputs is deterministic, offline, unit checked, and records
the atomic, abundance, ionization, and calibration assumptions in the output
``ResponseArtifact``.  All channels are folded from the same emissivity grid,
which prevents accidental abundance or ionization-equilibrium mismatches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u

from sunerf.response.artifact import (
    EMISSION_MEASURE_CONVENTIONS,
    ResponseArtifact,
    atomic_savez_compressed,
)
from sunerf.response.numerics import trapezoid_node_weights


SPECTRAL_EMISSIVITY_SCHEMA = "sunerf.chianti-spectral-emissivity"
INSTRUMENT_THROUGHPUT_SCHEMA = "sunerf.instrument-throughput"
BUILD_INPUT_SCHEMA_VERSION = 1
FOLD_ALGORITHM_VERSION = 1
SENSITIVITY_CONVENTIONS = frozenset(
    {"reference_epoch", "native_epoch", "static_assumed"}
)
MEASUREMENT_SEMANTICS = frozenset({"surface_brightness", "per_native_pixel"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _scalar_string(value, name: str) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    return str(value.reshape(()).item())


def _strict_axis(values, name: str, *, positive: bool = False, allow_single: bool = False):
    values = np.asarray(values, dtype=np.float64)
    minimum_size = 1 if allow_single else 2
    if values.ndim != 1 or values.size < minimum_size:
        raise ValueError(f"{name} must be one-dimensional with at least {minimum_size} entries")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _json_mapping(value, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    value = dict(value)
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and JSON serializable") from error
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


def _require_string(mapping: Mapping[str, Any], key: str, context: str):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string")


def _require_sha256(mapping: Mapping[str, Any], key: str, context: str):
    _require_string(mapping, key, context)
    if not _SHA256_PATTERN.fullmatch(mapping[key].lower()):
        raise ValueError(f"{context}.{key} must be a 64-character SHA-256 digest")


def _require_section(provenance, name: str, fields: Sequence[str], *, sha256: bool = False):
    section = provenance.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"provenance.{name} must be a mapping")
    for field in fields:
        _require_string(section, field, f"provenance.{name}")
    if sha256:
        _require_sha256(section, "sha256", f"provenance.{name}")


def _validate_spectral_provenance(provenance):
    provenance = _json_mapping(provenance, "spectral provenance")
    _require_section(provenance, "provider", ("name", "version"))
    for section in ("atomic_database", "abundance", "ionization_equilibrium"):
        _require_section(provenance, section, ("name", "version", "sha256"), sha256=True)
    components = provenance.get("emission_components")
    if (
        not isinstance(components, (list, tuple))
        or not components
        or any(not isinstance(component, str) or not component for component in components)
    ):
        raise ValueError(
            "spectral provenance.emission_components must be a non-empty list of strings"
        )
    return provenance


def _validate_throughput_provenance(provenance):
    provenance = _json_mapping(provenance, "throughput provenance")
    _require_section(provenance, "instrument", ("name",))
    _require_section(provenance, "provider", ("name", "version"))
    _require_section(provenance, "calibration", ("name", "version", "sha256"), sha256=True)
    sensitivity_convention = provenance.get("sensitivity_convention")
    if sensitivity_convention not in SENSITIVITY_CONVENTIONS:
        raise ValueError(
            "throughput provenance.sensitivity_convention must be one of "
            f"{sorted(SENSITIVITY_CONVENTIONS)}"
        )
    radiometry = provenance.get("radiometry")
    if not isinstance(radiometry, Mapping):
        raise ValueError("throughput provenance.radiometry must be a mapping")
    measurement_semantics = radiometry.get("measurement_semantics")
    if measurement_semantics not in MEASUREMENT_SEMANTICS:
        raise ValueError(
            "throughput provenance.radiometry.measurement_semantics must be one of "
            f"{sorted(MEASUREMENT_SEMANTICS)}"
        )
    if measurement_semantics == "per_native_pixel":
        solid_angle = radiometry.get("native_pixel_solid_angle_sr")
        tolerance = radiometry.get("native_pixel_solid_angle_relative_tolerance")
        if (
            not isinstance(solid_angle, (int, float))
            or not np.isfinite(solid_angle)
            or solid_angle <= 0
        ):
            raise ValueError(
                "per_native_pixel throughput provenance requires a finite positive "
                "radiometry.native_pixel_solid_angle_sr"
            )
        if (
            not isinstance(tolerance, (int, float))
            or not np.isfinite(tolerance)
            or not 0 <= tolerance < 1
        ):
            raise ValueError(
                "per_native_pixel throughput provenance requires an explicit finite "
                "radiometry.native_pixel_solid_angle_relative_tolerance in [0, 1)"
            )
    return provenance


def _canonical_epoch(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("calibration_epoch must be a non-empty ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("calibration_epoch must be a valid ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("calibration_epoch must include an explicit UTC offset")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _semantic_sha256(schema: str, metadata: Mapping[str, Any], arrays) -> str:
    """Hash scientific content independent of filename and NPZ compression."""
    digest = hashlib.sha256()
    digest.update(schema.encode("utf-8"))
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    for name, value in arrays:
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SpectralEmissivityGrid:
    """A shared CHIANTI spectral emissivity grid.

    ``emissivity`` is shaped ``(temperature, wavelength)`` or
    ``(density, temperature, wavelength)``. Wavelength coordinates are vacuum
    Angstrom and both logarithmic axes are base-10 in K and cm^-3 respectively.
    """

    wavelength_angstrom: np.ndarray
    log_temperature: np.ndarray
    emissivity: np.ndarray
    emissivity_unit: str
    emission_measure_convention: str
    provenance: Mapping[str, Any]
    log_density: np.ndarray | None = None
    schema_version: int = BUILD_INPUT_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != BUILD_INPUT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported spectral-emissivity schema version {self.schema_version}"
            )
        wavelength = _strict_axis(
            self.wavelength_angstrom, "wavelength_angstrom", positive=True
        )
        log_temperature = _strict_axis(self.log_temperature, "log_temperature")
        log_density = (
            None
            if self.log_density is None
            else _strict_axis(self.log_density, "log_density", allow_single=True)
        )
        emissivity = np.asarray(self.emissivity, dtype=np.float64)
        expected_shape = (
            (log_temperature.size, wavelength.size)
            if log_density is None
            else (log_density.size, log_temperature.size, wavelength.size)
        )
        if emissivity.shape != expected_shape:
            raise ValueError(
                f"emissivity must have shape {expected_shape}; received {emissivity.shape}"
            )
        if not np.isfinite(emissivity).all() or np.any(emissivity < 0):
            raise ValueError("emissivity must be finite and non-negative")
        try:
            u.Unit(self.emissivity_unit)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid emissivity_unit {self.emissivity_unit!r}") from error
        if self.emission_measure_convention not in EMISSION_MEASURE_CONVENTIONS:
            raise ValueError(
                "emission_measure_convention must be either 'ne2' or 'ne_nh'"
            )
        provenance = _validate_spectral_provenance(self.provenance)
        if self.emission_measure_convention == "ne_nh":
            ratio = provenance.get("hydrogen_to_electron_ratio")
            if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError(
                    "ne_nh spectral provenance must contain a finite "
                    "hydrogen_to_electron_ratio in (0, 1]"
                )

        object.__setattr__(self, "wavelength_angstrom", wavelength)
        object.__setattr__(self, "log_temperature", log_temperature)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "emissivity", emissivity)
        object.__setattr__(self, "provenance", provenance)

    @property
    def content_sha256(self):
        metadata = {
            "schema_version": self.schema_version,
            "emissivity_unit": self.emissivity_unit,
            "emission_measure_convention": self.emission_measure_convention,
            "provenance": dict(self.provenance),
        }
        arrays = [
            ("wavelength_angstrom", self.wavelength_angstrom),
            ("log_temperature", self.log_temperature),
            ("log_density", np.asarray([]) if self.log_density is None else self.log_density),
            ("emissivity", self.emissivity),
        ]
        return _semantic_sha256(SPECTRAL_EMISSIVITY_SCHEMA, metadata, arrays)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance = json.dumps(
            dict(self.provenance), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        atomic_savez_compressed(
            path,
            schema=np.asarray(SPECTRAL_EMISSIVITY_SCHEMA),
            schema_version=np.asarray(self.schema_version, dtype=np.int64),
            wavelength_angstrom=self.wavelength_angstrom,
            log_temperature=self.log_temperature,
            log_density=(np.asarray([], dtype=np.float64) if self.log_density is None else self.log_density),
            emissivity=self.emissivity,
            emissivity_unit=np.asarray(self.emissivity_unit),
            emission_measure_convention=np.asarray(self.emission_measure_convention),
            provenance_json=np.asarray(provenance),
        )


@dataclass(frozen=True)
class InstrumentThroughput:
    """Ordered wavelength-dependent instrumental throughput for several channels."""

    channels: tuple[str, ...]
    wavelength_angstrom: np.ndarray
    throughput: np.ndarray
    throughput_unit: str
    calibration_epoch: str
    provenance: Mapping[str, Any]
    schema_version: int = BUILD_INPUT_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != BUILD_INPUT_SCHEMA_VERSION:
            raise ValueError(f"unsupported throughput schema version {self.schema_version}")
        channels = tuple(str(channel) for channel in self.channels)
        if not channels or any(not channel for channel in channels):
            raise ValueError("throughput channels must be non-empty strings")
        if len(set(channels)) != len(channels):
            raise ValueError("throughput channels must be unique and ordered")
        wavelength = _strict_axis(
            self.wavelength_angstrom, "wavelength_angstrom", positive=True
        )
        throughput = np.asarray(self.throughput, dtype=np.float64)
        expected_shape = (len(channels), wavelength.size)
        if throughput.shape != expected_shape:
            raise ValueError(
                f"throughput must have shape {expected_shape}; received {throughput.shape}"
            )
        if not np.isfinite(throughput).all() or np.any(throughput < 0):
            raise ValueError("throughput must be finite and non-negative")
        if np.any(np.max(throughput, axis=1) <= 0):
            raise ValueError("every throughput channel must contain positive support")
        try:
            throughput_unit = u.Unit(self.throughput_unit)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid throughput_unit {self.throughput_unit!r}") from error
        calibration_epoch = _canonical_epoch(self.calibration_epoch)
        provenance = _validate_throughput_provenance(self.provenance)
        pixel_power = sum(
            power
            for base, power in zip(throughput_unit.bases, throughput_unit.powers)
            if base == u.pix
        )
        measurement_semantics = provenance["radiometry"]["measurement_semantics"]
        if measurement_semantics == "per_native_pixel" and pixel_power != -1:
            raise ValueError(
                "per_native_pixel throughput_unit must contain exactly pix-1"
            )
        if measurement_semantics == "surface_brightness" and pixel_power != 0:
            raise ValueError(
                "surface_brightness throughput_unit must not contain a pixel unit"
            )

        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "wavelength_angstrom", wavelength)
        object.__setattr__(self, "throughput", throughput)
        object.__setattr__(self, "calibration_epoch", calibration_epoch)
        object.__setattr__(self, "provenance", provenance)

    @property
    def content_sha256(self):
        metadata = {
            "schema_version": self.schema_version,
            "channels": self.channels,
            "throughput_unit": self.throughput_unit,
            "calibration_epoch": self.calibration_epoch,
            "provenance": dict(self.provenance),
        }
        return _semantic_sha256(
            INSTRUMENT_THROUGHPUT_SCHEMA,
            metadata,
            [
                ("wavelength_angstrom", self.wavelength_angstrom),
                ("throughput", self.throughput),
            ],
        )

    def select_channels(self, channels: Sequence[str] | None):
        if channels is None:
            return self
        requested = tuple(str(channel) for channel in channels)
        if not requested:
            raise ValueError("requested channels must not be empty")
        if len(set(requested)) != len(requested):
            raise ValueError("requested channels must be unique")
        missing = [channel for channel in requested if channel not in self.channels]
        if missing:
            raise KeyError(f"throughput does not contain channels {missing}")
        indices = [self.channels.index(channel) for channel in requested]
        return replace(self, channels=requested, throughput=self.throughput[indices])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance = json.dumps(
            dict(self.provenance), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        atomic_savez_compressed(
            path,
            schema=np.asarray(INSTRUMENT_THROUGHPUT_SCHEMA),
            schema_version=np.asarray(self.schema_version, dtype=np.int64),
            channels=np.asarray(self.channels),
            wavelength_angstrom=self.wavelength_angstrom,
            throughput=self.throughput,
            throughput_unit=np.asarray(self.throughput_unit),
            calibration_epoch=np.asarray(self.calibration_epoch),
            provenance_json=np.asarray(provenance),
        )


def _load_json(archive, name: str):
    try:
        return json.loads(_scalar_string(archive[name], name))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not valid JSON") from error


def _require_archive_fields(archive, required, artifact_name: str):
    missing = sorted(set(required).difference(archive.files))
    if missing:
        raise ValueError(f"{artifact_name} is missing required fields {missing}")


def load_spectral_emissivity(path) -> SpectralEmissivityGrid:
    from sunerf.resources import resolve_artifact_path

    path = resolve_artifact_path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema", "schema_version", "wavelength_angstrom", "log_temperature",
            "log_density", "emissivity", "emissivity_unit",
            "emission_measure_convention", "provenance_json",
        }
        _require_archive_fields(archive, required, "spectral-emissivity artifact")
        if _scalar_string(archive["schema"], "schema") != SPECTRAL_EMISSIVITY_SCHEMA:
            raise ValueError("file is not a SuNeRF CHIANTI spectral-emissivity artifact")
        log_density = np.asarray(archive["log_density"], dtype=np.float64)
        return SpectralEmissivityGrid(
            wavelength_angstrom=archive["wavelength_angstrom"],
            log_temperature=archive["log_temperature"],
            log_density=None if log_density.size == 0 else log_density,
            emissivity=archive["emissivity"],
            emissivity_unit=_scalar_string(archive["emissivity_unit"], "emissivity_unit"),
            emission_measure_convention=_scalar_string(
                archive["emission_measure_convention"], "emission_measure_convention"
            ),
            provenance=_load_json(archive, "provenance_json"),
            schema_version=int(np.asarray(archive["schema_version"]).reshape(())),
        )


def load_instrument_throughput(path) -> InstrumentThroughput:
    from sunerf.resources import resolve_artifact_path

    path = resolve_artifact_path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema", "schema_version", "channels", "wavelength_angstrom", "throughput",
            "throughput_unit", "calibration_epoch", "provenance_json",
        }
        _require_archive_fields(archive, required, "instrument-throughput artifact")
        if _scalar_string(archive["schema"], "schema") != INSTRUMENT_THROUGHPUT_SCHEMA:
            raise ValueError("file is not a SuNeRF instrument-throughput artifact")
        return InstrumentThroughput(
            channels=tuple(np.asarray(archive["channels"]).astype(str).tolist()),
            wavelength_angstrom=archive["wavelength_angstrom"],
            throughput=archive["throughput"],
            throughput_unit=_scalar_string(archive["throughput_unit"], "throughput_unit"),
            calibration_epoch=_scalar_string(
                archive["calibration_epoch"], "calibration_epoch"
            ),
            provenance=_load_json(archive, "provenance_json"),
            schema_version=int(np.asarray(archive["schema_version"]).reshape(())),
        )


def fold_temperature_response(
    spectral_emissivity: SpectralEmissivityGrid,
    instrument_throughput: InstrumentThroughput,
    *,
    channels: Sequence[str] | None = None,
    response_unit: str | None = None,
) -> ResponseArtifact:
    """Fold one shared emissivity cube through ordered channel throughputs.

    Throughput is interpolated linearly onto the emissivity wavelength axis and
    is exactly zero outside its measured support. Integration uses trapezoidal
    node weights on the potentially nonuniform wavelength grid. No response or
    channel normalization is applied.
    """
    if not isinstance(spectral_emissivity, SpectralEmissivityGrid):
        raise TypeError("spectral_emissivity must be a SpectralEmissivityGrid")
    if not isinstance(instrument_throughput, InstrumentThroughput):
        raise TypeError("instrument_throughput must be an InstrumentThroughput")
    throughput = instrument_throughput.select_channels(channels)

    wavelength = spectral_emissivity.wavelength_angstrom
    overlap_min = max(wavelength[0], throughput.wavelength_angstrom[0])
    overlap_max = min(wavelength[-1], throughput.wavelength_angstrom[-1])
    if overlap_max <= overlap_min:
        raise ValueError("spectral emissivity and throughput wavelength axes do not overlap")

    interpolated_throughput = np.stack(
        [
            np.interp(
                wavelength,
                throughput.wavelength_angstrom,
                channel_throughput,
                left=0.0,
                right=0.0,
            )
            for channel_throughput in throughput.throughput
        ],
        axis=0,
    )
    wavelength_weights = trapezoid_node_weights(wavelength)
    supported = (interpolated_throughput * wavelength_weights).sum(axis=1)
    if np.any(supported <= 0):
        missing = [
            channel for channel, integral in zip(throughput.channels, supported) if integral <= 0
        ]
        raise ValueError(f"throughput channels have no support on the emissivity grid: {missing}")

    if spectral_emissivity.log_density is None:
        response = np.einsum(
            "tw,cw,w->ct",
            spectral_emissivity.emissivity,
            interpolated_throughput,
            wavelength_weights,
            optimize=True,
        )
    else:
        response = np.einsum(
            "dtw,cw,w->cdt",
            spectral_emissivity.emissivity,
            interpolated_throughput,
            wavelength_weights,
            optimize=True,
        )

    native_unit = (
        u.Unit(spectral_emissivity.emissivity_unit)
        * u.Unit(throughput.throughput_unit)
        * u.AA
    )
    if response_unit is None:
        output_unit = native_unit
    else:
        try:
            output_unit = u.Unit(response_unit)
            response = response * (1.0 * native_unit).to_value(output_unit)
        except (TypeError, ValueError, u.UnitConversionError) as error:
            raise ValueError(
                f"requested response_unit {response_unit!r} is incompatible with {native_unit}"
            ) from error

    provenance = {
        "builder": {
            "name": "sunerf.response.builder",
            "algorithm_version": FOLD_ALGORITHM_VERSION,
        },
        "spectral_emissivity": {
            "schema": SPECTRAL_EMISSIVITY_SCHEMA,
            "schema_version": spectral_emissivity.schema_version,
            "content_sha256": spectral_emissivity.content_sha256,
            **dict(spectral_emissivity.provenance),
        },
        "instrument_throughput": {
            "schema": INSTRUMENT_THROUGHPUT_SCHEMA,
            "schema_version": throughput.schema_version,
            "content_sha256": throughput.content_sha256,
            "calibration_epoch": throughput.calibration_epoch,
            **dict(throughput.provenance),
        },
        "fold": {
            "wavelength_coordinate": "vacuum_angstrom",
            "throughput_interpolation": "linear_zero_outside_support",
            "wavelength_quadrature": "trapezoidal_node_weights",
            "normalization": "none",
        },
        # Promote these two fields so preparation/runtime do not need to know
        # the provider-specific nesting to reject degradation double counting.
        "sensitivity_convention": throughput.provenance["sensitivity_convention"],
        "calibration_epoch": throughput.calibration_epoch,
        "measurement_semantics": throughput.provenance["radiometry"][
            "measurement_semantics"
        ],
    }
    if provenance["measurement_semantics"] == "per_native_pixel":
        provenance["native_pixel_solid_angle_sr"] = throughput.provenance[
            "radiometry"
        ]["native_pixel_solid_angle_sr"]
        provenance["native_pixel_solid_angle_relative_tolerance"] = (
            throughput.provenance["radiometry"][
                "native_pixel_solid_angle_relative_tolerance"
            ]
        )
    if spectral_emissivity.emission_measure_convention == "ne_nh":
        provenance["hydrogen_to_electron_ratio"] = spectral_emissivity.provenance[
            "hydrogen_to_electron_ratio"
        ]
    return ResponseArtifact(
        channels=throughput.channels,
        log_temperature=spectral_emissivity.log_temperature,
        log_density=spectral_emissivity.log_density,
        response=response,
        response_unit=output_unit.to_string(),
        emission_measure_convention=spectral_emissivity.emission_measure_convention,
        provenance=provenance,
    )


def build_response_artifact(
    spectral_emissivity_file,
    instrument_throughput_file,
    output_file=None,
    *,
    channels: Sequence[str] | None = None,
    response_unit: str | None = None,
) -> ResponseArtifact:
    """Load validated build inputs, fold them, and optionally write artifact v1."""
    artifact = fold_temperature_response(
        load_spectral_emissivity(spectral_emissivity_file),
        load_instrument_throughput(instrument_throughput_file),
        channels=channels,
        response_unit=response_unit,
    )
    if output_file is not None:
        artifact.save(output_file)
    return artifact


def input_schema_description() -> dict[str, Any]:
    """Return a machine-readable template for the two offline build inputs.

    This describes the provider boundary; it intentionally contains no atomic
    calculation or instrument calibration defaults.
    """
    return {
        "spectral_emissivity_npz": {
            "schema": SPECTRAL_EMISSIVITY_SCHEMA,
            "schema_version": BUILD_INPUT_SCHEMA_VERSION,
            "axes": {
                "wavelength_angstrom": "strictly increasing vacuum wavelength nodes",
                "log_temperature": "strictly increasing log10(K) nodes",
                "log_density": "empty or strictly increasing log10(cm^-3) nodes",
            },
            "values": {
                "emissivity": "(temperature,wavelength) or (density,temperature,wavelength)",
                "emissivity_unit": "Astropy-compatible spectral-emissivity unit",
                "emission_measure_convention": sorted(EMISSION_MEASURE_CONVENTIONS),
            },
            "required_provenance": {
                "provider": {"name": "<backend>", "version": "<version>"},
                "atomic_database": {
                    "name": "CHIANTI",
                    "version": "<database-version>",
                    "sha256": "<64 hex characters>",
                },
                "abundance": {
                    "name": "<explicit-abundance-file>",
                    "version": "<version>",
                    "sha256": "<64 hex characters>",
                },
                "ionization_equilibrium": {
                    "name": "<explicit-ionization-file>",
                    "version": "<version>",
                    "sha256": "<64 hex characters>",
                },
                "emission_components": ["lines", "free_free", "free_bound", "two_photon"],
                "hydrogen_to_electron_ratio": "required finite scalar in (0,1] for ne_nh",
            },
        },
        "instrument_throughput_npz": {
            "schema": INSTRUMENT_THROUGHPUT_SCHEMA,
            "schema_version": BUILD_INPUT_SCHEMA_VERSION,
            "axes": {
                "channels": "ordered unique channel identifiers",
                "wavelength_angstrom": "strictly increasing vacuum wavelength nodes",
            },
            "values": {
                "throughput": "non-negative (channel,wavelength) array",
                "throughput_unit": "Astropy-compatible unit",
                "calibration_epoch": "timezone-aware ISO-8601 epoch",
            },
            "required_provenance": {
                "instrument": {"name": "<instrument/detector>"},
                "provider": {"name": "<calibration-provider>", "version": "<version>"},
                "calibration": {
                    "name": "<calibration-artifact>",
                    "version": "<version>",
                    "sha256": "<64 hex characters>",
                },
                "sensitivity_convention": sorted(SENSITIVITY_CONVENTIONS),
                "radiometry": {
                    "measurement_semantics": sorted(MEASUREMENT_SEMANTICS),
                    "native_pixel_solid_angle_sr": (
                        "required finite positive scalar for per_native_pixel"
                    ),
                    "native_pixel_solid_angle_relative_tolerance": (
                        "required finite scalar in [0,1) for per_native_pixel"
                    ),
                },
            },
        },
        "fold_contract": {
            "normalization": "none",
            "throughput_outside_support": "zero",
            "quadrature": "nonuniform trapezoidal node weights",
        },
    }

