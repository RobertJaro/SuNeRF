"""Self-describing EUV temperature-response artifacts.

The on-disk representation intentionally remains a small NumPy archive so it
can be loaded without network access or a database runtime. Every scientific
assumption required to interpret the array is part of the validated schema.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u

from sunerf.configuration import canonical_channel_id


RESPONSE_SCHEMA = "sunerf.euv-temperature-response"
RESPONSE_SCHEMA_VERSION = 1
EMISSION_MEASURE_CONVENTIONS = frozenset({"ne2", "ne_nh"})


def _scalar_string(value: np.ndarray, name: str) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    return str(value.reshape(()).item())


def _provenance_without_response_id(provenance: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dict(provenance)
    provenance.pop("response_id", None)
    return provenance


def _semantic_response_id(
    *,
    schema_version: int,
    channels: Sequence[str],
    log_temperature,
    log_density,
    response,
    response_unit: str,
    emission_measure_convention: str,
    provenance: Mapping[str, Any],
) -> str:
    """Return a canonical ID over response content, excluding the ID itself."""
    metadata = {
        "schema": RESPONSE_SCHEMA,
        "schema_version": int(schema_version),
        "channels": tuple(str(channel) for channel in channels),
        "response_unit": str(response_unit),
        "emission_measure_convention": str(emission_measure_convention),
        "provenance": _provenance_without_response_id(provenance),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    for name, value in (
        ("log_temperature", log_temperature),
        ("log_density", np.asarray([]) if log_density is None else log_density),
        ("response", response),
    ):
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


def atomic_savez_compressed(path, **arrays):
    """Atomically publish a compressed NPZ using a same-directory temp file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class ResponseArtifact:
    """An ordered instrumental response on temperature and density axes.

    ``response`` has shape ``(channel, log_temperature)`` when density
    independent and ``(channel, log_density, log_temperature)`` otherwise.
    """

    channels: tuple[str, ...]
    log_temperature: np.ndarray
    response: np.ndarray
    response_unit: str
    emission_measure_convention: str
    provenance: Mapping[str, Any]
    log_density: np.ndarray | None = None
    schema_version: int = RESPONSE_SCHEMA_VERSION

    def __post_init__(self):
        channels = tuple(str(channel) for channel in self.channels)
        log_temperature = np.asarray(self.log_temperature, dtype=np.float64)
        log_density = None if self.log_density is None else np.asarray(self.log_density, dtype=np.float64)
        response = np.asarray(self.response, dtype=np.float64)
        provenance = dict(self.provenance)

        if self.schema_version != RESPONSE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported response schema version {self.schema_version}; "
                f"expected {RESPONSE_SCHEMA_VERSION}"
            )
        if not channels or any(not channel for channel in channels):
            raise ValueError("channels must contain non-empty identifiers")
        if len(set(channels)) != len(channels):
            raise ValueError("channel identifiers must be unique and ordered")
        self._validate_axis(log_temperature, "log_temperature")
        if log_density is not None:
            self._validate_axis(log_density, "log_density", allow_single=True)

        expected_shape = (
            (len(channels), log_temperature.size)
            if log_density is None
            else (len(channels), log_density.size, log_temperature.size)
        )
        if response.shape != expected_shape:
            raise ValueError(f"response must have shape {expected_shape}; received {response.shape}")
        if not np.isfinite(response).all():
            raise ValueError("response contains non-finite values")
        if np.any(response < 0):
            raise ValueError("response values must be non-negative")
        if self.emission_measure_convention not in EMISSION_MEASURE_CONVENTIONS:
            raise ValueError(
                "emission_measure_convention must be one of "
                f"{sorted(EMISSION_MEASURE_CONVENTIONS)}"
            )
        if not isinstance(self.response_unit, str) or not self.response_unit.strip():
            raise ValueError("response_unit must be a non-empty Astropy-compatible unit string")
        try:
            parsed_response_unit = u.Unit(self.response_unit)
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid response_unit {self.response_unit!r}") from error
        try:
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be JSON serializable and finite") from error
        if not provenance:
            raise ValueError("provenance must record how the response was constructed")

        if provenance.get("adapter") == "legacy_npz":
            raise ValueError(
                "legacy response adapters are not supported; rebuild the response "
                "from a unified spectral emissivity"
            )
        if parsed_response_unit == u.dimensionless_unscaled:
            raise ValueError(
                "response artifacts must have a physical response_unit"
            )

        if self.emission_measure_convention == "ne_nh":
            ratio = provenance.get("hydrogen_to_electron_ratio")
            if not isinstance(ratio, (int, float)) or not np.isfinite(ratio) or not 0 < ratio <= 1:
                raise ValueError(
                    "ne_nh response provenance must contain a finite "
                    "hydrogen_to_electron_ratio in (0, 1]"
                )

        computed_response_id = _semantic_response_id(
            schema_version=self.schema_version,
            channels=channels,
            log_temperature=log_temperature,
            log_density=log_density,
            response=response,
            response_unit=self.response_unit,
            emission_measure_convention=self.emission_measure_convention,
            provenance=provenance,
        )
        stored_response_id = provenance.get("response_id")
        if stored_response_id is not None and stored_response_id != computed_response_id:
            raise ValueError(
                f"response_id verification failed: stored {stored_response_id!r}, "
                f"computed {computed_response_id!r}"
            )
        provenance["response_id"] = computed_response_id

        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "log_temperature", log_temperature)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "response", response)
        object.__setattr__(self, "provenance", provenance)

    @property
    def response_id(self) -> str:
        """Canonical identifier stored in and derived from the artifact."""
        return self.provenance["response_id"]

    @property
    def computed_response_id(self) -> str:
        return _semantic_response_id(
            schema_version=self.schema_version,
            channels=self.channels,
            log_temperature=self.log_temperature,
            log_density=self.log_density,
            response=self.response,
            response_unit=self.response_unit,
            emission_measure_convention=self.emission_measure_convention,
            provenance=self.provenance,
        )

    def verify_response_id(self) -> bool:
        if self.response_id != self.computed_response_id:
            raise ValueError(
                f"response_id verification failed: stored {self.response_id!r}, "
                f"computed {self.computed_response_id!r}"
            )
        return True

    def updated(self, **changes) -> "ResponseArtifact":
        """Create a scientifically modified artifact with a fresh response ID."""
        changes["provenance"] = _provenance_without_response_id(
            changes.get("provenance", self.provenance)
        )
        return replace(self, **changes)

    @staticmethod
    def _validate_axis(axis: np.ndarray, name: str, allow_single: bool = False):
        minimum_size = 1 if allow_single else 2
        if axis.ndim != 1 or axis.size < minimum_size:
            raise ValueError(f"{name} must be one-dimensional with at least {minimum_size} entries")
        if not np.isfinite(axis).all():
            raise ValueError(f"{name} contains non-finite values")
        if axis.size > 1 and np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} must be strictly increasing")

    def select_channels(self, channels: Sequence[str] | None) -> "ResponseArtifact":
        if channels is None:
            return self
        requested = tuple(str(channel) for channel in channels)
        requested_canonical = tuple(canonical_channel_id(channel) for channel in requested)
        if len(set(requested_canonical)) != len(requested_canonical):
            raise ValueError("requested channels must be unique")
        available = {}
        for channel in self.channels:
            canonical = canonical_channel_id(channel)
            if canonical in available:
                raise ValueError(
                    'response artifact channel aliases are ambiguous: '
                    f'{available[canonical]!r} and {channel!r}'
                )
            available[canonical] = channel
        missing = [
            channel for channel, canonical in zip(requested, requested_canonical)
            if canonical not in available
        ]
        if missing:
            raise KeyError(f"response artifact does not contain channels {missing}")
        resolved = tuple(available[canonical] for canonical in requested_canonical)
        indices = [self.channels.index(channel) for channel in resolved]
        return self.updated(channels=resolved, response=self.response[indices])

    def interpolate_temperature(self, target_log_temperature) -> "ResponseArtifact":
        """Interpolate linearly in log-temperature, with zero outside support."""
        target = np.asarray(target_log_temperature, dtype=np.float64)
        self._validate_axis(target, "target_log_temperature")
        flat = self.response.reshape(-1, self.log_temperature.size)
        interpolated = np.stack(
            [
                np.interp(
                    target,
                    self.log_temperature,
                    values,
                    left=0.0,
                    right=0.0,
                )
                for values in flat
            ],
            axis=0,
        )
        response = interpolated.reshape(*self.response.shape[:-1], target.size)
        return self.updated(log_temperature=target, response=response)

    def save(self, path):
        self.verify_response_id()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        provenance_json = json.dumps(
            dict(self.provenance), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        atomic_savez_compressed(
            path,
            schema=np.asarray(RESPONSE_SCHEMA),
            schema_version=np.asarray(self.schema_version, dtype=np.int64),
            channels=np.asarray(self.channels),
            log_temperature=self.log_temperature,
            log_density=(np.asarray([], dtype=np.float64) if self.log_density is None else self.log_density),
            response=self.response,
            response_unit=np.asarray(self.response_unit),
            emission_measure_convention=np.asarray(self.emission_measure_convention),
            provenance_json=np.asarray(provenance_json),
        )


def _load_versioned_archive(archive: np.lib.npyio.NpzFile) -> ResponseArtifact:
    required = {
        "schema",
        "schema_version",
        "channels",
        "log_temperature",
        "log_density",
        "response",
        "response_unit",
        "emission_measure_convention",
        "provenance_json",
    }
    missing = sorted(required.difference(archive.files))
    if missing:
        raise ValueError(f"response artifact is missing required fields {missing}")
    if _scalar_string(archive["schema"], "schema") != RESPONSE_SCHEMA:
        raise ValueError("file is not a SuNeRF EUV temperature-response artifact")
    try:
        provenance = json.loads(_scalar_string(archive["provenance_json"], "provenance_json"))
    except json.JSONDecodeError as error:
        raise ValueError("response provenance is not valid JSON") from error
    log_density = np.asarray(archive["log_density"], dtype=np.float64)
    return ResponseArtifact(
        channels=tuple(np.asarray(archive["channels"]).astype(str).tolist()),
        log_temperature=archive["log_temperature"],
        log_density=None if log_density.size == 0 else log_density,
        response=archive["response"],
        response_unit=_scalar_string(archive["response_unit"], "response_unit"),
        emission_measure_convention=_scalar_string(
            archive["emission_measure_convention"], "emission_measure_convention"
        ),
        provenance=provenance,
        schema_version=int(np.asarray(archive["schema_version"]).reshape(())),
    )


def load_response_artifact(
    path,
    *,
    channels: Sequence[str] | None = None,
) -> ResponseArtifact:
    """Load and validate a versioned SuNeRF temperature-response artifact."""
    from sunerf.resources import resolve_artifact_path

    path = resolve_artifact_path(path)
    with np.load(path, allow_pickle=False) as archive:
        if "schema" not in archive.files:
            raise ValueError("file is not a versioned SuNeRF temperature-response artifact")
        artifact = _load_versioned_archive(archive)
    return artifact.select_channels(channels)
