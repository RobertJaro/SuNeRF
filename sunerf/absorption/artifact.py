"""Validated, training-independent EUV photoionization absorption bundles.

The bundle is the only contract between atomic-data preparation and SuNeRF
training.  It contains no executable provider objects and can be loaded using
NumPy alone.  Every row is identified by both instrument and channel so that
nominally equal wavelengths from different telescopes remain distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sunerf.configuration import canonical_channel_id
from sunerf.response.artifact import atomic_savez_compressed


ABSORPTION_BUNDLE_SCHEMA = "sunerf.euv-photoionization-absorption"
ABSORPTION_BUNDLE_SCHEMA_VERSION = 1
ABSORPTION_SPECIES = ("H_I", "He_I", "He_II")


def _scalar_string(value, name: str) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar string")
    return str(value.reshape(()).item())


def _without_bundle_id(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(provenance)
    value.pop("bundle_id", None)
    return value


def _semantic_bundle_id(bundle: "AbsorptionBundle") -> str:
    metadata = {
        "schema": ABSORPTION_BUNDLE_SCHEMA,
        "schema_version": bundle.schema_version,
        "species": bundle.species,
        "instrument_keys": bundle.instrument_keys,
        "channels": bundle.channels,
        "cross_section_unit": "cm2",
        "density_convention": "total_hydrogen_nuclei_per_cm3",
        "temperature_convention": "log10_K",
        "provenance": _without_bundle_id(bundle.provenance),
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    for name, values in (
        ("log_temperature", bundle.log_temperature),
        ("ion_fraction", bundle.ion_fraction),
        ("electron_per_hydrogen", bundle.electron_per_hydrogen),
        ("abundance_per_hydrogen", bundle.abundance_per_hydrogen),
        ("effective_cross_section_cm2", bundle.effective_cross_section_cm2),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class AbsorptionBundle:
    """Channel-effective H/He opacity data and equilibrium ion fractions.

    ``ion_fraction`` has shape ``(species, temperature)`` and stores the
    fraction of each element in H I, He I, and He II.  Element abundances are
    kept separately in ``abundance_per_hydrogen``.  Cross sections have shape
    ``(instrument-channel row, species)``.
    """

    species: tuple[str, ...]
    log_temperature: np.ndarray
    ion_fraction: np.ndarray
    electron_per_hydrogen: np.ndarray
    abundance_per_hydrogen: np.ndarray
    instrument_keys: tuple[str, ...]
    channels: tuple[str, ...]
    effective_cross_section_cm2: np.ndarray
    provenance: Mapping[str, Any]
    schema_version: int = ABSORPTION_BUNDLE_SCHEMA_VERSION

    def __post_init__(self):
        species = tuple(str(value) for value in self.species)
        instrument_keys = tuple(str(value) for value in self.instrument_keys)
        channels = tuple(str(value) for value in self.channels)
        log_temperature = np.asarray(self.log_temperature, dtype=np.float64)
        ion_fraction = np.asarray(self.ion_fraction, dtype=np.float64)
        electron_per_hydrogen = np.asarray(
            self.electron_per_hydrogen, dtype=np.float64
        )
        abundance = np.asarray(self.abundance_per_hydrogen, dtype=np.float64)
        cross_section = np.asarray(self.effective_cross_section_cm2, dtype=np.float64)
        provenance = dict(self.provenance)

        if self.schema_version != ABSORPTION_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported absorption bundle version {self.schema_version}; "
                f"expected {ABSORPTION_BUNDLE_SCHEMA_VERSION}"
            )
        if species != ABSORPTION_SPECIES:
            raise ValueError(
                f"species must be exactly the ordered tuple {ABSORPTION_SPECIES}"
            )
        if (
            log_temperature.ndim != 1
            or log_temperature.size < 2
            or not np.isfinite(log_temperature).all()
            or np.any(np.diff(log_temperature) <= 0)
        ):
            raise ValueError("log_temperature must be finite and strictly increasing")
        expected_fraction_shape = (len(species), log_temperature.size)
        if ion_fraction.shape != expected_fraction_shape:
            raise ValueError(
                f"ion_fraction must have shape {expected_fraction_shape}"
            )
        if (
            not np.isfinite(ion_fraction).all()
            or np.any(ion_fraction < 0)
            or np.any(ion_fraction > 1)
        ):
            raise ValueError("ion_fraction values must be finite and lie in [0, 1]")
        if electron_per_hydrogen.shape != (log_temperature.size,):
            raise ValueError("electron_per_hydrogen must contain one value per temperature")
        if (
            not np.isfinite(electron_per_hydrogen).all()
            or np.any(electron_per_hydrogen <= 0)
        ):
            raise ValueError("electron_per_hydrogen must be finite and strictly positive")
        if abundance.shape != (len(species),):
            raise ValueError("abundance_per_hydrogen must contain one value per species")
        if not np.isfinite(abundance).all() or np.any(abundance <= 0):
            raise ValueError("abundance_per_hydrogen must be finite and positive")
        if not np.isclose(abundance[0], 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("H_I abundance_per_hydrogen must equal one")
        if not np.isclose(abundance[1], abundance[2], rtol=0.0, atol=1e-12):
            raise ValueError("He_I and He_II must use one shared helium abundance")
        if not instrument_keys or len(instrument_keys) != len(channels):
            raise ValueError("instrument_keys and channels must have the same non-zero length")
        identities = tuple(
            (instrument, canonical_channel_id(channel))
            for instrument, channel in zip(instrument_keys, channels)
        )
        if len(set(identities)) != len(identities):
            raise ValueError("instrument/channel identities must be unique")
        expected_cross_section_shape = (len(channels), len(species))
        if cross_section.shape != expected_cross_section_shape:
            raise ValueError(
                f"effective_cross_section_cm2 must have shape {expected_cross_section_shape}"
            )
        if not np.isfinite(cross_section).all() or np.any(cross_section < 0):
            raise ValueError("effective cross sections must be finite and non-negative")
        if np.any(np.max(cross_section, axis=1) <= 0):
            raise ValueError("every instrument/channel row must have non-zero opacity")
        try:
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must be finite and JSON serializable") from error
        if not provenance:
            raise ValueError("provenance must describe the bundle inputs")

        object.__setattr__(self, "species", species)
        object.__setattr__(self, "instrument_keys", instrument_keys)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "log_temperature", log_temperature)
        object.__setattr__(self, "ion_fraction", ion_fraction)
        object.__setattr__(self, "electron_per_hydrogen", electron_per_hydrogen)
        object.__setattr__(self, "abundance_per_hydrogen", abundance)
        object.__setattr__(self, "effective_cross_section_cm2", cross_section)
        object.__setattr__(self, "provenance", provenance)

        computed = _semantic_bundle_id(self)
        stored = provenance.get("bundle_id")
        if stored is not None and stored != computed:
            raise ValueError(
                f"bundle_id verification failed: stored {stored!r}, computed {computed!r}"
            )
        provenance["bundle_id"] = computed

    @property
    def bundle_id(self) -> str:
        return self.provenance["bundle_id"]

    def verify_bundle_id(self) -> bool:
        computed = _semantic_bundle_id(self)
        if computed != self.bundle_id:
            raise ValueError(
                f"bundle_id verification failed: stored {self.bundle_id!r}, computed {computed!r}"
            )
        return True

    def indices_for(self, instrument_key: str, channels: Sequence[str]) -> np.ndarray:
        from sunerf.resources import builtin_name

        # Instrument keys are compared in canonical form so a configuration key
        # such as ``EUVI-A`` selects the packaged ``euvi_a`` rows.
        available = {
            (builtin_name(instrument), canonical_channel_id(channel)): index
            for index, (instrument, channel) in enumerate(
                zip(self.instrument_keys, self.channels)
            )
        }
        indices = []
        for channel in channels:
            identity = (builtin_name(instrument_key), canonical_channel_id(channel))
            if identity not in available:
                raise KeyError(
                    f"absorption bundle has no row for instrument/channel {identity!r}"
                )
            indices.append(available[identity])
        return np.asarray(indices, dtype=np.int64)

    def save(self, path):
        self.verify_bundle_id()
        atomic_savez_compressed(
            path,
            schema=np.asarray(ABSORPTION_BUNDLE_SCHEMA),
            schema_version=np.asarray(self.schema_version, dtype=np.int64),
            species=np.asarray(self.species),
            log_temperature=self.log_temperature,
            ion_fraction=self.ion_fraction,
            electron_per_hydrogen=self.electron_per_hydrogen,
            abundance_per_hydrogen=self.abundance_per_hydrogen,
            instrument_keys=np.asarray(self.instrument_keys),
            channels=np.asarray(self.channels),
            effective_cross_section_cm2=self.effective_cross_section_cm2,
            provenance_json=np.asarray(
                json.dumps(
                    dict(self.provenance),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            ),
        )


def load_absorption_bundle(path) -> AbsorptionBundle:
    """Load and fully validate an offline-built absorption bundle."""
    from sunerf.resources import resolve_artifact_path

    path = resolve_artifact_path(path)
    required = {
        "schema", "schema_version", "species", "log_temperature",
        "ion_fraction", "electron_per_hydrogen", "abundance_per_hydrogen",
        "instrument_keys", "channels", "effective_cross_section_cm2",
        "provenance_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"absorption bundle is missing required fields {missing}")
        if _scalar_string(archive["schema"], "schema") != ABSORPTION_BUNDLE_SCHEMA:
            raise ValueError("file is not a SuNeRF absorption bundle")
        try:
            provenance = json.loads(
                _scalar_string(archive["provenance_json"], "provenance_json")
            )
        except json.JSONDecodeError as error:
            raise ValueError("absorption bundle provenance is not valid JSON") from error
        bundle = AbsorptionBundle(
            species=tuple(np.asarray(archive["species"]).astype(str).tolist()),
            log_temperature=archive["log_temperature"],
            ion_fraction=archive["ion_fraction"],
            electron_per_hydrogen=archive["electron_per_hydrogen"],
            abundance_per_hydrogen=archive["abundance_per_hydrogen"],
            instrument_keys=tuple(
                np.asarray(archive["instrument_keys"]).astype(str).tolist()
            ),
            channels=tuple(np.asarray(archive["channels"]).astype(str).tolist()),
            effective_cross_section_cm2=archive["effective_cross_section_cm2"],
            provenance=provenance,
            schema_version=int(np.asarray(archive["schema_version"]).reshape(())),
        )
    bundle.verify_bundle_id()
    return bundle
