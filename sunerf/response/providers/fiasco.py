"""Pinned FIASCO/CHIANTI hybrid spectral-emissivity provider.

This module intentionally keeps FIASCO out of SuNeRF's runtime dependency set.
It is an offline release tool for an isolated Python 3.12+ environment with
``fiasco==0.8.2`` and a caller-supplied CHIANTI 11.0.2 HDF5 database.  It never
downloads or builds a database implicitly.

The provider does not use :meth:`fiasco.IonCollection.spectrum`.  That API is
experimental and broadens lines with an arbitrary kernel.  Instead, bound-bound
emission is retained as an exact line list, while free-free, free-bound, and
two-photon continua are evaluated as wavelength densities.  Exact lines can be
folded directly through an instrument response or conservatively rasterized for
the provider-neutral :class:`~sunerf.response.builder.SpectralEmissivityGrid`.

FIASCO's line contribution functions and continuum functions are coefficients
per ``n_e n_H`` and are emitted over all solid angle.  We divide by ``4*pi sr``
and photon energy.  The production ``ne2`` export additionally multiplies each
temperature slice by FIASCO's composition-dependent ``n_H/n_e`` ratio.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import gc
import hashlib
import io
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import threading
from types import MethodType
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from astropy import constants as const
from astropy import units as u

from sunerf.response.artifact import ResponseArtifact
from sunerf.response.builder import InstrumentThroughput, SpectralEmissivityGrid
from sunerf.response.numerics import trapezoid_node_weights
from sunerf.response.providers.base import OptionalProviderDependencyError, sha256_file


FIASCO_VERSION = "0.8.2"
CHIANTI_VERSION = "11.0.2"
DEFAULT_ABUNDANCE = "sun_coronal_2021_chianti"
DEFAULT_IONIZATION_EQUILIBRIUM = "chianti"
DEFAULT_IONIZATION_POTENTIAL = "chianti"
FIASCO_PROVIDER_VERSION = "fiasco-0.8.2-chianti-11.0.2-hybrid-v1"
HYBRID_EMISSIVITY_SCHEMA = "sunerf.fiasco-hybrid-emissivity"
HYBRID_EMISSIVITY_SCHEMA_VERSION = 1
EXACT_FOLD_ALGORITHM_VERSION = 1

LINE_EMISSIVITY_UNIT = "ph cm3 / (s sr)"
CONTINUUM_EMISSIVITY_UNIT = "ph cm3 / (Angstrom s sr)"
_LINE_UNIT = u.ph * u.cm**3 / (u.s * u.sr)
_CONTINUUM_UNIT = u.ph * u.cm**3 / (u.AA * u.s * u.sr)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ORDER = ("lines", "free_free", "free_bound", "two_photon")
_FIASCO_IMPORT_LOCK = threading.Lock()


def _strict_axis(values, name: str, *, positive: bool = False, allow_single: bool = False):
    values = np.asarray(values, dtype=np.float64)
    minimum = 1 if allow_single else 2
    if values.ndim != 1 or values.size < minimum:
        raise ValueError(f"{name} must be one-dimensional with at least {minimum} entries")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive")
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _nonnegative(values, name: str, shape: tuple[int, ...]):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; received {values.shape}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return values


def _json_mapping(value, name: str):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    value = dict(value)
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and JSON serializable") from error
    return value


def _validate_optional_sha256(value: str | None, name: str):
    if value is None:
        return None
    value = value.lower()
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    return value


def _hash_array(digest, name: str, values):
    values = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest.update(name.encode("utf-8"))
    digest.update(json.dumps(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))


def _canonical_unit_product(*units):
    """Multiply units while combining identical bases across CompositeUnits.

    Astropy 5 can retain two identical prefixed bases (for example ``cm3 cm2``)
    when multiplying already-composite units.  Rebuilding from summed powers
    makes the serialized response unit stable across supported Astropy releases.
    """
    scale = 1.0
    powers = {}
    for unit in units:
        unit = u.Unit(unit)
        scale *= unit.scale
        for base, power in zip(unit.bases, unit.powers):
            powers[base] = powers.get(base, 0) + power
    active = [(base, power) for base, power in powers.items() if power]
    return u.CompositeUnit(
        scale,
        [base for base, _ in active],
        [power for _, power in active],
    )


def _decode_hdf5_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _installed_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _discard_derived_ion_cache(ion, preserved_keys: set[str]) -> None:
    """Drop large line-population caches while retaining constructor state."""
    for key in set(ion.__dict__).difference(preserved_keys):
        ion.__dict__.pop(key, None)


@dataclass(frozen=True)
class FiascoConfig:
    """Byte-resolved atomic inputs and numerical choices for one export.

    The abundance and ionization-equilibrium files are the original CHIANTI
    ASCII inputs used to build ``hdf5_database``.  They are required even
    though FIASCO reads their parsed datasets from HDF5, because recording
    their independent hashes prevents a dataset-name-only provenance claim.
    """

    hdf5_database: str | Path
    abundance_file: str | Path
    ionization_equilibrium_file: str | Path
    abundance_name: str = DEFAULT_ABUNDANCE
    ionization_equilibrium_name: str = DEFAULT_IONIZATION_EQUILIBRIUM
    ionization_potential_name: str = DEFAULT_IONIZATION_POTENTIAL
    chianti_version: str = CHIANTI_VERSION
    ions: tuple[str, ...] | None = None
    emission_components: tuple[str, ...] = _COMPONENT_ORDER
    include_protons: bool = True
    include_level_resolved_rate_correction: bool = True
    use_two_ion_model: bool = True
    two_photon_include_protons: bool = False
    free_bound_use_verner: bool = True
    expected_database_sha256: str | None = None
    expected_abundance_sha256: str | None = None
    expected_ionization_equilibrium_sha256: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "hdf5_database", Path(self.hdf5_database))
        object.__setattr__(self, "abundance_file", Path(self.abundance_file))
        object.__setattr__(
            self,
            "ionization_equilibrium_file",
            Path(self.ionization_equilibrium_file),
        )
        for name in (
            "abundance_name",
            "ionization_equilibrium_name",
            "ionization_potential_name",
            "chianti_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"FiascoConfig.{name} must be a non-empty string")
        if self.chianti_version != CHIANTI_VERSION:
            raise ValueError(
                f"This provider is pinned to CHIANTI {CHIANTI_VERSION}; "
                f"received {self.chianti_version}"
            )
        if self.abundance_file.stem != self.abundance_name:
            raise ValueError(
                "abundance_file stem must equal abundance_name so the hashed "
                "ASCII source and selected HDF5 dataset cannot diverge"
            )
        if self.ionization_equilibrium_file.stem != self.ionization_equilibrium_name:
            raise ValueError(
                "ionization_equilibrium_file stem must equal "
                "ionization_equilibrium_name"
            )
        components = tuple(self.emission_components)
        if not components or len(set(components)) != len(components):
            raise ValueError("emission_components must be non-empty and unique")
        unknown = sorted(set(components).difference(_COMPONENT_ORDER))
        if unknown:
            raise ValueError(f"Unsupported FIASCO emission components {unknown}")
        components = tuple(item for item in _COMPONENT_ORDER if item in components)
        object.__setattr__(self, "emission_components", components)
        if self.ions is not None:
            ions = tuple(str(item) for item in self.ions)
            if not ions or any(not item.strip() for item in ions):
                raise ValueError("ions must contain non-empty FIASCO ion names")
            if len(ions) != len(set(ions)):
                raise ValueError("ions must be unique")
            object.__setattr__(self, "ions", ions)
        for name in (
            "include_protons",
            "include_level_resolved_rate_correction",
            "use_two_ion_model",
            "two_photon_include_protons",
            "free_bound_use_verner",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"FiascoConfig.{name} must be boolean")
        for name in (
            "expected_database_sha256",
            "expected_abundance_sha256",
            "expected_ionization_equilibrium_sha256",
        ):
            object.__setattr__(self, name, _validate_optional_sha256(getattr(self, name), name))


def _checked_digest(path: Path, expected: str | None, label: str):
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} source is missing: {path}")
    digest = sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(
            f"{label} SHA-256 verification failed for {path}: "
            f"expected {expected}, received {digest}"
        )
    return digest


def inspect_fiasco_sources(config: FiascoConfig) -> dict[str, Any]:
    """Validate selected HDF5 datasets and return byte-exact provenance.

    This helper does not import FIASCO and performs no network access.  It is
    useful when constructing a source manifest before a long emissivity run.
    """
    if not isinstance(config, FiascoConfig):
        raise TypeError("config must be a FiascoConfig")
    database_digest = _checked_digest(
        config.hdf5_database,
        config.expected_database_sha256,
        "CHIANTI HDF5 database",
    )
    abundance_digest = _checked_digest(
        config.abundance_file,
        config.expected_abundance_sha256,
        "CHIANTI abundance",
    )
    ionization_digest = _checked_digest(
        config.ionization_equilibrium_file,
        config.expected_ionization_equilibrium_sha256,
        "CHIANTI ionization-equilibrium",
    )

    required = {
        f"h/abundance/{config.abundance_name}": "abundance dataset",
        f"h/h_1/ioneq/{config.ionization_equilibrium_name}": (
            "ionization-equilibrium dataset"
        ),
        f"h/h_1/ip/{config.ionization_potential_name}": "ionization-potential dataset",
    }
    version_groups = (
        "h/abundance",
        f"h/h_1/ioneq/{config.ionization_equilibrium_name}",
        "h/h_1/ip",
    )
    with h5py.File(config.hdf5_database, "r") as database:
        for path, description in required.items():
            if path not in database:
                raise ValueError(
                    f"CHIANTI HDF5 database does not contain the selected {description}: {path}"
                )
        selected_versions = set()
        for path in version_groups:
            value = database[path].attrs.get("chianti_version")
            if value is None:
                raise ValueError(f"CHIANTI HDF5 group {path} has no chianti_version attribute")
            selected_versions.add(_decode_hdf5_attr(value))
        all_versions = set()
        versioned_group_count = 0

        def collect_version(_name, item):
            nonlocal versioned_group_count
            value = item.attrs.get("chianti_version")
            if value is not None:
                all_versions.add(_decode_hdf5_attr(value))
                versioned_group_count += 1

        database.visititems(collect_version)
    if selected_versions != {config.chianti_version}:
        raise ValueError(
            "Selected HDF5 datasets do not all report CHIANTI "
            f"{config.chianti_version}: found {sorted(selected_versions)}"
        )
    if all_versions != {config.chianti_version}:
        raise ValueError(
            "CHIANTI HDF5 database contains mixed or unexpected dataset versions: "
            f"expected only {config.chianti_version}, found {sorted(all_versions)}"
        )

    return {
        "provider": {"name": "fiasco", "version": FIASCO_VERSION},
        "runtime_dependencies": {
            package: _installed_version(package)
            for package in ("astropy", "h5py", "numpy", "plasmapy")
        },
        "atomic_database": {
            "name": "CHIANTI",
            "version": config.chianti_version,
            "sha256": database_digest,
            "filename": config.hdf5_database.name,
            "size_bytes": config.hdf5_database.stat().st_size,
            "versioned_group_count": versioned_group_count,
            "reference_url": "https://www.chiantidatabase.org/chianti_download.html",
        },
        "abundance": {
            "name": config.abundance_name,
            "version": config.chianti_version,
            "sha256": abundance_digest,
            "filename": config.abundance_file.name,
            "size_bytes": config.abundance_file.stat().st_size,
        },
        "ionization_equilibrium": {
            "name": config.ionization_equilibrium_name,
            "version": config.chianti_version,
            "sha256": ionization_digest,
            "filename": config.ionization_equilibrium_file.name,
            "size_bytes": config.ionization_equilibrium_file.stat().st_size,
        },
        "ionization_potential": {
            "name": config.ionization_potential_name,
            "version": config.chianti_version,
            "source": "included in atomic_database SHA-256",
        },
        "emission_components": list(config.emission_components),
        "atomic_options": {
            "include_protons": config.include_protons,
            "include_level_resolved_rate_correction": (
                config.include_level_resolved_rate_correction
            ),
            "use_two_ion_model": config.use_two_ion_model,
            "two_photon_include_protons": config.two_photon_include_protons,
            "free_bound_use_verner": config.free_bound_use_verner,
        },
        "radiometry": {
            "native_fiasco_emission_measure_convention": "ne_nh",
            "isotropic_emission_divisor": "4*pi sr",
            "spectral_quantity": "photon_number",
        },
    }


@dataclass(frozen=True)
class ExactLineEmissivity:
    """Integrated photon emissivity for exact CHIANTI bound-bound lines.

    ``emissivity`` has shape ``(density, temperature, line)`` and units
    :data:`LINE_EMISSIVITY_UNIT`, per ``n_e n_H`` emission measure.  Duplicate
    wavelengths are allowed because distinct transitions remain identifiable.
    """

    wavelength_angstrom: np.ndarray
    ion: tuple[str, ...]
    emissivity: np.ndarray

    def __post_init__(self):
        wavelength = np.asarray(self.wavelength_angstrom, dtype=np.float64)
        if wavelength.ndim != 1:
            raise ValueError("line wavelength_angstrom must be one-dimensional")
        if wavelength.size and (
            not np.isfinite(wavelength).all()
            or np.any(wavelength <= 0)
            or np.any(np.diff(wavelength) < 0)
        ):
            raise ValueError("line wavelengths must be finite, positive, and sorted")
        ions = tuple(str(item) for item in self.ion)
        if len(ions) != wavelength.size or any(not item for item in ions):
            raise ValueError("ion must contain one non-empty label per line")
        emissivity = np.asarray(self.emissivity, dtype=np.float64)
        if emissivity.ndim != 3 or emissivity.shape[-1] != wavelength.size:
            raise ValueError(
                "line emissivity must have shape (density, temperature, line)"
            )
        if not np.isfinite(emissivity).all() or np.any(emissivity < 0):
            raise ValueError("line emissivity must be finite and non-negative")
        object.__setattr__(self, "wavelength_angstrom", wavelength)
        object.__setattr__(self, "ion", ions)
        object.__setattr__(self, "emissivity", emissivity)


@dataclass(frozen=True)
class ContinuumEmissivity:
    """Component-resolved photon continua per ``n_e n_H`` emission measure."""

    wavelength_angstrom: np.ndarray
    free_free: np.ndarray
    free_bound: np.ndarray
    two_photon: np.ndarray

    def __post_init__(self):
        wavelength = _strict_axis(
            self.wavelength_angstrom,
            "continuum wavelength_angstrom",
            positive=True,
        )
        free_free = np.asarray(self.free_free, dtype=np.float64)
        if free_free.ndim != 3:
            raise ValueError(
                "continuum arrays must have shape (density, temperature, wavelength)"
            )
        shape = free_free.shape
        if shape[-1] != wavelength.size:
            raise ValueError("continuum wavelength dimension does not match wavelength axis")
        free_free = _nonnegative(free_free, "free_free", shape)
        free_bound = _nonnegative(self.free_bound, "free_bound", shape)
        two_photon = _nonnegative(self.two_photon, "two_photon", shape)
        object.__setattr__(self, "wavelength_angstrom", wavelength)
        object.__setattr__(self, "free_free", free_free)
        object.__setattr__(self, "free_bound", free_bound)
        object.__setattr__(self, "two_photon", two_photon)

    @property
    def total(self):
        return self.free_free + self.free_bound + self.two_photon


@dataclass(frozen=True)
class FiascoEmissionModel:
    """Exact lines plus resolved continuum on shared temperature/density axes."""

    log_temperature: np.ndarray
    log_density: np.ndarray
    hydrogen_to_electron_ratio: np.ndarray
    lines: ExactLineEmissivity
    continuum: ContinuumEmissivity
    provenance: Mapping[str, Any]
    schema_version: int = HYBRID_EMISSIVITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != HYBRID_EMISSIVITY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FIASCO hybrid schema version {self.schema_version}"
            )
        log_temperature = _strict_axis(self.log_temperature, "log_temperature")
        log_density = _strict_axis(self.log_density, "log_density", allow_single=True)
        ratio = np.asarray(self.hydrogen_to_electron_ratio, dtype=np.float64)
        if ratio.shape != log_temperature.shape:
            raise ValueError("hydrogen_to_electron_ratio must have one value per temperature")
        if not np.isfinite(ratio).all() or np.any((ratio <= 0) | (ratio > 1)):
            raise ValueError("hydrogen_to_electron_ratio must be finite and in (0, 1]")
        expected_prefix = (log_density.size, log_temperature.size)
        if self.lines.emissivity.shape[:2] != expected_prefix:
            raise ValueError("line axes do not match model density/temperature axes")
        if self.continuum.free_free.shape[:2] != expected_prefix:
            raise ValueError("continuum axes do not match model density/temperature axes")
        provenance = _json_mapping(self.provenance, "FiascoEmissionModel.provenance")
        object.__setattr__(self, "log_temperature", log_temperature)
        object.__setattr__(self, "log_density", log_density)
        object.__setattr__(self, "hydrogen_to_electron_ratio", ratio)
        object.__setattr__(self, "provenance", provenance)

    @property
    def content_sha256(self):
        digest = hashlib.sha256()
        digest.update(HYBRID_EMISSIVITY_SCHEMA.encode("utf-8"))
        digest.update(
            json.dumps(
                {
                    "schema_version": self.schema_version,
                    "line_unit": LINE_EMISSIVITY_UNIT,
                    "continuum_unit": CONTINUUM_EMISSIVITY_UNIT,
                    "line_ions": self.lines.ion,
                    "provenance": dict(self.provenance),
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in (
            ("log_temperature", self.log_temperature),
            ("log_density", self.log_density),
            ("hydrogen_to_electron_ratio", self.hydrogen_to_electron_ratio),
            ("line_wavelength_angstrom", self.lines.wavelength_angstrom),
            ("line_emissivity", self.lines.emissivity),
            ("continuum_wavelength_angstrom", self.continuum.wavelength_angstrom),
            ("free_free", self.continuum.free_free),
            ("free_bound", self.continuum.free_bound),
            ("two_photon", self.continuum.two_photon),
        ):
            _hash_array(digest, name, values)
        return digest.hexdigest()

    def for_emission_measure(self, convention: str):
        """Return line and continuum arrays for ``ne_nh`` or ``ne2`` EM."""
        if convention not in {"ne_nh", "ne2"}:
            raise ValueError("emission_measure_convention must be 'ne_nh' or 'ne2'")
        line = self.lines.emissivity
        continuum = self.continuum.total
        if convention == "ne2":
            scale = self.hydrogen_to_electron_ratio[np.newaxis, :, np.newaxis]
            line = line * scale
            continuum = continuum * scale
        return line, continuum

    def fold(self, throughput: InstrumentThroughput, **kwargs) -> ResponseArtifact:
        """Fold exact lines and continuum through ``throughput``."""
        return fold_fiasco_emission(self, throughput, **kwargs)

    def rasterize(self, **kwargs) -> SpectralEmissivityGrid:
        """Rasterize the hybrid model onto a wavelength-density grid."""
        return rasterize_fiasco_emission(self, **kwargs)


def import_fiasco_offline():
    """Import FIASCO while blocking PlasmaPy's import-time GitHub probe.

    PlasmaPy 2026.2.0 performs an unconditional ``requests.get`` when its data
    downloader module is first imported. FIASCO does not need any PlasmaPy
    remote resource for CHIANTI calculations, so the response exporter blocks
    that one import-time request and suppresses PlasmaPy's resulting diagnostic.
    The patch is held under a lock and restored immediately after import.
    """
    with _FIASCO_IMPORT_LOCK:
        if "fiasco" in sys.modules:
            return sys.modules["fiasco"]
        import requests

        original_get = requests.get

        def offline_get(*args, **kwargs):
            del args, kwargs
            raise requests.exceptions.ConnectionError(
                "network access is disabled during the SuNeRF FIASCO import"
            )

        captured_stdout = io.StringIO()
        try:
            requests.get = offline_get
            with redirect_stdout(captured_stdout):
                import fiasco
        finally:
            requests.get = original_get
        return fiasco


def _load_fiasco():
    if sys.version_info < (3, 12):
        raise OptionalProviderDependencyError(
            "The pinned atomic exporter requires Python >=3.12 and "
            f"fiasco=={FIASCO_VERSION}; run it in the isolated response environment."
        )
    try:
        installed = importlib.metadata.version("fiasco")
    except importlib.metadata.PackageNotFoundError as error:
        raise OptionalProviderDependencyError(
            f"The atomic exporter requires fiasco=={FIASCO_VERSION}."
        ) from error
    if installed != FIASCO_VERSION:
        raise OptionalProviderDependencyError(
            f"The atomic exporter is validated only with fiasco=={FIASCO_VERSION}; "
            f"found fiasco=={installed}."
        )
    try:
        fiasco = import_fiasco_offline()
        from fiasco.util.exceptions import MissingDatasetException
    except ImportError as error:  # pragma: no cover - broken installations
        raise OptionalProviderDependencyError(
            f"Could not import required fiasco=={FIASCO_VERSION} APIs."
        ) from error
    return fiasco, MissingDatasetException


def _energy_to_photon_density(emissivity, wavelength):
    photon_energy = (const.h * const.c / wavelength).to(u.erg)
    return (
        emissivity / photon_energy * u.ph / (4 * np.pi * u.sr)
    ).to(_CONTINUUM_UNIT)


def _energy_to_integrated_photons(emissivity, wavelength):
    photon_energy = wavelength.to(u.erg, equivalencies=u.spectral())
    return (
        emissivity / photon_energy[np.newaxis, np.newaxis, :] * u.ph / (4 * np.pi * u.sr)
    ).to(_LINE_UNIT)


def compute_fiasco_emission_model(
    config: FiascoConfig,
    log_temperature: Sequence[float],
    log_density: Sequence[float],
    continuum_wavelength_angstrom: Sequence[float],
    *,
    line_wavelength_range_angstrom: Sequence[float] | None = None,
    show_progress: bool = False,
) -> FiascoEmissionModel:
    """Compute a full hybrid emissivity model using FIASCO 0.8.2.

    The function validates and hashes every caller-supplied source before
    importing FIASCO.  ``log_density`` is an independent density axis; constant
    pressure or temperature-coupled densities are intentionally not inferred.
    """
    if not isinstance(config, FiascoConfig):
        raise TypeError("config must be a FiascoConfig")
    log_temperature = _strict_axis(log_temperature, "log_temperature")
    log_density = _strict_axis(log_density, "log_density", allow_single=True)
    continuum_wavelength = _strict_axis(
        continuum_wavelength_angstrom,
        "continuum_wavelength_angstrom",
        positive=True,
    )
    if line_wavelength_range_angstrom is None:
        line_wavelength_range = np.array(
            [continuum_wavelength[0], continuum_wavelength[-1]],
            dtype=np.float64,
        )
        line_range_source = "continuum_endpoints"
    else:
        line_wavelength_range = np.asarray(
            line_wavelength_range_angstrom,
            dtype=np.float64,
        )
        if (
            line_wavelength_range.shape != (2,)
            or not np.isfinite(line_wavelength_range).all()
            or np.any(line_wavelength_range <= 0)
            or line_wavelength_range[1] <= line_wavelength_range[0]
        ):
            raise ValueError(
                "line_wavelength_range_angstrom must contain two finite positive "
                "strictly increasing bounds"
            )
        line_range_source = "explicit"
    provenance = inspect_fiasco_sources(config)
    fiasco, MissingDatasetException = _load_fiasco()
    # Missing optional proton/two-ion datasets are expected for many CHIANTI
    # ions and FIASCO already falls back deterministically. Keep a full release
    # readable while preserving unexpected exceptions.
    fiasco.log.setLevel("ERROR")

    temperature = 10**log_temperature * u.K
    density = 10**log_density / u.cm**3
    wavelength = continuum_wavelength * u.AA
    ion_kwargs = {
        "hdf5_dbase_root": config.hdf5_database,
        "abundance": config.abundance_name,
        "ionization_fraction": config.ionization_equilibrium_name,
        "ionization_potential": config.ionization_potential_name,
    }
    ion_names = config.ions
    if ion_names is None:
        ion_names = tuple(fiasco.list_ions(config.hdf5_database))
    ions = tuple(fiasco.Ion(name, temperature, **ion_kwargs) for name in ion_names)
    if not ions:
        raise ValueError("No ions were selected from the CHIANTI database")
    collection = fiasco.IonCollection(*ions)

    ratio_quantity = fiasco.proton_electron_ratio(temperature, **ion_kwargs)
    # Ion.proton_electron_ratio is a cached property, but computing it once per
    # selected ion repeats the same all-element composition calculation hundreds
    # of times. Seed each ion with the identical, explicitly selected result.
    # This changes no physics and makes a complete-database release practical.
    for ion in ions:
        ion.__dict__["proton_electron_ratio"] = ratio_quantity
    # The two-ion population model constructs the next ion repeatedly. Reuse
    # one temporary instance per active ion and seed it with the same ratio;
    # the temporary and its derived matrices are discarded after that ion.
    ion_type = type(ions[0])
    original_next_ion = ion_type.next_ion

    def next_ion_with_shared_ratio(active_ion):
        next_ion = active_ion.__dict__.get("_sunerf_next_ion")
        if next_ion is None:
            next_ion = original_next_ion(active_ion)
            next_ion.__dict__["proton_electron_ratio"] = ratio_quantity
            active_ion.__dict__["_sunerf_next_ion"] = next_ion
        return next_ion

    for ion in ions:
        ion.next_ion = MethodType(next_ion_with_shared_ratio, ion)
    preserved_ion_keys = tuple(set(ion.__dict__) for ion in ions)
    ratio = np.asarray(
        ratio_quantity.to_value(u.dimensionless_unscaled),
        dtype=np.float64,
    )

    line_wavelengths = []
    line_ions: list[str] = []
    line_emissivities = []
    skipped_line_ions = []
    zeroed_absent_ion_entries = {}
    candidate_line_count = 0
    excluded_line_count = 0
    if "lines" in config.emission_components:
        population_options = {
            "include_protons": config.include_protons,
            "include_level_resolved_rate_correction": (
                config.include_level_resolved_rate_correction
            ),
            "use_two_ion_model": config.use_two_ion_model,
        }
        line_ions_iterator = ions
        if show_progress:
            from tqdm.auto import tqdm

            line_ions_iterator = tqdm(
                ions,
                desc="FIASCO bound-bound ions",
                unit="ion",
            )
        for ion, preserved_cache_keys in zip(line_ions_iterator, preserved_ion_keys):
            try:
                transitions = ion.transitions
                bound_bound = transitions.is_bound_bound
                ion_wavelength = transitions.wavelength[bound_bound]
            except MissingDatasetException:
                skipped_line_ions.append(ion.ion_name)
                _discard_derived_ion_cache(ion, preserved_cache_keys)
                continue
            if ion_wavelength.size == 0:
                skipped_line_ions.append(ion.ion_name)
                _discard_derived_ion_cache(ion, preserved_cache_keys)
                continue
            candidate_line_count += ion_wavelength.size
            line_in_range = (
                (ion_wavelength >= line_wavelength_range[0] * u.AA)
                & (ion_wavelength <= line_wavelength_range[1] * u.AA)
            )
            excluded_line_count += int(np.count_nonzero(~line_in_range))
            if not np.any(line_in_range):
                _discard_derived_ion_cache(ion, preserved_cache_keys)
                continue
            try:
                contribution = ion.contribution_function(density, **population_options)
            except MissingDatasetException:
                skipped_line_ions.append(ion.ion_name)
                _discard_derived_ion_cache(ion, preserved_cache_keys)
                continue
            if contribution.shape[-1] != ion_wavelength.size:
                raise ValueError(
                    f"FIASCO {ion.ion_name} contribution-function line axis does not "
                    "match its bound-bound transition mask"
                )
            ion_wavelength = ion_wavelength[line_in_range]
            contribution = contribution[..., line_in_range]
            # The level-population solve can underflow to NaN far below an
            # ion's formation temperature. Where the ionization fraction is
            # exactly zero the emissivity is zero by definition; any other
            # non-finite value still fails validation below.
            absent = np.asarray(ion.ionization_fraction) == 0
            undefined = ~np.isfinite(contribution.value) & absent[:, np.newaxis, np.newaxis]
            if np.any(undefined):
                contribution = np.where(undefined, 0.0, contribution.value) * contribution.unit
                zeroed_absent_ion_entries[ion.ion_name] = int(np.count_nonzero(undefined))
            line_wavelengths.append(ion_wavelength.to_value(u.AA))
            line_ions.extend([ion.ion_name] * ion_wavelength.size)
            photons = _energy_to_integrated_photons(contribution, ion_wavelength)
            line_emissivities.append(
                np.transpose(photons.to_value(_LINE_UNIT), (1, 0, 2))
            )
            _discard_derived_ion_cache(ion, preserved_cache_keys)

    if line_wavelengths:
        exact_wavelength = np.concatenate(line_wavelengths)
        exact_emissivity = np.concatenate(line_emissivities, axis=2)
        del line_wavelengths, line_emissivities
        order = np.argsort(exact_wavelength, kind="stable")
        exact_wavelength = exact_wavelength[order]
        exact_emissivity = exact_emissivity[..., order]
        line_ions_array = np.asarray(line_ions, dtype=str)
        del line_ions
        line_ions_tuple = tuple(line_ions_array[order].tolist())
        del line_ions_array, order
    else:
        exact_wavelength = np.empty(0, dtype=np.float64)
        exact_emissivity = np.empty(
            (log_density.size, log_temperature.size, 0),
            dtype=np.float64,
        )
        line_ions_tuple = ()

    continuum_shape = (log_density.size, log_temperature.size, continuum_wavelength.size)
    zeros = np.zeros(continuum_shape, dtype=np.float64)
    free_free = zeros.copy()
    free_bound = zeros.copy()
    two_photon = zeros.copy()
    if "free_free" in config.emission_components:
        value = _energy_to_photon_density(collection.free_free(wavelength), wavelength)
        free_free = np.broadcast_to(
            value.to_value(_CONTINUUM_UNIT)[np.newaxis, ...],
            continuum_shape,
        ).copy()
    if "free_bound" in config.emission_components:
        value = _energy_to_photon_density(
            collection.free_bound(wavelength, use_verner=config.free_bound_use_verner),
            wavelength,
        )
        free_bound = np.broadcast_to(
            value.to_value(_CONTINUUM_UNIT)[np.newaxis, ...],
            continuum_shape,
        ).copy()
    if "two_photon" in config.emission_components:
        # FIASCO 0.8.2 divides the (temperature, density, 1) level population
        # by a (density,) array, which broadcasts to (temperature, density,
        # density) for more than one density. Evaluating one density at a time
        # is exact and independent of that defect.
        per_density = [
            _energy_to_photon_density(
                collection.two_photon(
                    wavelength,
                    density[index:index + 1],
                    include_protons=config.two_photon_include_protons,
                ),
                wavelength,
            ).to_value(_CONTINUUM_UNIT)
            for index in range(density.size)
        ]
        two_photon = np.transpose(np.concatenate(per_density, axis=1), (1, 0, 2))

    # Continuum calls can repopulate per-ion cached datasets and the temporary
    # next-ion objects. Remove those and the instance-bound method cycle before
    # returning the compact numerical model.
    for ion, preserved_cache_keys in zip(ions, preserved_ion_keys):
        _discard_derived_ion_cache(ion, preserved_cache_keys)
        ion.__dict__.pop("_sunerf_next_ion", None)
        ion.__dict__.pop("next_ion", None)
    del collection, ions
    gc.collect()

    provenance["provider"]["adapter"] = FIASCO_PROVIDER_VERSION
    provenance["ion_selection"] = {
        "mode": "all_database_ions" if config.ions is None else "explicit",
        "requested_count": len(ion_names),
        "line_emitting_count": len(set(line_ions_tuple)),
        "skipped_bound_bound_ions": sorted(set(skipped_line_ions)),
        "zeroed_undefined_entries_of_absent_ions": dict(sorted(zeroed_absent_ion_entries.items())),
    }
    provenance["line_selection"] = {
        "wavelength_coordinate": "vacuum_angstrom",
        "wavelength_range_angstrom": line_wavelength_range.tolist(),
        "range_source": line_range_source,
        "candidate_bound_bound_line_count": candidate_line_count,
        "included_line_count": int(exact_wavelength.size),
        "excluded_outside_range_count": excluded_line_count,
    }
    provenance["normalization"] = {
        "native_emission_measure_convention": "ne_nh",
        "production_emission_measure_convention": "ne2",
        "ne2_conversion": "multiply each temperature slice by fiasco.proton_electron_ratio",
        "hydrogen_to_electron_ratio_sha256": hashlib.sha256(
            np.ascontiguousarray(ratio, dtype="<f8").tobytes()
        ).hexdigest(),
        "hydrogen_to_electron_ratio_min": float(ratio.min()),
        "hydrogen_to_electron_ratio_max": float(ratio.max()),
        "shared_ratio_cache": (
            "one identical fiasco.proton_electron_ratio evaluation reused by all ions"
        ),
        "line_population_cache_policy": (
            "discard derived per-ion collision/rate matrices after copying line coefficients"
        ),
        "dependency_network_policy": (
            "PlasmaPy import-time GitHub probe blocked; no remote PlasmaPy data used"
        ),
    }

    return FiascoEmissionModel(
        log_temperature=log_temperature,
        log_density=log_density,
        hydrogen_to_electron_ratio=ratio,
        lines=ExactLineEmissivity(
            wavelength_angstrom=exact_wavelength,
            ion=line_ions_tuple,
            emissivity=exact_emissivity,
        ),
        continuum=ContinuumEmissivity(
            wavelength_angstrom=continuum_wavelength,
            free_free=free_free,
            free_bound=free_bound,
            two_photon=two_photon,
        ),
        provenance=provenance,
    )


def rasterize_lines(
    line_wavelength_angstrom,
    integrated_line_emissivity,
    wavelength_angstrom,
    *,
    outside: str = "error",
):
    """Conservatively place integrated delta-function lines on a node grid.

    Each line is split between its two bracketing nodes.  Division by the
    nonuniform trapezoid node weights makes the wavelength integral equal the
    original integrated line emissivity.  It also preserves folding against
    any response that is linear between those two grid nodes.
    """
    line_wavelength = np.asarray(line_wavelength_angstrom, dtype=np.float64)
    if line_wavelength.ndim != 1 or (
        line_wavelength.size
        and (not np.isfinite(line_wavelength).all() or np.any(line_wavelength <= 0))
    ):
        raise ValueError("line_wavelength_angstrom must be finite, positive, and one-dimensional")
    line_emissivity = np.asarray(integrated_line_emissivity, dtype=np.float64)
    if line_emissivity.ndim < 1 or line_emissivity.shape[-1] != line_wavelength.size:
        raise ValueError("integrated_line_emissivity last axis must match line wavelengths")
    if not np.isfinite(line_emissivity).all() or np.any(line_emissivity < 0):
        raise ValueError("integrated_line_emissivity must be finite and non-negative")
    wavelength = _strict_axis(wavelength_angstrom, "wavelength_angstrom", positive=True)
    if outside not in {"error", "drop"}:
        raise ValueError("outside must be 'error' or 'drop'")
    in_range = (line_wavelength >= wavelength[0]) & (line_wavelength <= wavelength[-1])
    if outside == "error" and np.any(~in_range):
        raise ValueError(
            f"{np.count_nonzero(~in_range)} exact lines fall outside the raster wavelength range"
        )
    line_wavelength = line_wavelength[in_range]
    line_emissivity = line_emissivity[..., in_range]
    output = np.zeros(line_emissivity.shape[:-1] + (wavelength.size,), dtype=np.float64)
    if line_wavelength.size == 0:
        return output

    node_weights = trapezoid_node_weights(wavelength)
    right = np.searchsorted(wavelength, line_wavelength, side="left")
    right = np.clip(right, 1, wavelength.size - 1)
    left = right - 1
    fraction_right = (
        (line_wavelength - wavelength[left]) / (wavelength[right] - wavelength[left])
    )
    at_left_edge = line_wavelength == wavelength[0]
    left[at_left_edge] = 0
    right[at_left_edge] = 0
    fraction_right[at_left_edge] = 0.0

    flat_input = line_emissivity.reshape((-1, line_wavelength.size))
    flat_output = output.reshape((-1, wavelength.size))
    for input_row, output_row in zip(flat_input, flat_output):
        np.add.at(
            output_row,
            left,
            input_row * (1.0 - fraction_right) / node_weights[left],
        )
        use_right = right != left
        np.add.at(
            output_row,
            right[use_right],
            input_row[use_right]
            * fraction_right[use_right]
            / node_weights[right[use_right]],
        )
    return output


def _interpolate_last_axis(values, source_wavelength, target_wavelength):
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape((-1, values.shape[-1]))
    result = np.stack(
        [
            np.interp(target_wavelength, source_wavelength, row, left=0.0, right=0.0)
            for row in flat
        ],
        axis=0,
    )
    return result.reshape(values.shape[:-1] + (len(target_wavelength),))


def _export_provenance(
    model: FiascoEmissionModel,
    convention: str,
    hydrogen_to_electron_ratio: float | None,
):
    provenance = dict(model.provenance)
    provenance["emission_measure_export"] = {
        "convention": convention,
        "native_convention": "ne_nh",
        "ne2_conversion": (
            "temperature-dependent fiasco.proton_electron_ratio"
            if convention == "ne2"
            else "none"
        ),
    }
    if convention == "ne_nh":
        if (
            not isinstance(hydrogen_to_electron_ratio, (int, float))
            or not np.isfinite(hydrogen_to_electron_ratio)
            or not 0 < hydrogen_to_electron_ratio <= 1
        ):
            raise ValueError(
                "ne_nh ResponseArtifact/SpectralEmissivityGrid export requires an explicit "
                "scalar hydrogen_to_electron_ratio for SuNeRF runtime conversion; prefer "
                "the exact temperature-dependent ne2 production convention"
            )
        provenance["hydrogen_to_electron_ratio"] = float(hydrogen_to_electron_ratio)
        provenance["emission_measure_export"]["runtime_ratio_is_scalar_override"] = True
    return provenance


def rasterize_fiasco_emission(
    model: FiascoEmissionModel,
    *,
    wavelength_angstrom: Sequence[float] | None = None,
    emission_measure_convention: str = "ne2",
    hydrogen_to_electron_ratio: float | None = None,
) -> SpectralEmissivityGrid:
    """Export a provider-neutral spectral grid from a hybrid FIASCO model."""
    if not isinstance(model, FiascoEmissionModel):
        raise TypeError("model must be a FiascoEmissionModel")
    wavelength = (
        model.continuum.wavelength_angstrom
        if wavelength_angstrom is None
        else _strict_axis(wavelength_angstrom, "wavelength_angstrom", positive=True)
    )
    line, continuum = model.for_emission_measure(emission_measure_convention)
    line_density = rasterize_lines(
        model.lines.wavelength_angstrom,
        line,
        wavelength,
        outside="drop",
    )
    continuum_density = _interpolate_last_axis(
        continuum,
        model.continuum.wavelength_angstrom,
        wavelength,
    )
    provenance = _export_provenance(
        model,
        emission_measure_convention,
        hydrogen_to_electron_ratio,
    )
    provenance["spectral_export"] = {
        "line_representation": "conservative_linear_two_node_deposition",
        "line_unit_before_rasterization": LINE_EMISSIVITY_UNIT,
        "continuum_interpolation": "linear_zero_outside_source_range",
        "excluded_line_count": int(
            np.count_nonzero(
                (model.lines.wavelength_angstrom < wavelength[0])
                | (model.lines.wavelength_angstrom > wavelength[-1])
            )
        ),
        "hybrid_content_sha256": model.content_sha256,
    }
    return SpectralEmissivityGrid(
        wavelength_angstrom=wavelength,
        log_temperature=model.log_temperature,
        log_density=model.log_density,
        emissivity=line_density + continuum_density,
        emissivity_unit=CONTINUUM_EMISSIVITY_UNIT,
        emission_measure_convention=emission_measure_convention,
        provenance=provenance,
    )


def fold_fiasco_emission(
    model: FiascoEmissionModel,
    throughput: InstrumentThroughput,
    *,
    channels: Sequence[str] | None = None,
    emission_measure_convention: str = "ne2",
    hydrogen_to_electron_ratio: float | None = None,
    response_unit: str | None = None,
) -> ResponseArtifact:
    """Fold exact lines and resolved continuum without an arbitrary line kernel.

    Throughput is linearly interpolated at every exact line wavelength and is
    zero outside its tabulated domain.  Continuum is linearly interpolated onto
    the throughput wavelength nodes and integrated with nonuniform trapezoid
    weights.  Positive throughput support must lie inside the computed
    continuum interval; missing atomic continuum coverage is never treated as
    zero silently.
    """
    if not isinstance(model, FiascoEmissionModel):
        raise TypeError("model must be a FiascoEmissionModel")
    if not isinstance(throughput, InstrumentThroughput):
        raise TypeError("throughput must be an InstrumentThroughput")
    throughput = throughput.select_channels(channels)
    line, continuum = model.for_emission_measure(emission_measure_convention)

    line_throughput = np.stack(
        [
            np.interp(
                model.lines.wavelength_angstrom,
                throughput.wavelength_angstrom,
                row,
                left=0.0,
                right=0.0,
            )
            for row in throughput.throughput
        ],
        axis=0,
    )
    line_response = np.einsum("dtl,cl->cdt", line, line_throughput, optimize=True)

    continuum_min = model.continuum.wavelength_angstrom[0]
    continuum_max = model.continuum.wavelength_angstrom[-1]
    positive_support = np.any(throughput.throughput > 0, axis=0)
    unsupported = positive_support & (
        (throughput.wavelength_angstrom < continuum_min)
        | (throughput.wavelength_angstrom > continuum_max)
    )
    if np.any(unsupported):
        raise ValueError(
            "positive throughput support falls outside the computed FIASCO continuum range"
        )
    continuum_on_throughput = _interpolate_last_axis(
        continuum,
        model.continuum.wavelength_angstrom,
        throughput.wavelength_angstrom,
    )
    wavelength_weights = trapezoid_node_weights(throughput.wavelength_angstrom)
    continuum_response = np.einsum(
        "dtw,cw,w->cdt",
        continuum_on_throughput,
        throughput.throughput,
        wavelength_weights,
        optimize=True,
    )

    throughput_unit = u.Unit(throughput.throughput_unit)
    line_native_unit = _canonical_unit_product(_LINE_UNIT, throughput_unit)
    continuum_native_unit = _canonical_unit_product(
        _CONTINUUM_UNIT,
        throughput_unit,
        u.AA,
    )
    try:
        continuum_scale = (1.0 * continuum_native_unit).to_value(line_native_unit)
    except u.UnitConversionError as error:
        raise ValueError(
            "line and continuum folds have incompatible units for this throughput: "
            f"{line_native_unit} and {continuum_native_unit}"
        ) from error
    response = line_response + continuum_scale * continuum_response
    output_unit = line_native_unit
    if response_unit is not None:
        try:
            scale = (1.0 * output_unit).to_value(u.Unit(response_unit))
        except (TypeError, ValueError, u.UnitConversionError) as error:
            raise ValueError(
                f"requested response_unit {response_unit!r} is incompatible with {output_unit}"
            ) from error
        response = response * scale
        output_unit = u.Unit(response_unit)

    spectral_provenance = _export_provenance(
        model,
        emission_measure_convention,
        hydrogen_to_electron_ratio,
    )
    provenance = {
        "builder": {
            "name": "sunerf.response.providers.fiasco.fold_fiasco_emission",
            "algorithm_version": EXACT_FOLD_ALGORITHM_VERSION,
        },
        "spectral_emissivity": {
            "schema": HYBRID_EMISSIVITY_SCHEMA,
            "schema_version": model.schema_version,
            "content_sha256": model.content_sha256,
            **spectral_provenance,
        },
        "instrument_throughput": {
            "content_sha256": throughput.content_sha256,
            "calibration_epoch": throughput.calibration_epoch,
            **dict(throughput.provenance),
        },
        "fold": {
            "lines": "linear_throughput_interpolation_at_exact_line_wavelength",
            "line_normalization": "integrated_photon_emissivity_no_kernel",
            "continuum_interpolation": "linear_onto_throughput_wavelength_nodes",
            "continuum_quadrature": "trapezoidal_node_weights",
            "normalization": "none",
        },
        "sensitivity_convention": throughput.provenance["sensitivity_convention"],
        "calibration_epoch": throughput.calibration_epoch,
        "measurement_semantics": throughput.provenance["radiometry"][
            "measurement_semantics"
        ],
    }
    if provenance["measurement_semantics"] == "per_native_pixel":
        provenance["native_pixel_solid_angle_sr"] = throughput.provenance["radiometry"][
            "native_pixel_solid_angle_sr"
        ]
        provenance["native_pixel_solid_angle_relative_tolerance"] = throughput.provenance[
            "radiometry"
        ]["native_pixel_solid_angle_relative_tolerance"]
    if emission_measure_convention == "ne_nh":
        provenance["hydrogen_to_electron_ratio"] = float(hydrogen_to_electron_ratio)

    return ResponseArtifact(
        channels=throughput.channels,
        log_temperature=model.log_temperature,
        log_density=model.log_density,
        response=response,
        response_unit=output_unit.to_string(),
        emission_measure_convention=emission_measure_convention,
        provenance=provenance,
    )


__all__ = [
    "CHIANTI_VERSION",
    "CONTINUUM_EMISSIVITY_UNIT",
    "DEFAULT_ABUNDANCE",
    "DEFAULT_IONIZATION_EQUILIBRIUM",
    "DEFAULT_IONIZATION_POTENTIAL",
    "EXACT_FOLD_ALGORITHM_VERSION",
    "FIASCO_PROVIDER_VERSION",
    "FIASCO_VERSION",
    "HYBRID_EMISSIVITY_SCHEMA",
    "HYBRID_EMISSIVITY_SCHEMA_VERSION",
    "LINE_EMISSIVITY_UNIT",
    "ContinuumEmissivity",
    "ExactLineEmissivity",
    "FiascoConfig",
    "FiascoEmissionModel",
    "compute_fiasco_emission_model",
    "fold_fiasco_emission",
    "inspect_fiasco_sources",
    "import_fiasco_offline",
    "rasterize_fiasco_emission",
    "rasterize_lines",
]
