"""Offline-built, deterministic EUV photoionization absorption bundles."""

from .artifact import (
    ABSORPTION_BUNDLE_SCHEMA,
    ABSORPTION_BUNDLE_SCHEMA_VERSION,
    ABSORPTION_SPECIES,
    AbsorptionBundle,
    load_absorption_bundle,
)

__all__ = [
    "ABSORPTION_BUNDLE_SCHEMA",
    "ABSORPTION_BUNDLE_SCHEMA_VERSION",
    "ABSORPTION_SPECIES",
    "AbsorptionBundle",
    "load_absorption_bundle",
]
