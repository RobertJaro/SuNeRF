"""Build the shared, pinned CHIANTI spectral-emissivity artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sunerf.response.builder import load_spectral_emissivity
from sunerf.response.providers.chianti import (
    ABUNDANCE_SHA256,
    CHIANTI_VERSION,
    IONIZATION_EQUILIBRIUM_SHA256,
    prepare_chianti_database,
)
from sunerf.response.providers.fiasco import FiascoConfig, compute_fiasco_emission_model


# log10(T / K): the native CHIANTI ionization-equilibrium spacing. 10^4 K is
# required because the pointwise temperature is shared with the H/He opacity;
# above 10^7.5 K only continuum contributes to the supported channels.
LOG_TEMPERATURE = np.linspace(4.0, 8.0, 81)
# log10(n_e / cm^-3): level populations are density independent below 10^7
# (coronal limit) and emitting plasma denser than 10^11 only occurs in flares,
# so the renderer clamps its lookup to this axis.
LOG_DENSITY = np.linspace(7.0, 11.0, 9)
WAVELENGTH_ANGSTROM = np.arange(10.0, 1000.0 + 0.5, 1.0)
ARTIFACT_NAME = "chianti_coronal_2021.spectral.npz"


def default_output_path(root: str | Path) -> Path:
    """Return the shared artifact path beneath a calibration root."""
    return Path(root) / "responses" / ARTIFACT_NAME


def _validate_existing(path: Path) -> None:
    grid = load_spectral_emissivity(path)
    expected_axes = (
        ("log_temperature", grid.log_temperature, LOG_TEMPERATURE),
        ("log_density", grid.log_density, LOG_DENSITY),
        ("wavelength_angstrom", grid.wavelength_angstrom, WAVELENGTH_ANGSTROM),
    )
    for name, actual, expected in expected_axes:
        if actual is None or not np.array_equal(actual, expected):
            raise ValueError(
                f"Existing spectral-emissivity artifact has an unexpected {name}: {path}. "
                "Use --overwrite to rebuild it."
            )
    provenance = grid.provenance
    if (
        provenance.get("atomic_database", {}).get("version") != CHIANTI_VERSION
        or provenance.get("abundance", {}).get("sha256") != ABUNDANCE_SHA256
        or provenance.get("ionization_equilibrium", {}).get("sha256")
        != IONIZATION_EQUILIBRIUM_SHA256
        or grid.emission_measure_convention != "ne2"
    ):
        raise ValueError(
            f"Existing spectral-emissivity artifact does not match the pinned model: {path}. "
            "Use --overwrite to rebuild it."
        )


def generate_spectral_emissivity(
    root: str | Path,
    *,
    output: str | Path | None = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Generate or validate the one emissivity table shared by all runs."""
    root = Path(root)
    output = default_output_path(root) if output is None else Path(output)
    if output.exists() and not overwrite:
        _validate_existing(output)
        return output

    sources = prepare_chianti_database(
        root / "chianti" / CHIANTI_VERSION,
        show_progress=show_progress,
    )
    config = FiascoConfig(
        hdf5_database=sources["database"],
        abundance_file=sources["abundance"],
        ionization_equilibrium_file=sources["ionization_equilibrium"],
        expected_abundance_sha256=ABUNDANCE_SHA256,
        expected_ionization_equilibrium_sha256=IONIZATION_EQUILIBRIUM_SHA256,
    )
    model = compute_fiasco_emission_model(
        config,
        LOG_TEMPERATURE,
        LOG_DENSITY,
        WAVELENGTH_ANGSTROM,
        line_wavelength_range_angstrom=(
            WAVELENGTH_ANGSTROM[0],
            WAVELENGTH_ANGSTROM[-1],
        ),
        show_progress=show_progress,
    )
    grid = model.rasterize(
        wavelength_angstrom=WAVELENGTH_ANGSTROM,
        emission_measure_convention="ne2",
    )
    grid.save(output)
    _validate_existing(output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the shared CHIANTI 11.0.2 coronal emissivity table."
    )
    parser.add_argument(
        "--root",
        default="data/response_calibration",
        help="Shared calibration cache (default: data/response_calibration).",
    )
    parser.add_argument("--output", help="Override the output NPZ path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute an existing output artifact.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide FIASCO progress bars.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = generate_spectral_emissivity(
        args.root,
        output=args.output,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
