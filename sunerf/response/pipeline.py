"""Download instrument responses and fold one shared emissivity table.

The default workflow has one deliberately narrow boundary:

* instrument-specific code downloads and converts calibration files to
  :class:`~sunerf.response.builder.InstrumentThroughput`; and
* provider-neutral code folds one already computed spectral emissivity grid
  through every throughput artifact in exactly the same way.

No atomic package is imported here. FIASCO, ChiantiPy, SolarSoft, or another
tool may produce the one spectral-emissivity input. Training only consumes the
resulting temperature-response artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from sunerf.response.builder import (
    fold_temperature_response,
    input_schema_description,
    load_instrument_throughput,
    load_spectral_emissivity,
)
from sunerf.response.providers.aia import export_aia_throughput, fetch_aia_sources
from sunerf.response.providers.eui import export_eui_throughput, fetch_eui_sources
from sunerf.response.providers.secchi import (
    export_secchi_euvi_throughput,
    fetch_secchi_sources,
)


THROUGHPUT_FILENAMES = {
    "aia": "aia.throughput.npz",
    "euvi_a": "euvi_a.throughput.npz",
    "euvi_b": "euvi_b.throughput.npz",
    "eui_fsi": "eui_fsi.throughput.npz",
}
INSTRUMENTS = tuple(THROUGHPUT_FILENAMES)


def _select_instruments(instruments: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = INSTRUMENTS if instruments is None else tuple(instruments)
    unknown = sorted(set(selected).difference(INSTRUMENTS))
    if unknown:
        raise ValueError(f"unsupported instruments: {unknown}")
    if not selected:
        raise ValueError("at least one instrument is required")
    if len(selected) != len(set(selected)):
        raise ValueError("instruments must be unique")
    return selected


def pipeline_paths(root: str | Path) -> dict[str, Path]:
    """Return the intentionally small on-disk layout used by this workflow."""
    root = Path(root)
    return {
        "root": root,
        "aia_sources": root / "instruments" / "aia",
        "secchi_sources": root / "instruments" / "secchi",
        "eui_sources": root / "instruments" / "eui",
        "throughputs": root / "throughputs",
        "responses": root / "responses",
    }


def fetch_instrument_inputs(
    root: str | Path,
    *,
    instruments: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Download only the official instrument wavelength-response inputs."""
    paths = pipeline_paths(root)
    selected = _select_instruments(instruments)
    products: dict[str, object] = {}
    if "aia" in selected:
        products["aia"] = fetch_aia_sources(paths["aia_sources"], force=force)
    spacecraft = tuple(
        spacecraft
        for instrument, spacecraft in (("euvi_a", "A"), ("euvi_b", "B"))
        if instrument in selected
    )
    if spacecraft:
        products["secchi_euvi"] = fetch_secchi_sources(
            paths["secchi_sources"], spacecraft=spacecraft, force=force
        )
    if "eui_fsi" in selected:
        products["eui_fsi"] = fetch_eui_sources(paths["eui_sources"], force=force)
    return products


def export_throughputs(
    root: str | Path,
    *,
    instruments: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Convert every instrument response to the same validated NPZ schema."""
    paths = pipeline_paths(root)
    output = paths["throughputs"]
    selected = _select_instruments(instruments)
    products = {key: output / THROUGHPUT_FILENAMES[key] for key in selected}
    if "aia" in selected:
        export_aia_throughput(paths["aia_sources"], products["aia"])
    for instrument, spacecraft in (("euvi_a", "A"), ("euvi_b", "B")):
        if instrument in selected:
            export_secchi_euvi_throughput(
                paths["secchi_sources"],
                products[instrument],
                spacecraft=spacecraft,
                secchi_prep_normalized_to_open=False,
            )
    if "eui_fsi" in selected:
        export_eui_throughput(paths["eui_sources"], products["eui_fsi"])
    return products


def prepare_instruments(
    root: str | Path,
    *,
    instruments: tuple[str, ...] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download the calibrations and immediately publish canonical throughputs."""
    selected = _select_instruments(instruments)
    fetch_instrument_inputs(root, instruments=selected, force=force)
    return export_throughputs(root, instruments=selected)


def resolve_throughput_paths(
    root: str | Path,
    instruments: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Resolve canonical throughput paths without downloading or exporting."""
    output = pipeline_paths(root)["throughputs"]
    selected = _select_instruments(instruments)
    return {key: output / THROUGHPUT_FILENAMES[key] for key in selected}


def build_responses(
    spectral_emissivity_file: str | Path,
    *,
    root: str | Path,
    output_dir: str | Path | None = None,
    label: str = "reference",
    instruments: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Fold one common emissivity grid through every instrument uniformly.

    The spectral artifact is loaded once.  Every instrument then follows the
    same interpolation, wavelength quadrature, unit conversion, and provenance
    path in :func:`sunerf.response.builder.fold_temperature_response`.
    """
    if not label or any(character in label for character in "/\\"):
        raise ValueError("label must be a non-empty filename-safe value")
    spectral = load_spectral_emissivity(spectral_emissivity_file)
    selected_paths = resolve_throughput_paths(root, instruments)

    output = pipeline_paths(root)["responses"] if output_dir is None else Path(output_dir)
    products: dict[str, Path] = {}
    for instrument, throughput_path in selected_paths.items():
        throughput = load_instrument_throughput(throughput_path)
        artifact = fold_temperature_response(spectral, throughput)
        destination = output / f"{instrument}_{label}.sunerf.npz"
        artifact.save(destination)
        products[instrument] = destination
    return products


def _json_paths(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _json_paths(item) for key, item in value.items()}
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download instrument responses, convert them to one throughput "
            "schema, and fold one shared emissivity grid uniformly."
        )
    )
    parser.add_argument(
        "--root",
        default="data/response_calibration",
        help="Workflow root (default: data/response_calibration).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("schema", help="Print the common NPZ schemas as JSON.")
    prepare = subparsers.add_parser(
        "prepare",
        help="Download and convert AIA, EUVI-A/B, and EUI/FSI responses.",
    )
    prepare.add_argument("--force", action="store_true")
    prepare.add_argument(
        "--instrument",
        choices=INSTRUMENTS,
        action="append",
        dest="instruments",
        help="Instrument to prepare; repeat as needed (default: all).",
    )

    build = subparsers.add_parser(
        "build",
        help="Fold one common spectral-emissivity NPZ through every instrument.",
    )
    build.add_argument("--spectral-emissivity", required=True)
    build.add_argument("--output-dir", default=None)
    build.add_argument("--label", default="reference")
    build.add_argument(
        "--instrument",
        choices=INSTRUMENTS,
        action="append",
        dest="instruments",
        help="Instrument to build; repeat as needed (default: all).",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "schema":
        print(json.dumps(input_schema_description(), indent=2, sort_keys=True))
        return
    if args.command == "prepare":
        products = prepare_instruments(
            args.root,
            instruments=None if args.instruments is None else tuple(args.instruments),
            force=args.force,
        )
    else:
        products = build_responses(
            args.spectral_emissivity,
            root=args.root,
            output_dir=args.output_dir,
            label=args.label,
            instruments=None if args.instruments is None else tuple(args.instruments),
        )
    print(json.dumps(_json_paths(products), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
