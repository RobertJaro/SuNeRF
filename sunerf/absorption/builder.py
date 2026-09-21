"""Offline fetch, conversion, and construction of absorption bundles.

No function in this module is imported by the training pipeline.  Network
access is confined to the explicit ``fetch`` command; ``build`` consumes only
local, hash-recorded inputs and produces one immutable NumPy bundle.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

import numpy as np

from sunerf.absorption.artifact import ABSORPTION_SPECIES, AbsorptionBundle
from sunerf.response import load_instrument_throughput
from sunerf.response.builder import load_spectral_emissivity
from sunerf.response.numerics import trapezoid_node_weights


IONIZATION_INPUT_SCHEMA = "sunerf.h-he-ionization-fractions"
IONIZATION_INPUT_SCHEMA_VERSION = 1
VERNER_SOURCE_URL = "https://www.pa.uky.edu/~verner/dima/photo/photo.dat"
HC_EV_ANGSTROM = 12398.419843320026
MEGABARN_CM2 = 1.0e-18


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pinned_file(url: str, output, expected_sha256: str) -> Path:
    """Download one source atomically and reject content with the wrong hash."""
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urlopen(url) as source, tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{output.name}.", suffix=".tmp",
            dir=output.parent, delete=False,
        ) as destination:
            temporary = Path(destination.name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                destination.write(block)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise ValueError(
                f"download SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
        temporary.replace(output)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return output


def _json_scalar(value):
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError("JSON metadata must be scalar")
    return json.loads(str(value.reshape(()).item()))


def save_ionization_input(
    path,
    *,
    log_temperature,
    h_i,
    h_ii,
    he_i,
    he_ii,
    he_iii,
    provenance,
    metal_electron_per_hydrogen=None,
):
    """Write the small provider-neutral input consumed by ``build``."""
    from sunerf.response.artifact import atomic_savez_compressed

    log_temperature = np.asarray(log_temperature, dtype=np.float64)
    arrays = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in {
            "h_i": h_i, "h_ii": h_ii, "he_i": he_i,
            "he_ii": he_ii, "he_iii": he_iii,
        }.items()
    }
    _validate_ionization_arrays(log_temperature, arrays)
    if metal_electron_per_hydrogen is None:
        metal_electron_per_hydrogen = np.zeros_like(log_temperature)
    metal_electron_per_hydrogen = np.asarray(
        metal_electron_per_hydrogen, dtype=np.float64
    )
    if (
        metal_electron_per_hydrogen.shape != log_temperature.shape
        or not np.isfinite(metal_electron_per_hydrogen).all()
        or np.any(metal_electron_per_hydrogen < 0)
    ):
        raise ValueError("metal_electron_per_hydrogen must be finite and non-negative")
    provenance = dict(provenance)
    if not provenance:
        raise ValueError("ionization provenance must not be empty")
    atomic_savez_compressed(
        path,
        schema=np.asarray(IONIZATION_INPUT_SCHEMA),
        schema_version=np.asarray(IONIZATION_INPUT_SCHEMA_VERSION, dtype=np.int64),
        log_temperature=log_temperature,
        metal_electron_per_hydrogen=metal_electron_per_hydrogen,
        provenance_json=np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ),
        **arrays,
    )


def _validate_ionization_arrays(log_temperature, arrays):
    if (
        log_temperature.ndim != 1
        or log_temperature.size < 2
        or not np.isfinite(log_temperature).all()
        or np.any(np.diff(log_temperature) <= 0)
    ):
        raise ValueError("ionization log_temperature must be finite and increasing")
    for name, values in arrays.items():
        if values.shape != log_temperature.shape:
            raise ValueError(f"ionization {name} must contain one value per temperature")
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"ionization {name} must lie in [0, 1]")
    if not np.allclose(arrays["h_i"] + arrays["h_ii"], 1.0, rtol=0, atol=1e-6):
        raise ValueError("H I and H II fractions must sum to one")
    helium_sum = arrays["he_i"] + arrays["he_ii"] + arrays["he_iii"]
    if not np.allclose(helium_sum, 1.0, rtol=0, atol=1e-6):
        raise ValueError("He I, He II, and He III fractions must sum to one")


def load_ionization_input(path):
    required = {
        "schema", "schema_version", "log_temperature", "h_i", "h_ii",
        "he_i", "he_ii", "he_iii", "metal_electron_per_hydrogen",
        "provenance_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"ionization input is missing fields {missing}")
        if str(np.asarray(archive["schema"]).reshape(()).item()) != IONIZATION_INPUT_SCHEMA:
            raise ValueError("file is not a SuNeRF H/He ionization input")
        if int(np.asarray(archive["schema_version"]).reshape(())) != IONIZATION_INPUT_SCHEMA_VERSION:
            raise ValueError("unsupported ionization input version")
        log_temperature = np.asarray(archive["log_temperature"], dtype=np.float64)
        arrays = {
            name: np.asarray(archive[name], dtype=np.float64)
            for name in ("h_i", "h_ii", "he_i", "he_ii", "he_iii")
        }
        _validate_ionization_arrays(log_temperature, arrays)
        metal = np.asarray(archive["metal_electron_per_hydrogen"], dtype=np.float64)
        if metal.shape != log_temperature.shape or np.any(metal < 0) or not np.isfinite(metal).all():
            raise ValueError("invalid metal electron contribution")
        provenance = _json_scalar(archive["provenance_json"])
    return {"log_temperature": log_temperature, **arrays,
            "metal_electron_per_hydrogen": metal, "provenance": provenance}


def convert_ionization_csv(input_path, output_path, *, provenance):
    """Convert an exported CHIANTI/other table with stable named columns."""
    table = np.genfromtxt(input_path, delimiter=",", names=True, dtype=np.float64)
    required = ("log_temperature", "h_i", "h_ii", "he_i", "he_ii", "he_iii")
    missing = [name for name in required if name not in (table.dtype.names or ())]
    if missing:
        raise ValueError(f"ionization CSV is missing columns {missing}")
    metal = (
        table["metal_electron_per_hydrogen"]
        if "metal_electron_per_hydrogen" in table.dtype.names
        else None
    )
    save_ionization_input(
        output_path,
        log_temperature=table["log_temperature"],
        h_i=table["h_i"], h_ii=table["h_ii"],
        he_i=table["he_i"], he_ii=table["he_ii"], he_iii=table["he_iii"],
        metal_electron_per_hydrogen=metal,
        provenance=provenance,
    )


def build_chianti_ionization_input(
    output_path,
    *,
    hdf5_database,
    abundance_name,
    ionization_equilibrium_name,
    log_temperature=None,
    provenance=None,
):
    """Export H/He equilibrium fractions and electrons per hydrogen with FIASCO.

    Returns the He/H abundance of the selected abundance table so the bundle is
    built with the same composition as the temperature responses.
    """
    import astropy.units as u
    import fiasco

    if log_temperature is None:
        log_temperature = np.linspace(4.0, 8.0, 81)
    log_temperature = np.asarray(log_temperature, dtype=np.float64)
    temperature = 10.0**log_temperature * u.K
    kwargs = {
        "hdf5_dbase_root": str(hdf5_database),
        "abundance": abundance_name,
        "ionization_fraction": ionization_equilibrium_name,
    }
    fiasco.log.setLevel("ERROR")

    def fraction(name):
        ion = fiasco.Ion(name, temperature, **kwargs)
        values = np.asarray(ion.ionization_fraction, dtype=np.float64)
        return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0), ion

    h_i, _ = fraction("H 1")
    h_ii, _ = fraction("H 2")
    he_i, _ = fraction("He 1")
    he_ii, helium = fraction("He 2")
    he_iii, _ = fraction("He 3")
    # Tabulated fractions are rounded; renormalize each element exactly.
    hydrogen_sum = h_i + h_ii
    helium_sum = he_i + he_ii + he_iii
    h_i, h_ii = h_i / hydrogen_sum, h_ii / hydrogen_sum
    he_i, he_ii, he_iii = he_i / helium_sum, he_ii / helium_sum, he_iii / helium_sum
    helium_abundance = float(np.asarray(helium.abundance))

    proton_electron_ratio = np.asarray(
        fiasco.proton_electron_ratio(temperature, **kwargs), dtype=np.float64
    )
    hydrogen_helium_electrons = h_ii + helium_abundance * (he_ii + 2.0 * he_iii)
    # n_e / n_H = f(H II) / (n_p / n_e). Where hydrogen is neutral the ratio is
    # not defined by this identity; metals then supply the remaining electrons.
    with np.errstate(divide="ignore", invalid="ignore"):
        electron_per_hydrogen = np.where(
            proton_electron_ratio > 0, h_ii / proton_electron_ratio, np.nan
        )
    metal = np.nan_to_num(electron_per_hydrogen - hydrogen_helium_electrons, nan=0.0)
    metal = np.clip(metal, 0.0, None)

    full_provenance = {
        "provider": "fiasco",
        "version": str(fiasco.__version__),
        "hdf5_database": str(Path(hdf5_database).resolve()),
        "abundance": str(abundance_name),
        "ionization_equilibrium": str(ionization_equilibrium_name),
        "helium_per_hydrogen": helium_abundance,
        "assumption": "collisional_ionization_equilibrium",
    }
    full_provenance.update(dict(provenance or {}))
    save_ionization_input(
        output_path,
        log_temperature=log_temperature,
        h_i=h_i, h_ii=h_ii, he_i=he_i, he_ii=he_ii, he_iii=he_iii,
        metal_electron_per_hydrogen=metal,
        provenance=full_provenance,
    )
    return helium_abundance


def load_verner_parameters(path):
    """Return Verner-1996 rows for H I, He I, and He II."""
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 11:
        raise ValueError("Verner photo.dat must contain eleven numeric columns")
    identities = {"H_I": (1, 1), "He_I": (2, 2), "He_II": (2, 1)}
    selected = {}
    for species, (atomic_number, electron_count) in identities.items():
        rows = values[
            (values[:, 0] == atomic_number) & (values[:, 1] == electron_count)
        ]
        if rows.shape != (1, 11):
            raise ValueError(f"Verner table does not uniquely contain {species}")
        selected[species] = rows[0, 2:]
    return selected


def verner_cross_section_cm2(wavelength_angstrom, parameters):
    """Evaluate equation (1) of Verner et al. (1996)."""
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64)
    if np.any(wavelength <= 0) or not np.isfinite(wavelength).all():
        raise ValueError("wavelength must be finite and positive")
    threshold, maximum, e0, sigma0, ya, p, yw, y0, y1 = parameters
    energy = HC_EV_ANGSTROM / wavelength
    x = energy / e0 - y0
    y = np.sqrt(x * x + y1 * y1)
    profile = (
        ((x - 1.0) ** 2 + yw * yw)
        * np.power(y, 0.5 * p - 5.5)
        * np.power(1.0 + np.sqrt(y / ya), -p)
    )
    cross_section = sigma0 * profile * MEGABARN_CM2
    valid = (energy >= threshold) & (energy <= maximum)
    return np.where(valid, cross_section, 0.0)


def _reference_spectrum(spectral, channel_throughput):
    """Spectrum at the temperature where one channel is most sensitive.

    The first-order attenuation of a channel is weighted by the detected
    spectrum ``emissivity * throughput``. Its temperature dependence is weak
    compared with the wavelength dependence of the cross sections, so the
    spectrum at the peak of the channel response is used.
    """
    emissivity = spectral.emissivity
    if spectral.log_density is not None:
        density_index = int(np.argmin(np.abs(spectral.log_density - 9.0)))
        emissivity = emissivity[density_index]
    weights = trapezoid_node_weights(spectral.wavelength_angstrom)
    response = (emissivity * channel_throughput[None] * weights[None]).sum(axis=1)
    if not np.any(response > 0):
        raise ValueError("spectral emissivity has no overlap with a throughput channel")
    peak = int(np.argmax(response))
    return emissivity[peak], float(spectral.log_temperature[peak])


def _fold_channel_cross_sections(throughput, parameters, spectral=None):
    """Fold the cross sections over each channel.

    Without a spectral grid the weight is the instrument throughput. With one,
    it is the detected spectrum at the channel's peak-response temperature.
    """
    if spectral is None:
        wavelength = throughput.wavelength_angstrom
        channel_throughputs = throughput.throughput
    else:
        wavelength = spectral.wavelength_angstrom
        channel_throughputs = np.stack([
            np.interp(
                wavelength, throughput.wavelength_angstrom, values, left=0.0, right=0.0
            )
            for values in throughput.throughput
        ])
    weights = trapezoid_node_weights(wavelength)
    sigma = {
        species: verner_cross_section_cm2(wavelength, parameters[species])
        for species in ABSORPTION_SPECIES
    }
    folded = np.empty((len(throughput.channels), len(ABSORPTION_SPECIES)))
    reference_log_temperature = []
    for channel_index, channel_throughput in enumerate(channel_throughputs):
        spectrum_weight = channel_throughput
        if spectral is not None:
            spectrum, log_temperature = _reference_spectrum(spectral, channel_throughput)
            spectrum_weight = channel_throughput * spectrum
            reference_log_temperature.append(log_temperature)
        denominator = np.sum(spectrum_weight * weights)
        if denominator <= 0:
            raise ValueError("throughput channel has no positive wavelength integral")
        for species_index, species in enumerate(ABSORPTION_SPECIES):
            folded[channel_index, species_index] = np.sum(
                spectrum_weight * sigma[species] * weights
            ) / denominator
    return folded, reference_log_temperature


def build_absorption_bundle(
    *, verner_table_path, ionization_path, throughputs, helium_abundance,
    abundance_provenance, output_path=None, spectral_emissivity_path=None,
):
    """Build one bundle from local inputs; ``throughputs`` maps instrument to path.

    ``spectral_emissivity_path`` selects spectrum-weighted cross sections folded
    with the same spectral grid as the temperature responses.
    """
    helium_abundance = float(helium_abundance)
    if not np.isfinite(helium_abundance) or not 0 < helium_abundance < 1:
        raise ValueError("helium_abundance must be a finite number in (0, 1)")
    if not throughputs:
        raise ValueError("at least one instrument throughput is required")
    abundance_provenance = dict(abundance_provenance)
    missing_abundance = [
        field for field in ("name", "version", "sha256")
        if not str(abundance_provenance.get(field, "")).strip()
    ]
    if missing_abundance:
        raise ValueError(
            f"abundance_provenance is missing required fields {missing_abundance}"
        )
    abundance_sha256 = str(abundance_provenance["sha256"]).lower()
    if len(abundance_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in abundance_sha256
    ):
        raise ValueError("abundance provenance sha256 must be a hexadecimal SHA-256")
    abundance_provenance["sha256"] = abundance_sha256
    ionization = load_ionization_input(ionization_path)
    parameters = load_verner_parameters(verner_table_path)

    spectral = (
        None if spectral_emissivity_path is None
        else load_spectral_emissivity(spectral_emissivity_path)
    )
    instrument_keys = []
    channels = []
    cross_sections = []
    throughput_provenance = []
    for instrument_key, throughput_path in throughputs.items():
        throughput = load_instrument_throughput(throughput_path)
        folded, reference_log_temperature = _fold_channel_cross_sections(
            throughput, parameters, spectral
        )
        instrument_keys.extend([str(instrument_key)] * len(throughput.channels))
        channels.extend(throughput.channels)
        cross_sections.append(folded)
        throughput_provenance.append({
            "instrument_key": str(instrument_key),
            "path": str(Path(throughput_path).resolve()),
            "content_sha256": throughput.content_sha256,
            "calibration_epoch": throughput.calibration_epoch,
            "reference_log_temperature": reference_log_temperature,
        })

    electron_per_hydrogen = (
        ionization["h_ii"]
        + helium_abundance * (ionization["he_ii"] + 2.0 * ionization["he_iii"])
        + ionization["metal_electron_per_hydrogen"]
    )
    # Completely neutral endpoints are physically valid but cannot be inverted
    # from an electron-density field.  A tiny floor makes that limitation
    # explicit and numerically finite; a future total-mass-density model bypasses
    # this conversion entirely.
    electron_per_hydrogen = np.maximum(electron_per_hydrogen, 1.0e-12)
    provenance = {
        "builder": "sunerf.absorption.builder",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cross_sections": {
            "source": "Verner et al. 1996",
            "path": str(Path(verner_table_path).resolve()),
            "sha256": sha256_file(verner_table_path),
            "url": VERNER_SOURCE_URL,
        },
        "ionization": {
            "path": str(Path(ionization_path).resolve()),
            "sha256": sha256_file(ionization_path),
            "provenance": ionization["provenance"],
        },
        "abundance": {
            "model": abundance_provenance,
            "helium_per_hydrogen": helium_abundance,
        },
        "throughputs": throughput_provenance,
        "folding": (
            {
                "method": "throughput_weighted_cross_section",
                "spectral_weighting": "instrument_throughput_only",
                "quadrature": "trapezoid_nodes",
            }
            if spectral is None else {
                "method": "detected_spectrum_weighted_cross_section",
                "spectral_weighting": "emissivity_times_throughput_at_peak_response",
                "quadrature": "trapezoid_nodes",
                "spectral_emissivity_sha256": spectral.content_sha256,
                "spectral_emissivity_path": str(Path(spectral_emissivity_path).resolve()),
            }
        ),
        "ionization_assumption": "provider_table_equilibrium",
        "density_fallback": "runtime_hydrogen_density_convention",
    }
    bundle = AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=ionization["log_temperature"],
        ion_fraction=np.stack(
            [ionization["h_i"], ionization["he_i"], ionization["he_ii"]]
        ),
        electron_per_hydrogen=electron_per_hydrogen,
        abundance_per_hydrogen=np.asarray([1.0, helium_abundance, helium_abundance]),
        instrument_keys=tuple(instrument_keys),
        channels=tuple(channels),
        effective_cross_section_cm2=np.concatenate(cross_sections, axis=0),
        provenance=provenance,
    )
    if output_path is not None:
        bundle.save(output_path)
    return bundle


def _parse_throughputs(values):
    mapping = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--throughput must use INSTRUMENT_KEY=PATH")
        key, path = value.split("=", 1)
        if not key or not path or key in mapping:
            raise ValueError("throughput instrument keys must be non-empty and unique")
        mapping[key] = path
    return mapping


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fetch and build training-independent SuNeRF absorption bundles"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser("fetch", help="download one hash-pinned source")
    fetch.add_argument("--url", default=VERNER_SOURCE_URL)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--output", required=True)

    convert = subparsers.add_parser(
        "convert-ionization", help="convert a named-column CSV to the stable input schema"
    )
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--provider", required=True)
    convert.add_argument("--version", required=True)
    convert.add_argument("--source-sha256", required=True)

    ionization = subparsers.add_parser(
        "ionization", help="export H/He equilibrium fractions from a FIASCO CHIANTI database"
    )
    ionization.add_argument("--hdf5-database", required=True)
    ionization.add_argument("--abundance", required=True)
    ionization.add_argument("--ionization-equilibrium", default="chianti")
    ionization.add_argument("--output", required=True)

    build = subparsers.add_parser("build", help="build one immutable absorption bundle")
    build.add_argument("--verner-table", required=True)
    build.add_argument("--ionization", required=True)
    build.add_argument("--throughput", action="append", required=True)
    build.add_argument("--helium-abundance", type=float, required=True)
    build.add_argument("--abundance-name", required=True)
    build.add_argument("--abundance-version", required=True)
    build.add_argument("--abundance-sha256", required=True)
    build.add_argument("--output", required=True)
    build.add_argument(
        "--spectral-emissivity",
        default=None,
        help="Fold cross sections with this spectral grid instead of throughput only.",
    )

    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch_pinned_file(args.url, args.output, args.sha256)
    elif args.command == "ionization":
        helium_abundance = build_chianti_ionization_input(
            args.output,
            hdf5_database=args.hdf5_database,
            abundance_name=args.abundance,
            ionization_equilibrium_name=args.ionization_equilibrium,
        )
        print(f"helium_per_hydrogen={helium_abundance}")
    elif args.command == "convert-ionization":
        convert_ionization_csv(
            args.input,
            args.output,
            provenance={
                "provider": args.provider,
                "version": args.version,
                "source_sha256": args.source_sha256,
            },
        )
    else:
        bundle = build_absorption_bundle(
            verner_table_path=args.verner_table,
            ionization_path=args.ionization,
            throughputs=_parse_throughputs(args.throughput),
            helium_abundance=args.helium_abundance,
            abundance_provenance={
                "name": args.abundance_name,
                "version": args.abundance_version,
                "sha256": args.abundance_sha256,
            },
            output_path=args.output,
            spectral_emissivity_path=args.spectral_emissivity,
        )
        print(bundle.bundle_id)
