"""Rebuild and install the physics resources shipped with SuNeRF.

``build`` runs the offline builders from scratch (CHIANTI spectral grid,
instrument throughputs, response tables, H/He ionization table, absorption
bundle). ``--install`` copies the finished products into ``sunerf/resources``
and writes the manifest. ``verify`` checks the installed files and needs neither
the atomic database nor the build extras.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sunerf.resources import (
    MANIFEST_NAME,
    MANIFEST_SCHEMA,
    MANIFEST_SCHEMA_VERSION,
    resource_root,
    sha256_file,
    verify_resources,
)

INSTRUMENTS = ("aia", "euvi_a", "euvi_b", "eui_fsi")
ABSORPTION_NAME = "h_he_photoionization"
IONIZATION_NAME = "h_he_chianti"
VERNER_NAME = "verner96_h_he.dat"
# Verner, Ferland, Korista & Yakovlev (1996, ApJ 465, 487), Table 1 rows for
# H I, He I and He II: Z, N, E_th [eV], E_max [eV], E_0 [eV], sigma_0 [Mb],
# y_a, P, y_w, y_0, y_1.
VERNER_1996_H_HE = (
    (1, 1, 1.360e1, 5.000e4, 4.298e-1, 5.475e4, 3.288e1, 2.963, 0.0, 0.0, 0.0),
    (2, 2, 2.459e1, 5.000e4, 1.361e1, 9.492e2, 1.469, 3.188, 2.039, 4.434e-1, 2.136),
    (2, 1, 5.442e1, 5.000e4, 1.720, 1.369e4, 3.288e1, 2.963, 0.0, 0.0, 0.0),
)
LICENSES = """# Packaged physics resources: sources and attribution

- CHIANTI atomic database (Dere et al. 1997; Dufresne et al. 2024), used through
  fiasco. CHIANTI is a collaborative project involving George Mason University,
  the University of Michigan (USA), University of Cambridge (UK) and NASA Goddard
  Space Flight Center (USA). The abundance file is redistributed unchanged.
- Photoionization cross sections: Verner, Ferland, Korista & Yakovlev (1996),
  ApJ 465, 487 (three fit-parameter rows for H I, He I and He II).
- SDO/AIA effective areas: AIA team / SolarSoft calibration (Boerner et al. 2012).
- STEREO/SECCHI EUVI effective areas: SECCHI team calibration files.
- Solar Orbiter/EUI FSI spectral calibration: EUI consortium public release.

All derived tables in this directory are produced by `sunerf-resources build`;
`manifest.json` records their hashes and builder provenance.
"""


def write_verner_rows(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "Verner et al. 1996 (ApJ 465, 487) fit parameters for H I, He I, He II\n"
        "Z N E_th[eV] E_max[eV] E_0[eV] sigma_0[Mb] y_a P y_w y_0 y_1"
    )
    np.savetxt(path, np.asarray(VERNER_1996_H_HE, dtype=np.float64), fmt="%.6e", header=header)
    return path


def _spectral_grid(root: Path, *, overwrite: bool, show_progress: bool) -> Path:
    from sunerf.response.emissivity import default_output_path, generate_spectral_emissivity

    try:
        return generate_spectral_emissivity(
            root, overwrite=overwrite, show_progress=show_progress
        )
    except ValueError:
        if not default_output_path(root).exists():
            raise
        # An existing grid on superseded axes is rebuilt rather than reused.
        return generate_spectral_emissivity(root, overwrite=True, show_progress=show_progress)


def build_resources(
    root,
    *,
    fetch: bool = False,
    overwrite_spectral: bool = False,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Run every offline builder and return the produced files by role."""
    from sunerf.absorption.builder import (
        build_absorption_bundle,
        build_chianti_ionization_input,
    )
    from sunerf.response.builder import load_spectral_emissivity
    from sunerf.response.pipeline import (
        build_responses,
        export_throughputs,
        fetch_instrument_inputs,
        resolve_throughput_paths,
    )
    from sunerf.response.providers.chianti import (
        ABUNDANCE_RELATIVE_PATH,
        CHIANTI_VERSION,
        prepare_chianti_database,
    )
    from sunerf.response.providers.fiasco import (
        DEFAULT_ABUNDANCE,
        DEFAULT_IONIZATION_EQUILIBRIUM,
    )

    root = Path(root)
    spectral_path = _spectral_grid(
        root, overwrite=overwrite_spectral, show_progress=show_progress
    )
    spectral = load_spectral_emissivity(spectral_path)

    throughputs = resolve_throughput_paths(root, INSTRUMENTS)
    if fetch or not all(path.is_file() for path in throughputs.values()):
        fetch_instrument_inputs(root, instruments=INSTRUMENTS, force=fetch)
        throughputs = export_throughputs(root, instruments=INSTRUMENTS)

    build_directory = root / "resources_build"
    responses = build_responses(
        spectral_path, root=root, output_dir=build_directory / "response",
        label="reference", instruments=INSTRUMENTS,
    )

    sources = prepare_chianti_database(
        root / "chianti" / CHIANTI_VERSION, show_progress=show_progress
    )
    ionization_path = build_directory / "ionization" / f"{IONIZATION_NAME}_{CHIANTI_VERSION}.npz"
    helium_abundance = build_chianti_ionization_input(
        ionization_path,
        hdf5_database=sources["database"],
        abundance_name=DEFAULT_ABUNDANCE,
        ionization_equilibrium_name=DEFAULT_IONIZATION_EQUILIBRIUM,
        log_temperature=spectral.log_temperature,
        provenance={
            "abundance_sha256": spectral.provenance["abundance"]["sha256"],
            "ionization_equilibrium_sha256": spectral.provenance[
                "ionization_equilibrium"
            ]["sha256"],
        },
    )
    verner_path = write_verner_rows(build_directory / "atomic" / VERNER_NAME)
    abundance = spectral.provenance["abundance"]
    absorption_path = build_directory / "absorption" / f"{ABSORPTION_NAME}.npz"
    build_absorption_bundle(
        verner_table_path=verner_path,
        ionization_path=ionization_path,
        throughputs={key: str(path) for key, path in throughputs.items()},
        helium_abundance=helium_abundance,
        abundance_provenance={
            key: abundance[key] for key in ("name", "version", "sha256")
        },
        spectral_emissivity_path=spectral_path,
        output_path=absorption_path,
    )
    return {
        "spectral": spectral_path,
        "abundance": Path(sources["abundance"]),
        "ionization": ionization_path,
        "verner": verner_path,
        "absorption": absorption_path,
        **{f"throughput/{key}": Path(path) for key, path in throughputs.items()},
        **{f"response/{key}": Path(path) for key, path in responses.items()},
        "_abundance_relative": Path(ABUNDANCE_RELATIVE_PATH),
    }


def install_resources(products: dict[str, Path], destination=None) -> Path:
    """Copy built products into the package and write the manifest."""
    from sunerf.absorption import load_absorption_bundle
    from sunerf.response import load_response_artifact
    from sunerf.response.builder import load_spectral_emissivity

    destination = resource_root() if destination is None else Path(destination)
    layout = {
        f"spectral/{products['spectral'].name}": products["spectral"],
        f"abundance/{products['abundance'].name}": products["abundance"],
        f"ionization/{products['ionization'].name}": products["ionization"],
        f"atomic/{products['verner'].name}": products["verner"],
        f"absorption/{products['absorption'].name}": products["absorption"],
    }
    aliases = {ABSORPTION_NAME: f"absorption/{products['absorption'].name}"}
    for key in INSTRUMENTS:
        layout[f"throughput/{key}.throughput.npz"] = products[f"throughput/{key}"]
        layout[f"response/{key}.sunerf.npz"] = products[f"response/{key}"]
        aliases[key] = f"response/{key}.sunerf.npz"

    for folder in {Path(relative).parts[0] for relative in layout}:
        shutil.rmtree(destination / folder, ignore_errors=True)
    files = {}
    for relative, source in sorted(layout.items()):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        record = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
        if relative.startswith("response/"):
            record["response_id"] = load_response_artifact(target).response_id
        elif relative.startswith("absorption/"):
            record["bundle_id"] = load_absorption_bundle(target).bundle_id
        elif relative.startswith("spectral/"):
            record["content_sha256"] = load_spectral_emissivity(target).content_sha256
        files[relative] = record

    spectral = load_spectral_emissivity(products["spectral"])
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "atomic_database": spectral.provenance["atomic_database"],
        "abundance": spectral.provenance["abundance"],
        "ionization_equilibrium": spectral.provenance["ionization_equilibrium"],
        "axes": {
            "log10_temperature_K": [
                float(spectral.log_temperature[0]), float(spectral.log_temperature[-1]),
                int(spectral.log_temperature.size),
            ],
            "log10_density_cm3": (
                None if spectral.log_density is None else [
                    float(spectral.log_density[0]), float(spectral.log_density[-1]),
                    int(spectral.log_density.size),
                ]
            ),
        },
        "aliases": aliases,
        "files": files,
    }
    (destination / "LICENSES.md").write_text(LICENSES, encoding="utf-8")
    with open(destination / MANIFEST_NAME, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    verify_resources(destination)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="rebuild every resource from scratch")
    build.add_argument("--root", default="data/response_calibration")
    build.add_argument("--fetch", action="store_true", help="re-download instrument calibrations")
    build.add_argument("--overwrite-spectral", action="store_true")
    build.add_argument("--install", action="store_true", help="copy products into the package")
    build.add_argument("--no-progress", action="store_true")
    subparsers.add_parser("verify", help="check packaged files against the manifest")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        manifest = verify_resources()
        print(f"{len(manifest['files'])} packaged resources verified under {resource_root()}")
        return 0
    products = build_resources(
        args.root,
        fetch=args.fetch,
        overwrite_spectral=args.overwrite_spectral,
        show_progress=not args.no_progress,
    )
    products.pop("_abundance_relative")
    if args.install:
        print(install_resources(products))
    else:
        print(json.dumps({key: str(value) for key, value in products.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
