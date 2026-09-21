"""Verified CHIANTI 11.0.2 acquisition and FIASCO database construction.

The atomic release is intentionally separate from instrument calibration.
Fetching pins the official CHIANTI archive byte-for-byte, extraction rejects
unsafe tar members, and the FIASCO HDF5 database is published atomically only
after FIASCO's per-file hash checks have succeeded.
"""

from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

import h5py

from sunerf.response.providers.base import (
    OptionalProviderDependencyError,
    SourceFile,
    fetch_source,
    sha256_file,
    verify_source,
)


CHIANTI_VERSION = "11.0.2"
FIASCO_VERSION = "0.8.2"
CHIANTI_ARCHIVE_SOURCE = SourceFile(
    key="chianti_ascii_archive",
    filename="CHIANTI_11.0.2_database.tar.gz",
    url="https://download.chiantidatabase.org/CHIANTI_11.0.2_database.tar.gz",
    sha256="abbb8c418b0092439d692b72d92ead22a01c3d89bf2f3659d46a2cb81682cd79",
    size_bytes=607_363_221,
    version=CHIANTI_VERSION,
    description="Official complete CHIANTI 11.0.2 ASCII atomic database",
    reference_url="https://www.chiantidatabase.org/chianti_download.html",
)
ABUNDANCE_RELATIVE_PATH = Path("abundance/sun_coronal_2021_chianti.abund")
ABUNDANCE_SHA256 = "396f76c5accb9178069ec6e89b78d06f0b8ebb5ec5567e2a10f35eff12ccaa40"
IONIZATION_EQUILIBRIUM_RELATIVE_PATH = Path("ioneq/chianti.ioneq")
IONIZATION_EQUILIBRIUM_SHA256 = (
    "706029c824704ab727c40570a8709bae42bf69745db2fa5cd54d5b8b12bc41db"
)


def fetch_chianti_archive(
    destination_dir: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 600.0,
) -> Path:
    """Download and verify the official CHIANTI 11.0.2 archive."""
    destination_dir = Path(destination_dir)
    path = fetch_source(
        CHIANTI_ARCHIVE_SOURCE,
        destination_dir,
        force=force,
        timeout_seconds=timeout_seconds,
    )
    return path


def validate_chianti_ascii(ascii_root: str | Path) -> Path:
    """Validate the version and selected science inputs in an ASCII tree."""
    ascii_root = Path(ascii_root)
    version_path = ascii_root / "VERSION"
    if not version_path.is_file():
        raise FileNotFoundError(f"CHIANTI VERSION file is missing: {version_path}")
    version = version_path.read_text(encoding="ascii").strip()
    if version != CHIANTI_VERSION:
        raise ValueError(
            f"Expected CHIANTI ASCII version {CHIANTI_VERSION}, received {version!r}"
        )
    selected = (
        (ABUNDANCE_RELATIVE_PATH, ABUNDANCE_SHA256, "coronal abundance"),
        (
            IONIZATION_EQUILIBRIUM_RELATIVE_PATH,
            IONIZATION_EQUILIBRIUM_SHA256,
            "ionization equilibrium",
        ),
    )
    for relative_path, expected_digest, label in selected:
        path = ascii_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"CHIANTI {label} file is missing: {path}")
        digest = sha256_file(path)
        if digest != expected_digest:
            raise ValueError(
                f"CHIANTI {label} SHA-256 verification failed for {path}: "
                f"expected {expected_digest}, received {digest}"
            )
    return ascii_root


def _validate_tar_members(members: list[tarfile.TarInfo]) -> None:
    for member in members:
        name = PurePosixPath(member.name)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe path in CHIANTI archive: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                "CHIANTI archive contains an unsupported non-file member: "
                f"{member.name!r}"
            )


def extract_chianti_ascii(
    archive_path: str | Path,
    ascii_root: str | Path,
) -> Path:
    """Safely extract a verified archive, reusing a valid existing tree."""
    archive_path = verify_source(archive_path, CHIANTI_ARCHIVE_SOURCE)
    ascii_root = Path(ascii_root)
    if ascii_root.exists():
        return validate_chianti_ascii(ascii_root)

    ascii_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{ascii_root.name}.", dir=ascii_root.parent)
    )
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            _validate_tar_members(members)
            archive.extractall(temporary_root, members=members)
        validate_chianti_ascii(temporary_root)
        os.replace(temporary_root, ascii_root)
        temporary_root = None
    finally:
        if temporary_root is not None:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return ascii_root


def _load_fiasco_builder():
    if sys.version_info < (3, 12):
        raise OptionalProviderDependencyError(
            "Building the pinned atomic database requires Python >=3.12 and "
            f"fiasco=={FIASCO_VERSION}."
        )
    try:
        installed = importlib.metadata.version("fiasco")
    except importlib.metadata.PackageNotFoundError as error:
        raise OptionalProviderDependencyError(
            f"Building the atomic database requires fiasco=={FIASCO_VERSION}."
        ) from error
    if installed != FIASCO_VERSION:
        raise OptionalProviderDependencyError(
            f"Expected fiasco=={FIASCO_VERSION}, found fiasco=={installed}."
        )
    from sunerf.response.providers.fiasco import import_fiasco_offline

    import_fiasco_offline()
    from fiasco.util import build_hdf5_dbase

    return build_hdf5_dbase


def validate_fiasco_database(path: str | Path) -> Path:
    """Validate the generated database identity and required parsed datasets."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"FIASCO HDF5 database is missing: {path}")
    with h5py.File(path, mode="r") as database:
        version = str(database.attrs.get("fiasco_version", ""))
        if version != FIASCO_VERSION:
            raise ValueError(
                f"Expected FIASCO database builder {FIASCO_VERSION}, received {version!r}"
            )
        for dataset in (
            "h/abundance/sun_coronal_2021_chianti",
            "h/h_1/ioneq/chianti",
            "ion_index",
        ):
            if dataset not in database:
                raise ValueError(f"FIASCO HDF5 database is missing {dataset}")
    return path


def build_fiasco_database(
    ascii_root: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """Build and atomically publish a hash-checked FIASCO HDF5 database."""
    ascii_root = validate_chianti_ascii(ascii_root)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return validate_fiasco_database(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder = _load_fiasco_builder()

    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
    temporary_path.unlink()
    try:
        builder(
            ascii_root,
            temporary_path,
            check_hash=True,
            overwrite=False,
            show_progress=show_progress,
        )
        validate_fiasco_database(temporary_path)
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def prepare_chianti_database(
    root: str | Path,
    *,
    force_download: bool = False,
    rebuild_hdf5: bool = False,
    show_progress: bool = True,
) -> dict[str, Path]:
    """Fetch, extract, and index the complete pinned atomic release."""
    root = Path(root)
    archive = fetch_chianti_archive(root, force=force_download)
    ascii_root = extract_chianti_ascii(archive, root / "ascii")
    database = build_fiasco_database(
        ascii_root,
        root / f"chianti_{CHIANTI_VERSION}.h5",
        overwrite=rebuild_hdf5,
        show_progress=show_progress,
    )
    return {
        "archive": archive,
        "ascii_root": ascii_root,
        "database": database,
        "abundance": ascii_root / ABUNDANCE_RELATIVE_PATH,
        "ionization_equilibrium": ascii_root
        / IONIZATION_EQUILIBRIUM_RELATIVE_PATH,
    }


__all__ = [
    "ABUNDANCE_RELATIVE_PATH",
    "ABUNDANCE_SHA256",
    "CHIANTI_ARCHIVE_SOURCE",
    "CHIANTI_VERSION",
    "FIASCO_VERSION",
    "IONIZATION_EQUILIBRIUM_RELATIVE_PATH",
    "IONIZATION_EQUILIBRIUM_SHA256",
    "build_fiasco_database",
    "extract_chianti_ascii",
    "fetch_chianti_archive",
    "prepare_chianti_database",
    "validate_chianti_ascii",
    "validate_fiasco_database",
]
