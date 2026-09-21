from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile

import h5py
import pytest

from sunerf.response.providers import chianti


def _ascii_tree(tmp_path, monkeypatch):
    root = tmp_path / "ascii"
    abundance = root / chianti.ABUNDANCE_RELATIVE_PATH
    ionization = root / chianti.IONIZATION_EQUILIBRIUM_RELATIVE_PATH
    abundance.parent.mkdir(parents=True)
    ionization.parent.mkdir(parents=True)
    (root / "VERSION").write_text("11.0.2\n", encoding="ascii")
    abundance.write_bytes(b"abundance\n")
    ionization.write_bytes(b"ionization\n")
    monkeypatch.setattr(
        chianti,
        "ABUNDANCE_SHA256",
        hashlib.sha256(abundance.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        chianti,
        "IONIZATION_EQUILIBRIUM_SHA256",
        hashlib.sha256(ionization.read_bytes()).hexdigest(),
    )
    return root


def test_official_archive_and_selected_atomic_files_are_pinned():
    source = chianti.CHRIANTI_ARCHIVE_SOURCE if False else chianti.CHIANTI_ARCHIVE_SOURCE
    assert source.url == (
        "https://download.chiantidatabase.org/CHIANTI_11.0.2_database.tar.gz"
    )
    assert source.size_bytes == 607_363_221
    assert source.sha256 == (
        "abbb8c418b0092439d692b72d92ead22a01c3d89bf2f3659d46a2cb81682cd79"
    )
    assert chianti.ABUNDANCE_RELATIVE_PATH.name == "sun_coronal_2021_chianti.abund"
    assert chianti.IONIZATION_EQUILIBRIUM_RELATIVE_PATH.name == "chianti.ioneq"


def test_ascii_validation_checks_version_and_selected_hashes(tmp_path, monkeypatch):
    root = _ascii_tree(tmp_path, monkeypatch)
    assert chianti.validate_chianti_ascii(root) == root

    (root / "VERSION").write_text("latest\n", encoding="ascii")
    with pytest.raises(ValueError, match="Expected CHIANTI ASCII version"):
        chianti.validate_chianti_ascii(root)


def test_tar_validation_rejects_traversal_and_special_members():
    safe = tarfile.TarInfo("abundance/file.abund")
    safe.type = tarfile.REGTYPE
    chianti._validate_tar_members([safe])

    traversal = tarfile.TarInfo("../escape")
    traversal.type = tarfile.REGTYPE
    with pytest.raises(ValueError, match="Unsafe path"):
        chianti._validate_tar_members([traversal])

    link = tarfile.TarInfo("link")
    link.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="unsupported non-file"):
        chianti._validate_tar_members([link])


def test_hdf5_builder_checks_hashes_and_publishes_atomically(tmp_path, monkeypatch):
    ascii_root = _ascii_tree(tmp_path, monkeypatch)
    calls = {}

    def fake_builder(root, output, **kwargs):
        calls["root"] = Path(root)
        calls["kwargs"] = kwargs
        with h5py.File(output, "w") as database:
            database.attrs["fiasco_version"] = "0.8.2"
            database.create_dataset("h/abundance/sun_coronal_2021_chianti", data=1.0)
            database.create_dataset("h/h_1/ioneq/chianti", data=1.0)
            database.create_dataset("ion_index", data=[b"H 1"])

    monkeypatch.setattr(chianti, "_load_fiasco_builder", lambda: fake_builder)
    output = tmp_path / "chianti.h5"
    assert chianti.build_fiasco_database(ascii_root, output, show_progress=False) == output
    assert output.is_file()
    assert calls == {
        "root": ascii_root,
        "kwargs": {
            "check_hash": True,
            "overwrite": False,
            "show_progress": False,
        },
    }
    assert not list(tmp_path.glob(".chianti.h5.*.tmp"))
