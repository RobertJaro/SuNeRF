import json

import numpy as np
import pytest

from sunerf import resources
from sunerf.absorption.builder import load_verner_parameters, verner_cross_section_cm2
from sunerf.resources import (
    normalize_artifact_reference,
    resolve_artifact_path,
    verify_resources,
)
from sunerf.resources.build import VERNER_1996_H_HE, write_verner_rows


def _packaged(tmp_path, monkeypatch):
    (tmp_path / "response").mkdir()
    artifact = tmp_path / "response" / "aia.sunerf.npz"
    artifact.write_bytes(b"response")
    manifest = {
        "schema": resources.MANIFEST_SCHEMA,
        "schema_version": resources.MANIFEST_SCHEMA_VERSION,
        "aliases": {"aia": "response/aia.sunerf.npz"},
        "files": {"response/aia.sunerf.npz": {"sha256": resources.sha256_file(artifact)}},
    }
    (tmp_path / resources.MANIFEST_NAME).write_text(json.dumps(manifest))
    monkeypatch.setattr(resources, "resource_root", lambda: tmp_path)
    return artifact


def test_builtin_references_resolve_and_stay_portable(tmp_path, monkeypatch):
    artifact = _packaged(tmp_path, monkeypatch)

    assert resolve_artifact_path("builtin:aia") == artifact
    assert resolve_artifact_path(str(artifact)) == artifact
    assert normalize_artifact_reference("builtin:aia") == "builtin:aia"
    assert normalize_artifact_reference("relative.npz").endswith("/relative.npz")
    assert resources.builtin_name("EUVI-A") == "euvi_a"
    with pytest.raises(FileNotFoundError, match="sunerf-resources build"):
        resolve_artifact_path("builtin:missing")
    with pytest.raises(ValueError, match="invalid packaged resource"):
        resolve_artifact_path("builtin:../secrets")


def test_manifest_verification_detects_modified_files(tmp_path, monkeypatch):
    artifact = _packaged(tmp_path, monkeypatch)
    verify_resources(tmp_path)

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_resources(tmp_path)


def test_packaged_verner_rows_reproduce_threshold_cross_sections(tmp_path):
    parameters = load_verner_parameters(write_verner_rows(tmp_path / "verner.dat"))

    assert len(VERNER_1996_H_HE) == 3
    assert verner_cross_section_cm2(910.0, parameters["H_I"]) == pytest.approx(6.3e-18, rel=0.02)
    assert verner_cross_section_cm2(504.0, parameters["He_I"]) == pytest.approx(7.4e-18, rel=0.02)
    assert verner_cross_section_cm2(227.8, parameters["He_II"]) == pytest.approx(1.6e-18, rel=0.02)
    # Photons below an ionization threshold are not absorbed by that species.
    assert verner_cross_section_cm2(304.0, parameters["He_II"]) == 0.0


def test_installed_resources_match_their_manifest_when_present():
    if not (resources.resource_root() / resources.MANIFEST_NAME).is_file():
        pytest.skip("packaged resources have not been built in this checkout")
    manifest = verify_resources()
    from sunerf.absorption import load_absorption_bundle
    from sunerf.response import load_response_artifact

    bundle = load_absorption_bundle("builtin:h_he_photoionization")
    for name in ("aia", "euvi_a", "euvi_b", "eui_fsi"):
        artifact = load_response_artifact(f"builtin:{name}")
        artifact.verify_response_id()
        assert artifact.log_density is not None and artifact.log_density.size > 1
        abundance = artifact.provenance["spectral_emissivity"]["abundance"]
        assert abundance["sha256"] == bundle.provenance["abundance"]["model"]["sha256"]
        assert abundance["sha256"] == manifest["abundance"]["sha256"]
    assert np.all(bundle.effective_cross_section_cm2 >= 0)
