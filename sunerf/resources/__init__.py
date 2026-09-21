"""Physics resources shipped with SuNeRF.

The response tables, H/He ionization table, absorption bundle, abundance file,
and instrument throughputs in this directory are produced exclusively by the
offline builders (``sunerf-resources build``). Training and evaluation only
read them. A configuration selects a packaged artifact with ``builtin:<name>``
instead of a filesystem path.
"""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources
from pathlib import Path

BUILTIN_PREFIX = "builtin:"
MANIFEST_NAME = "manifest.json"
MANIFEST_SCHEMA = "sunerf.resources.manifest"
MANIFEST_SCHEMA_VERSION = 1
# Search order for a bare ``builtin:<name>`` without a manifest alias.
_CANDIDATE_PATTERNS = (
    "response/{name}.sunerf.npz",
    "absorption/{name}.npz",
    "throughput/{name}.throughput.npz",
    "spectral/{name}.spectral.npz",
    "ionization/{name}.npz",
)


def resource_root() -> Path:
    """Return the installed resource directory."""
    return Path(str(resources.files(__name__)))


def is_builtin(value) -> bool:
    return isinstance(value, str) and value.startswith(BUILTIN_PREFIX)


def load_manifest(root: str | Path | None = None) -> dict:
    root = resource_root() if root is None else Path(root)
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"SuNeRF resource manifest is missing: {path}. Run 'sunerf-resources build --install'."
        )
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a SuNeRF resource manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported SuNeRF resource manifest version")
    return manifest


def builtin_name(instrument_key: str) -> str:
    """Canonical packaged name of an instrument key, e.g. ``EUVI-A`` -> ``euvi_a``."""
    return str(instrument_key).strip().lower().replace("-", "_").replace(" ", "_")


def resolve_artifact_path(value) -> Path:
    """Resolve a filesystem path or a ``builtin:<name>`` reference to a file."""
    if not is_builtin(value):
        return Path(os.fspath(value))
    name = value[len(BUILTIN_PREFIX):].strip()
    if not name or ".." in Path(name).parts or Path(name).is_absolute():
        raise ValueError(f"invalid packaged resource reference {value!r}")
    root = resource_root()
    try:
        aliases = load_manifest(root).get("aliases", {})
    except FileNotFoundError:
        aliases = {}
    candidates = []
    if name in aliases:
        candidates.append(root / aliases[name])
    candidates.append(root / name)
    candidates.extend(root / pattern.format(name=name) for pattern in _CANDIDATE_PATTERNS)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"packaged SuNeRF resource {value!r} does not exist under {root}. "
        "Run 'sunerf-resources build --install' or configure an explicit path."
    )


def normalize_artifact_reference(value):
    """Keep ``builtin:`` references portable and make filesystem paths absolute."""
    if is_builtin(value):
        return value
    return os.path.abspath(os.fspath(value))


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(resolve_artifact_path(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resources(root: str | Path | None = None) -> dict:
    """Check every packaged file against the manifest; return the manifest."""
    root = resource_root() if root is None else Path(root)
    manifest = load_manifest(root)
    problems = []
    for relative, record in sorted(manifest.get("files", {}).items()):
        path = root / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif sha256_file(path) != record["sha256"]:
            problems.append(f"hash mismatch {relative}")
    for alias, relative in sorted(manifest.get("aliases", {}).items()):
        if relative not in manifest.get("files", {}):
            problems.append(f"alias {alias!r} points to unlisted file {relative}")
    if problems:
        raise ValueError("SuNeRF resources do not match their manifest: " + "; ".join(problems))
    return manifest


__all__ = [
    "BUILTIN_PREFIX",
    "builtin_name",
    "is_builtin",
    "load_manifest",
    "normalize_artifact_reference",
    "resolve_artifact_path",
    "resource_root",
    "sha256_file",
    "verify_resources",
]
