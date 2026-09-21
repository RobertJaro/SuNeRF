"""Shared primitives for reproducible response-calibration providers.

Provider modules describe upstream files with immutable content hashes.  The
download helper publishes a file only after its byte count and SHA-256 digest
have been verified, so a mutable upstream URL cannot silently change a response
release.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping
from urllib.request import Request, urlopen


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_USER_AGENT = "SuNeRF-response-provider/1"


class SourceVerificationError(ValueError):
    """Raised when downloaded or cached calibration bytes are not the pin."""


class OptionalProviderDependencyError(ImportError):
    """Raised when an exporter needs an intentionally optional dependency."""


@dataclass(frozen=True)
class SourceFile:
    """One byte-exact upstream input used by a response provider."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    version: str
    description: str
    reference_url: str | None = None

    def __post_init__(self):
        for name in ("key", "filename", "url", "version", "description"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SourceFile.{name} must be a non-empty string")
        if Path(self.filename).name != self.filename:
            raise ValueError("SourceFile.filename must be a basename")
        digest = self.sha256.lower()
        if not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError("SourceFile.sha256 must contain 64 hexadecimal characters")
        if not isinstance(self.size_bytes, int) or self.size_bytes <= 0:
            raise ValueError("SourceFile.size_bytes must be a positive integer")
        object.__setattr__(self, "sha256", digest)

    def as_provenance(self) -> dict[str, object]:
        """Return stable source metadata without machine-local paths or times."""
        value = asdict(self)
        value.pop("key")
        return {key: item for key, item in value.items() if item is not None}


def sha256_file(path: str | Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(path: str | Path, source: SourceFile) -> Path:
    """Validate a local file against a provider's byte-exact source pin."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required response source is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != source.size_bytes:
        raise SourceVerificationError(
            f"{source.key} size verification failed for {path}: "
            f"expected {source.size_bytes}, received {actual_size}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != source.sha256:
        raise SourceVerificationError(
            f"{source.key} SHA-256 verification failed for {path}: "
            f"expected {source.sha256}, received {actual_digest}"
        )
    return path


def fetch_source(
    source: SourceFile,
    destination_dir: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> Path:
    """Download, verify, and atomically publish one calibration source.

    A valid cached file is reused.  A mismatching cached file is never replaced
    unless ``force`` is explicitly requested; even then the replacement is
    published only after verification succeeds.
    """
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.filename
    if destination.exists() and not force:
        return verify_source(destination, source)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{source.filename}.",
            suffix=".download",
            dir=destination_dir,
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            request = Request(source.url, headers={"User-Agent": _USER_AGENT})
            digest = hashlib.sha256()
            byte_count = 0
            with urlopen(request, timeout=timeout_seconds) as response:
                while chunk := response.read(_DOWNLOAD_CHUNK_SIZE):
                    output.write(chunk)
                    digest.update(chunk)
                    byte_count += len(chunk)
            output.flush()
            os.fsync(output.fileno())

        if byte_count != source.size_bytes:
            raise SourceVerificationError(
                f"{source.key} download size verification failed: expected "
                f"{source.size_bytes}, received {byte_count}"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != source.sha256:
            raise SourceVerificationError(
                f"{source.key} download SHA-256 verification failed: expected "
                f"{source.sha256}, received {actual_digest}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def fetch_sources(
    sources: Iterable[SourceFile],
    destination_dir: str | Path,
    *,
    force: bool = False,
    timeout_seconds: float = 120.0,
) -> dict[str, Path]:
    """Fetch a set of uniquely keyed sources into one directory."""
    sources = tuple(sources)
    keys = [source.key for source in sources]
    if len(keys) != len(set(keys)):
        raise ValueError("Response-provider source keys must be unique")
    return {
        source.key: fetch_source(
            source,
            destination_dir,
            force=force,
            timeout_seconds=timeout_seconds,
        )
        for source in sources
    }


def require_source_paths(
    sources: Iterable[SourceFile],
    directory: str | Path,
) -> Mapping[str, Path]:
    """Resolve and validate all provider inputs without network access."""
    directory = Path(directory)
    return {
        source.key: verify_source(directory / source.filename, source)
        for source in sources
    }
