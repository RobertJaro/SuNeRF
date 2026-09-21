"""Load PSI plasma cubes as a directly renderable spherical grid."""

from __future__ import annotations

import glob
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sunerf.data.psi.read_psi import read_PSI_fields
from sunerf.model.spherical_grid import SphericalGridPlasmaModel


@dataclass(frozen=True)
class PSIGrid:
    """A grid-backed plasma model and its normalized time metadata."""

    model: SphericalGridPlasmaModel
    normalized_times: np.ndarray
    observation_times: tuple[datetime, ...]
    ref_date: datetime
    source_provenance: dict


def _frame_id(path):
    match = re.search(r"(\d+)$", Path(path).stem)
    if match is None:
        raise ValueError(f"Could not infer a numeric PSI frame id from {path}")
    return int(match.group(1))


def _index_unique_frames(paths, field_name):
    indexed = {}
    for path in paths:
        frame_id = _frame_id(path)
        if frame_id in indexed:
            raise ValueError(f"Duplicate {field_name} file for PSI frame {frame_id}")
        indexed[frame_id] = path
    return indexed


def pair_psi_files(data_path, n_frames=None, frame_ids=None):
    """Pair PSI density and temperature files by frame id.

    ``data_path`` is the directory that contains the ``rho/`` and ``t/``
    folders. ``frame_ids`` selects specific snapshots instead of the first
    ``n_frames``.
    """
    data_path = Path(data_path)
    missing_folders = [name for name in ("rho", "t") if not (data_path / name).is_dir()]
    if missing_folders:
        raise FileNotFoundError(
            f"PSI data path {data_path} must contain 'rho' and 't' folders; "
            f"missing {missing_folders}"
        )
    rho = _index_unique_frames(glob.glob(str(Path(data_path) / "rho" / "*.h5")), "density")
    temperature = _index_unique_frames(
        glob.glob(str(Path(data_path) / "t" / "*.h5")), "temperature"
    )
    missing_temperature = sorted(set(rho) - set(temperature))
    missing_density = sorted(set(temperature) - set(rho))
    if missing_temperature or missing_density:
        raise ValueError(
            "Unpaired PSI plasma frames: "
            f"missing temperature={missing_temperature}, missing density={missing_density}"
        )
    available = sorted(rho)
    if not available:
        raise FileNotFoundError(f"No paired PSI .h5 files found below {data_path}")
    if frame_ids is not None:
        if n_frames is not None:
            raise ValueError("frame_ids and n_frames are mutually exclusive")
        frame_ids = [int(frame_id) for frame_id in frame_ids]
        unknown = sorted(set(frame_ids) - set(available))
        if unknown or not frame_ids or len(set(frame_ids)) != len(frame_ids):
            raise ValueError(
                f"frame_ids must be unique available PSI frames; unknown {unknown}"
            )
        frame_ids = sorted(frame_ids)
    else:
        frame_ids = available
    if n_frames is not None:
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        frame_ids = frame_ids[:n_frames]
    return [(rho[frame_id], temperature[frame_id]) for frame_id in frame_ids]


def _read_psi_pair(args):
    rho_path, temperature_path, read_options = args
    return read_PSI_fields(rho_path, temperature_path, **read_options)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_psi_frames(read_args, workers):
    if workers and workers > 1:
        with Pool(workers) as pool:
            yield from pool.imap(_read_psi_pair, read_args)
    else:
        for args in read_args:
            yield _read_psi_pair(args)


def _sort_axis(axis, fields, dimension, name):
    order = np.argsort(axis)
    sorted_axis = np.asarray(axis, dtype=np.float32)[order]
    if sorted_axis.size > 1 and np.any(np.diff(sorted_axis) <= 0):
        raise ValueError(f"PSI {name} coordinates must be unique")
    return sorted_axis, np.take(fields, order, axis=dimension)


def _normalize_longitude(longitude, fields, dimension, period=2 * np.pi):
    longitude = np.asarray(longitude, dtype=np.float64)
    origin = float(np.min(longitude))
    wrapped = np.remainder(longitude - origin, period) + origin
    order = np.argsort(wrapped)
    wrapped = wrapped[order]
    fields = np.take(fields, order, axis=dimension)

    # Many simulation grids contain both 0 and 2*pi. They are the same periodic
    # cell, so retaining both would create a zero-width interpolation interval.
    keep = np.ones(wrapped.shape, dtype=bool)
    if wrapped.size > 1:
        keep[1:] = ~np.isclose(np.diff(wrapped), 0.0, rtol=0.0, atol=1e-6)
    wrapped = wrapped[keep]
    fields = np.take(fields, np.flatnonzero(keep), axis=dimension)
    if wrapped.size == 0:
        raise ValueError("PSI longitude axis is empty")
    return wrapped.astype(np.float32), fields


def load_psi_grid(
    data_path,
    *,
    log_T,
    density_unit_scale_cm3,
    temperature_unit_scale_K,
    reference_frame_id,
    longitude_frame,
    Rs_per_ds=1.0,
    seconds_per_dt=86400.0,
    time_step_seconds=3600.0,
    n_frames=None,
    frame_ids=None,
    workers=0,
    ref_date=datetime(2025, 1, 1),
    min_radius=1.0,
    max_radius=2.6,
):
    """Create a grid-backed model directly from paired PSI simulation cubes.

    Source density and temperature values are converted exactly once by the
    required unit scales. ``ref_date`` is the time of ``reference_frame_id``.
    ``longitude_frame`` declares the physical frame of the source longitude
    axis; it is metadata, not an implicit coordinate transform.
    """
    if Rs_per_ds <= 0 or seconds_per_dt <= 0 or time_step_seconds <= 0:
        raise ValueError("distance and time scales must be positive")
    for name, value in (
        ("density_unit_scale_cm3", density_unit_scale_cm3),
        ("temperature_unit_scale_K", temperature_unit_scale_K),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if isinstance(reference_frame_id, (bool, np.bool_)) or not isinstance(
        reference_frame_id, (int, np.integer)
    ):
        raise TypeError("reference_frame_id must be an integer")
    longitude_frame = str(longitude_frame).strip().lower()
    if longitude_frame != "carrington":
        raise ValueError(
            "longitude_frame must be 'carrington'; other frames require an explicit "
            "coordinate transform that the PSI grid loader does not apply"
        )
    pairs = pair_psi_files(data_path, n_frames=n_frames, frame_ids=frame_ids)
    read_options = {
        "density_unit_scale_cm3": float(density_unit_scale_cm3),
        "temperature_unit_scale_K": float(temperature_unit_scale_K),
        "reference_frame_id": int(reference_frame_id),
        "min_radius": min_radius,
        "max_radius": max_radius,
    }
    read_args = [(rho, temperature, read_options) for rho, temperature in pairs]
    frame_iterator = iter(_iter_psi_frames(read_args, workers))
    first_frame = next(frame_iterator)
    reference_axes = tuple(
        np.asarray(first_frame[name], dtype=np.float32)
        for name in ("radius", "latitude", "longitude")
    )
    field_shape = first_frame["rho"].shape
    fields = np.empty((len(pairs), *field_shape, 2), dtype=np.float32)
    raw_times = np.empty(len(pairs), dtype=np.float64)

    def store_frame(index, frame):
        axes = tuple(frame[name] for name in ("radius", "latitude", "longitude"))
        for name, reference, candidate in zip(
            ("radius", "latitude", "longitude"), reference_axes, axes
        ):
            if reference.shape != candidate.shape or not np.allclose(
                reference, candidate, equal_nan=False
            ):
                raise ValueError(f"PSI {name} grid changes between frames")
        if frame["rho"].shape != field_shape or frame["T"].shape != field_shape:
            raise ValueError("PSI field shape changes between frames")
        fields[index, ..., 0] = frame["rho"]
        fields[index, ..., 1] = frame["T"]
        raw_times[index] = frame["time"]

    store_frame(0, first_frame)
    for frame_index, frame in enumerate(frame_iterator, start=1):
        store_frame(frame_index, frame)

    radius, fields = _sort_axis(reference_axes[0] / Rs_per_ds, fields, 1, "radius")
    latitude, fields = _sort_axis(reference_axes[1], fields, 2, "latitude")
    longitude, fields = _normalize_longitude(reference_axes[2], fields, 3)

    normalized_times = raw_times * time_step_seconds / seconds_per_dt
    time_order = np.argsort(normalized_times)
    normalized_times = normalized_times[time_order]
    fields = fields[time_order]
    if normalized_times.size > 1 and np.any(np.diff(normalized_times) <= 0):
        raise ValueError("PSI frame times must be unique")

    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(fields[..., 0], out=fields[..., 0])
        np.log10(fields[..., 1], out=fields[..., 1])

    model = SphericalGridPlasmaModel(
        log_density=fields[..., 0],
        log_temperature=fields[..., 1],
        time=normalized_times.astype(np.float32),
        radius=radius,
        latitude=latitude,
        longitude=longitude,
        log_T=log_T,
    )
    observation_times = tuple(
        ref_date + timedelta(seconds=float(time * seconds_per_dt))
        for time in normalized_times
    )
    source_frames = []
    for density_path, temperature_path in pairs:
        density_path = Path(density_path).resolve()
        temperature_path = Path(temperature_path).resolve()
        source_frames.append({
            "frame_id": _frame_id(density_path),
            "density": {
                "path": str(density_path),
                "sha256": _sha256(density_path),
            },
            "temperature": {
                "path": str(temperature_path),
                "sha256": _sha256(temperature_path),
            },
        })
    source_provenance = {
        "schema": "sunerf.psi_grid_source.v1",
        "source_root": str(Path(data_path).resolve()),
        "frames": source_frames,
        "source_unit_conversion": {
            "density_unit_scale_cm3": float(density_unit_scale_cm3),
            "temperature_unit_scale_K": float(temperature_unit_scale_K),
            "density_output_unit": "cm-3",
            "temperature_output_unit": "K",
        },
        "temporal_reference": {
            "reference_frame_id": int(reference_frame_id),
            "reference_date": ref_date.isoformat(),
            "time_step_seconds": float(time_step_seconds),
        },
        "coordinate_convention": {
            "radius_unit": "solar-radius",
            "latitude_unit": "rad",
            "longitude_unit": "rad",
            "longitude_frame": longitude_frame,
        },
    }
    return PSIGrid(
        model=model,
        normalized_times=normalized_times.astype(np.float32),
        observation_times=observation_times,
        ref_date=ref_date,
        source_provenance=source_provenance,
    )
