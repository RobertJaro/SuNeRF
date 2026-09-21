"""Build a directly renderable SuNeRF state from PSI plasma grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header

from sunerf.data.psi.spherical_grid import load_psi_grid
from sunerf.model.plasma import (
    PLASMA_GRID_ARTIFACT_FORMAT_VERSION,
    PLASMA_GRID_ARTIFACT_TYPE,
    to_safe_artifact_primitive,
)
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.plasma import PlasmaRadiativeTransfer, init_absorption_model
from sunerf.resources import normalize_artifact_reference, resolve_artifact_path
from sunerf.response import RESPONSE_SCHEMA, load_response_artifact
from sunerf.train.runtime import atomic_torch_save


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_response_for_synthesis(
    response_file,
    requested_channels,
):
    """Load response metadata for direct synthetic rendering."""
    response_file = Path(response_file)
    if not response_file.is_file():
        raise FileNotFoundError(f"Temperature-response file does not exist: {response_file}")
    artifact = load_response_artifact(response_file)
    source_response_id = artifact.response_id
    selected_artifact = artifact.select_channels(requested_channels)
    if selected_artifact is not artifact:
        provenance = dict(selected_artifact.provenance)
        provenance["source_response_id"] = source_response_id
        selected_artifact = selected_artifact.updated(provenance=provenance)
    artifact = selected_artifact
    return artifact


def _response_renderer_config(
    artifact,
    response_path,
    *,
    Rs_per_ds,
    hydrogen_to_electron_ratio=None,
):
    config = {
        "artifact": str(response_path),
        "channels": artifact.channels,
        "learnable": False,
        "Rs_per_ds": float(Rs_per_ds),
    }
    if artifact.emission_measure_convention == "ne_nh":
        artifact_ratio = artifact.provenance.get("hydrogen_to_electron_ratio")
        if hydrogen_to_electron_ratio is None:
            hydrogen_to_electron_ratio = artifact_ratio
        if hydrogen_to_electron_ratio is None:
            raise ValueError(
                "an ne_nh response requires --hydrogen-to-electron-ratio; "
                "no composition-dependent ratio is assumed implicitly"
            )
        hydrogen_to_electron_ratio = float(hydrogen_to_electron_ratio)
        if not 0 < hydrogen_to_electron_ratio <= 1:
            raise ValueError("hydrogen_to_electron_ratio must be in (0, 1]")
        if artifact_ratio is not None and not np.isclose(
            hydrogen_to_electron_ratio, float(artifact_ratio), rtol=1e-7, atol=0.0
        ):
            raise ValueError(
                "hydrogen_to_electron_ratio does not match the response artifact provenance"
            )
        config["hydrogen_to_electron_ratio"] = hydrogen_to_electron_ratio
    return config


def _response_state_metadata(artifact, response_path):
    # Round-trip through canonical JSON so the saved state cannot retain a
    # caller-owned mutable provenance mapping.
    provenance = json.loads(json.dumps(dict(artifact.provenance), sort_keys=True))
    return {
        "source_format": "response_artifact_v1",
        "schema": RESPONSE_SCHEMA,
        "schema_version": artifact.schema_version,
        "path": str(response_path),
        "sha256": _sha256(response_path),
        "channels": artifact.channels,
        "log_temperature": artifact.log_temperature.tolist(),
        "log_density": (
            None if artifact.log_density is None else artifact.log_density.tolist()
        ),
        "response_unit": artifact.response_unit,
        "emission_measure_convention": artifact.emission_measure_convention,
        "provenance": provenance,
        "response_id": artifact.provenance.get("source_response_id", artifact.response_id),
    }


def _channel_wavelength_angstrom(channel):
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", str(channel).strip())
    if match is None:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def _default_channel_cmap(wavelength):
    aliases = {174: 171, 195: 193, 284: 211}
    wavelength = aliases.get(wavelength, wavelength)
    if wavelength in {94, 131, 171, 193, 211, 304, 335}:
        return f"sdoaia{wavelength}"
    return "gray"


def _synthetic_instrument_metadata(artifact, response_path, instrument_key):
    """Build the same ordered channel contract used by saved plasma models."""
    response_metadata = _response_state_metadata(artifact, response_path)
    response_id = response_metadata["response_id"]
    measurement_unit = (u.Unit(artifact.response_unit) * u.cm**-5).to_string()
    channels = []
    for response_channel_id in artifact.channels:
        wavelength = _channel_wavelength_angstrom(response_channel_id)
        channels.append({
            "id": str(response_channel_id),
            "response_channel_id": str(response_channel_id),
            "wavelength_angstrom": wavelength,
            "cmap": _default_channel_cmap(wavelength),
            "measurement_unit": measurement_unit,
            "response_id": response_id,
        })
    image_scaling = {
        "schema": "sunerf.image_scaling.v1",
        "operation": "divide",
        "channel_ids": [channel["id"] for channel in channels],
        "divisor": [1.0] * len(channels),
        "inverse_operation": "multiply",
    }
    instrument_metadata = {
        str(instrument_key): {
            "type": "plasma",
            "channels": channels,
            "response": response_metadata,
            "image_scaling": image_scaling,
            "cmap_default": "gray",
        }
    }
    data_metadata = {
        "channel_ids": tuple(channel["id"] for channel in channels),
        "cmaps": tuple(channel["cmap"] for channel in channels),
        "measurement_units": tuple(channel["measurement_unit"] for channel in channels),
        "response_ids": tuple(channel["response_id"] for channel in channels),
        "image_scaling": image_scaling,
    }
    return instrument_metadata, data_metadata


def _cpu_state_dict(module):
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in module.state_dict().items()
    }


def _grid_construction_spec(model):
    return {
        "type": "spherical_grid_plasma",
        "longitude_period": float(model.longitude_period),
        "fill_log_density": float(model.fill_log_density),
        "fill_log_temperature": float(model.fill_log_temperature),
        "clamp_time": bool(model.clamp_time),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create a SuNeRF-compatible synthetic-observation state by "
            "interpolating the PSI density/temperature grid directly."
        )
    )
    parser.add_argument("--temperature-response-artifact", dest="temperature_response_file", required=True)
    parser.add_argument("--response-channels", dest="response_channels", nargs="+", default=None)
    parser.add_argument(
        "--hydrogen-to-electron-ratio",
        dest="hydrogen_to_electron_ratio",
        type=float,
        default=None,
        help=(
            "Required for responses defined per n_e*n_H; this composition-dependent "
            "factor converts n_e^2 to n_e*n_H."
        ),
    )
    parser.add_argument("--data-path", dest="data_path", required=True)
    parser.add_argument("--out-path", dest="out_path", required=True)
    parser.set_defaults(Rs_per_ds=1.0)
    parser.add_argument("--seconds-per-dt", dest="seconds_per_dt", type=float, default=86400.0)
    parser.add_argument("--time-step-seconds", dest="time_step_seconds", type=float, default=3600.0)
    parser.add_argument(
        "--source-density-scale-cm3",
        dest="density_unit_scale_cm3",
        type=float,
        required=True,
        help="Number of cm^-3 represented by one source density unit.",
    )
    parser.add_argument(
        "--source-temperature-scale-k",
        dest="temperature_unit_scale_K",
        type=float,
        required=True,
        help="Number of kelvin represented by one source temperature unit.",
    )
    parser.add_argument(
        "--reference-frame-id",
        type=int,
        required=True,
        help="PSI frame whose timestamp is --reference-date.",
    )
    parser.add_argument(
        "--longitude-frame",
        choices=("carrington",),
        required=True,
        help="Physical coordinate frame represented by the source longitude axis.",
    )
    parser.add_argument("--reference-date", dest="reference_date", default="2025-01-01T00:00:00")
    parser.add_argument("--n-frames", dest="n_frames", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-radius", dest="min_radius", type=float, default=1.0)
    parser.add_argument("--max-radius", dest="max_radius", type=float, default=2.6)
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=128)
    parser.add_argument("--n-hierarchical-samples", dest="n_hierarchical_samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("NY", "NX"), default=(512, 512))
    parser.add_argument("--fov-arcsec", dest="fov_arcsec", type=float, nargs=2, metavar=("Y", "X"), default=(2400, 2400))
    parser.add_argument("--instrument-key", dest="instrument_key", default="PSI")
    parser.add_argument(
        "--temperature-cutoff",
        dest="temperature_cutoff",
        type=float,
        nargs=2,
        metavar=("T_CUT_K", "DELTA_T_K"),
        default=None,
        help="Transition-region emission cutoff, e.g. 4e5 5e4 for thermodynamic MAS cubes.",
    )
    parser.add_argument(
        "--absorption-artifact",
        dest="absorption_artifact",
        default=None,
        help="H/He photoionization bundle (path or builtin:<name>); the bundle "
             "rows are selected with --absorption-instrument-key.",
    )
    parser.add_argument(
        "--absorption-instrument-key", dest="absorption_instrument_key", default=None,
    )
    return parser


def build_grid_absorption_model(absorption_config, channels):
    """Construct the deterministic opacity recorded in a grid artifact."""
    if absorption_config is None:
        return None
    config = dict(absorption_config)
    instrument_key = config.pop("instrument_key")
    resolve_artifact_path(config["artifact"])
    return init_absorption_model(config, instrument_key=instrument_key, channels=list(channels))


def build_synthetic_state(args):
    if not np.isclose(float(args.Rs_per_ds), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            'Direct PSI plasma artifacts require Rs_per_ds=1 so model distance '
            'units remain solar radii throughout rendering and evaluation.'
        )
    if args.density_unit_scale_cm3 <= 0:
        raise ValueError("density_unit_scale_cm3 must be positive")
    if args.temperature_unit_scale_K <= 0:
        raise ValueError("temperature_unit_scale_K must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.n_samples < 2 or args.n_hierarchical_samples < 1:
        raise ValueError("at least two coarse and one hierarchical sample are required")
    if any(size <= 0 for size in args.resolution) or any(fov <= 0 for fov in args.fov_arcsec):
        raise ValueError("resolution and field of view must be positive")
    if not args.instrument_key:
        raise ValueError("instrument_key must not be empty")

    ref_date = datetime.fromisoformat(args.reference_date.replace("Z", "+00:00"))
    response_path = Path(args.temperature_response_file).resolve()
    response_artifact = _load_response_for_synthesis(
        response_path,
        args.response_channels,
    )
    response_channels = response_artifact.channels
    # Use the artifact's validated temperature grid directly. A hidden global
    # arange silently changed the response support and duplicated grid policy.
    log_T_range = np.asarray(response_artifact.log_temperature, dtype=np.float32)
    grid = load_psi_grid(
        args.data_path,
        log_T=log_T_range,
        density_unit_scale_cm3=args.density_unit_scale_cm3,
        temperature_unit_scale_K=args.temperature_unit_scale_K,
        reference_frame_id=args.reference_frame_id,
        longitude_frame=args.longitude_frame,
        Rs_per_ds=args.Rs_per_ds,
        seconds_per_dt=args.seconds_per_dt,
        time_step_seconds=args.time_step_seconds,
        n_frames=args.n_frames,
        workers=args.workers,
        ref_date=ref_date,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )

    response_config = _response_renderer_config(
        response_artifact,
        response_path,
        Rs_per_ds=args.Rs_per_ds,
        hydrogen_to_electron_ratio=getattr(args, "hydrogen_to_electron_ratio", None),
    )

    temperature_cutoff = getattr(args, "temperature_cutoff", None)
    if temperature_cutoff is not None:
        response_config["temperature_cutoff"] = {
            "T_cut_K": float(temperature_cutoff[0]),
            "delta_T_K": float(temperature_cutoff[1]),
        }
    absorption_config = None
    if getattr(args, "absorption_artifact", None) is not None:
        absorption_config = {
            "type": "photoionization",
            "artifact": normalize_artifact_reference(args.absorption_artifact),
            "instrument_key": str(
                getattr(args, "absorption_instrument_key", None) or args.instrument_key
            ),
            "hydrogen_density_convention": "fully_ionized_proxy",
        }
    radiative_transfer = PlasmaRadiativeTransfer(
        temperature_response_config=response_config,
        log_T_range=log_T_range,
        absorption_model=build_grid_absorption_model(absorption_config, response_channels),
    )
    sampling_config = {
        "type": "spherical",
        "min_distance": args.min_radius,
        "max_distance": args.max_radius,
        "n_samples": args.n_samples,
        "perturb": False,
    }
    hierarchical_sampling_config = {
        "type": "hierarchical",
        "n_samples": args.n_hierarchical_samples,
        "perturb": False,
    }
    rendering = BasicRenderingModule(
        model=grid.model,
        rendering_modules={args.instrument_key: radiative_transfer},
        Rs_per_ds=args.Rs_per_ds,
        seconds_per_dt=args.seconds_per_dt,
        sampling_config=sampling_config,
        hierarchical_sampling_config=hierarchical_sampling_config,
    ).eval()

    ny, nx = args.resolution
    fov_y, fov_x = args.fov_arcsec
    observer = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        1 * u.AU,
        frame=frames.HeliographicStonyhurst,
        obstime=grid.ref_date,
    )
    reference_coord = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=grid.ref_date,
        observer=observer,
        frame=frames.Helioprojective,
    )
    image_shape = (ny, nx)
    scale = [fov_x / nx, fov_y / ny] * u.arcsec / u.pix
    wcs = make_fitswcs_header(np.zeros(image_shape, dtype=np.float32), reference_coord, scale=scale)

    instrument_metadata, synthetic_data_metadata = _synthetic_instrument_metadata(
        response_artifact, response_path, args.instrument_key
    )

    data_config = {
        args.instrument_key: {
            "times": grid.observation_times,
            "image_shape": image_shape,
            "wcs": wcs,
            "instrument_key": args.instrument_key,
            "channels": response_channels,
            **synthetic_data_metadata,
        }
    }
    rendering_state = _cpu_state_dict(rendering)
    grid_state = _cpu_state_dict(grid.model)
    grid_prefix = "model."
    rendering_state_without_grid = {
        key: value
        for key, value in rendering_state.items()
        if not key.startswith(grid_prefix)
    }
    if {
        key.removeprefix(grid_prefix): value
        for key, value in rendering_state.items()
        if key.startswith(grid_prefix)
    }.keys() != grid_state.keys():
        raise RuntimeError("Synthetic renderer grid state does not match its source model.")

    response_metadata = _response_state_metadata(response_artifact, response_path)
    provenance = {
        "response_artifacts": {
            args.instrument_key: instrument_metadata[args.instrument_key]["response"]
        },
        "synthetic_source": "direct_psi_grid_interpolation",
        "psi_grid_source": grid.source_provenance,
    }
    state = {
        "artifact_type": PLASMA_GRID_ARTIFACT_TYPE,
        "artifact_format_version": PLASMA_GRID_ARTIFACT_FORMAT_VERSION,
        "artifact_security": "weights_only",
        "construction_spec": to_safe_artifact_primitive({
            "model": _grid_construction_spec(grid.model),
            "rendering": {
                "instrument_key": str(args.instrument_key),
                "Rs_per_ds": float(args.Rs_per_ds),
                "seconds_per_dt": float(args.seconds_per_dt),
                "sampling_config": sampling_config,
                "hierarchical_sampling_config": hierarchical_sampling_config,
                "temperature_response_config": response_config,
                "absorption_config": absorption_config,
            },
        }),
        # The large grid occurs exactly once. Renderer state stores only sampler
        # and radiative-transfer buffers/parameters, never an executable module.
        "grid_state_dict": grid_state,
        "rendering_state_dict": rendering_state_without_grid,
        "data_config": to_safe_artifact_primitive(data_config),
        "Rs_per_ds": float(args.Rs_per_ds),
        "seconds_per_dt": float(args.seconds_per_dt),
        "ref_date": to_safe_artifact_primitive(grid.ref_date),
        "temperature_grid": {
            "source": "response_artifact",
            "log10_K": log_T_range.astype(float).tolist(),
        },
        "log_T_range": torch.as_tensor(log_T_range, dtype=torch.float32),
        "instrument_metadata": to_safe_artifact_primitive(instrument_metadata),
        "temperature_response": to_safe_artifact_primitive(response_metadata),
        "synthetic_grid": to_safe_artifact_primitive({
            **grid.source_provenance,
            "normalized_times": grid.normalized_times,
            "interpolation": "joint-linear(time, radius, latitude, periodic-longitude)",
        }),
        "provenance": to_safe_artifact_primitive(provenance),
    }
    # Final recursive pass is intentional: state_dict extra state and metadata
    # supplied by third-party libraries can otherwise retain NumPy scalar
    # objects that PyTorch's restricted unpickler correctly rejects.
    return to_safe_artifact_primitive(state)


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "save_state.grid.pt"
    state = build_synthetic_state(args)
    atomic_torch_save(state, output_path)
    torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"Saved weights-only grid-backed synthetic SuNeRF state: {output_path}")


if __name__ == "__main__":
    main()
