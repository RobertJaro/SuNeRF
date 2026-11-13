#!/usr/bin/env python3
"""Synthetic in-situ time series of a virtual spacecraft inside the PSI/MAS cubes.

The probe follows the real trajectory of a spacecraft (JPL Horizons, default
Parker Solar Probe) around one of its perihelia, shifted in time so that the
perihelion falls at ``--perihelion-time`` of the simulation and rotated about
the solar rotation axis so that the probe sits at the Carrington longitude of
the CME at ``--align-time``.  A Sun-centred Keplerian orbit keeps its shape and
speed under both operations.  ``--longitude-offset-deg`` moves the probe away
from the CME direction, e.g. for a held-out validation probe.

Every simulation dump yields one sample at its tabulated observation time
(``--dump-times``); ``--pre-duration`` prepends samples before the first dump
at ``--pre-cadence``, taken from the first cube in rigid corotation exactly as
``render_psi_thomson`` does for the images.  Samples outside the radial extent
of the cubes are dropped.

The electron density is sampled from the ``rho`` cubes.  With ``--velocity-dir``
the MAS velocity cubes ``vr``, ``vt``, ``vp`` give the RTN velocity; the cubes
live in the corotating frame, so the tangential component receives the
corotation term ``Omega r sin(theta)``.  Without velocity cubes the velocity
targets are missing (``has_velocity = 0``).

Two files are written in the format of the prepared PSP/Solar Orbiter products
(``sunerf.data.prep.psp_insitu``): ``--out-file`` with lognormal density
scatter, Gaussian velocity errors and velocity gaps, and ``<out-file
stem>_truth.npz`` with the clean samples.  ``--plot-dir`` adds a QA figure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.time import Time
from sunpy.coordinates import frames
from sunerf.data.horizons import get_horizons_coord

from sunerf.data.coronagraph.prep_common import parse_duration, parse_time, positive_float
from sunerf.data.prep.base_insitu import BaseInSituPreparer, inertial_coords_from_hci
from sunerf.data.psi.density_cube import (
    MAS_RHO_TO_ELECTRON_CM3,
    MAS_VELOCITY_TO_KMS,
    PSIDensityCube,
    PSIFieldCube,
    dump_index,
    read_psi_density,
)
from sunerf.data.psi.dump_times import load_dump_times
from sunerf.data.psi.render_psi_thomson import (
    carrington_to_hci_longitude,
    discover_cubes,
    frame_schedule,
)

VELOCITY_COMPONENTS = ("vr", "vt", "vp")
# Sidereal Carrington rotation rate of the MAS frame.
CARRINGTON_OMEGA_RAD_S = 2.0 * np.pi / (25.38 * 86400.0)
KM_PER_RSUN = float((1.0 * u.R_sun).to_value(u.km))
DENSITY_SOURCE_SYNTHETIC = 4.0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--density-dir", type=Path, required=True,
                        help="Directory with the PSI rho<dump>.hdf sequence.")
    parser.add_argument("--dump-times", type=Path, required=True,
                        help="JSON table of the dump observation times from sunerf.data.psi.dump_times.")
    parser.add_argument("--velocity-dir", type=Path, default=None,
                        help="Directory with the MAS vr<dump>.hdf, vt<dump>.hdf and vp<dump>.hdf cubes.")
    parser.add_argument("--out-file", type=Path, required=True, help="Output .npz (prepared in-situ format).")
    parser.add_argument("--key", default="PSP_synth", help="Probe name written to the metadata.")
    parser.add_argument("--horizons-target", default="Parker Solar Probe",
                        help="JPL Horizons target whose trajectory the probe follows.")
    parser.add_argument("--reference-perihelion", type=parse_time, default=parse_time("2025-09-15T20:18:00"),
                        help="Perihelion time of the reference trajectory (PSP encounter 25 by default).")
    parser.add_argument("--perihelion-time", type=parse_time, default=parse_time("2021-10-27T20:00:00"),
                        help="Simulation time at which the probe passes perihelion.")
    parser.add_argument("--cme-longitude-deg", type=float, default=245.0,
                        help="Carrington longitude of the CME direction.")
    parser.add_argument("--align-time", type=parse_time, default=parse_time("2021-10-28T20:00:00"),
                        help="Simulation time at which the probe sits at the CME longitude.")
    parser.add_argument("--longitude-offset-deg", type=float, default=0.0,
                        help="Additional rotation of the trajectory about the solar rotation axis.")
    parser.add_argument("--pre-duration", type=parse_duration, default=parse_duration("20d"),
                        help="Time span before the first dump sampled from the first cube.")
    parser.add_argument("--pre-cadence", type=parse_duration, default=parse_duration("1h"),
                        help="Sample cadence inside --pre-duration.")
    parser.add_argument("--density-noise", type=float, default=0.15,
                        help="Standard deviation of the lognormal density scatter.")
    parser.add_argument("--velocity-noise-kms", type=float, default=25.0,
                        help="Standard deviation of the Gaussian error of every velocity component.")
    parser.add_argument("--gap-fraction", type=float, default=0.2,
                        help="Fraction of --gap-duration windows without velocity.")
    parser.add_argument("--gap-duration", type=parse_duration, default=parse_duration("30m"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--density-unit-scale-cm3", type=positive_float, default=MAS_RHO_TO_ELECTRON_CM3,
                        help="Electron density in cm^-3 of one cube density unit.")
    parser.add_argument("--velocity-unit-scale-kms", type=positive_float, default=MAS_VELOCITY_TO_KMS,
                        help="Velocity in km/s of one cube velocity unit.")
    parser.add_argument("--plot-dir", type=Path, default=None, help="Directory of the QA figure.")
    parser.add_argument("--device", default="cpu", help="Torch device of the cube interpolation.")
    return parser.parse_args(argv)


##########################################################
# trajectory

def horizons_hci_positions(target, times, chunk_size=1000):
    """HCI Cartesian positions [R_sun] of ``target`` at ``times`` (list of datetime)."""
    positions = []
    for start in range(0, len(times), chunk_size):
        obstime = Time(times[start:start + chunk_size])
        coord = get_horizons_coord(target, obstime).transform_to(frames.HeliocentricInertial(obstime=obstime))
        positions.append(np.stack([
            coord.cartesian.x.to_value(u.R_sun),
            coord.cartesian.y.to_value(u.R_sun),
            coord.cartesian.z.to_value(u.R_sun),
        ], axis=-1))
    return np.concatenate(positions, axis=0)


def rotate_about_z(position, angle_rad):
    """Rotate HCI positions ``[N, 3]`` by ``angle_rad`` about the solar rotation axis."""
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = position[:, 0], position[:, 1], position[:, 2]
    return np.stack([cos * x - sin * y, sin * x + cos * y, z], axis=-1)


def shifted_trajectory(times, args):
    """HCI positions [R_sun] of the probe at the simulation ``times``.

    The reference trajectory is read at ``time - perihelion_time +
    reference_perihelion`` and rotated so that the probe sits at the CME
    Carrington longitude (plus the offset) at ``align_time``.
    """
    shift = args.reference_perihelion - args.perihelion_time
    reference_times = [time + shift for time in times] + [args.align_time + shift]
    positions = horizons_hci_positions(args.horizons_target, reference_times)
    positions, align_position = positions[:-1], positions[-1]
    align_longitude = np.arctan2(align_position[1], align_position[0])
    target_longitude = np.deg2rad(args.cme_longitude_deg) + carrington_to_hci_longitude(args.align_time)
    rotation = target_longitude - align_longitude + np.deg2rad(args.longitude_offset_deg)
    return rotate_about_z(positions, rotation), float(rotation)


##########################################################
# sampling

def carrington_angles(position, time):
    """Colatitude and Carrington longitude [rad] of HCI positions at ``time``."""
    radius = np.linalg.norm(position, axis=-1)
    theta = np.arccos(np.clip(position[:, 2] / radius, -1.0, 1.0))
    phi = np.arctan2(position[:, 1], position[:, 0]) - carrington_to_hci_longitude(time)
    return radius, theta, phi % (2.0 * np.pi)


def load_velocity_cubes(velocity_dir, dump, unit_scale, device):
    cubes = {}
    for component in VELOCITY_COMPONENTS:
        matches = [path for path in sorted(velocity_dir.glob(f"{component}*.hdf")) if dump_index(path) == dump]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {component} cube of dump {dump} in {velocity_dir}.")
        cubes[component] = PSIFieldCube(*read_psi_density(matches[0]), unit_scale=unit_scale).to(device)
    return cubes


@torch.no_grad()
def sample_cubes(density_cube, velocity_cubes, radius, theta, phi, device):
    """Electron density [cm^-3] and RTN velocity [km/s] at the mesh coordinates."""
    as_tensor = lambda array: torch.tensor(np.asarray(array), dtype=torch.float32, device=device)
    r, t, p = as_tensor(radius), as_tensor(theta), as_tensor(phi)
    density = density_cube(r, t, p).cpu().numpy().astype(np.float64)
    if velocity_cubes is None:
        return density, np.full((radius.size, 3), np.nan)
    vr, vt, vp = (velocity_cubes[c](r, t, p).cpu().numpy().astype(np.float64) for c in VELOCITY_COMPONENTS)
    # Corotating MAS frame -> inertial: the rotation adds an azimuthal (T) component.
    corotation = CARRINGTON_OMEGA_RAD_S * radius * KM_PER_RSUN * np.sin(theta)
    # RTN: T along +phi, N along -theta (towards the north pole).
    return density, np.stack([vr, vp + corotation, -vt], axis=-1)


def sample_schedule(schedule, position, args, device):
    """Sample the cubes along the schedule; returns the rows inside the cubes."""
    cubes = discover_cubes(args.density_dir)
    dumps = np.array([dump for _, dump in schedule])
    rows = {"time": [], "dump": [], "position": [], "density": [], "velocity": [], "theta": [], "phi": []}
    for dump in sorted(set(dumps)):
        density_cube = PSIDensityCube(*read_psi_density(cubes[dump]),
                                      density_unit_scale_cm3=args.density_unit_scale_cm3).to(device)
        velocity_cubes = None
        if args.velocity_dir is not None:
            velocity_cubes = load_velocity_cubes(args.velocity_dir, dump, args.velocity_unit_scale_kms, device)
        for index in np.flatnonzero(dumps == dump):
            time = schedule[index][0]
            radius, theta, phi = carrington_angles(position[index:index + 1], time)
            r_min, r_max = density_cube.radial_range
            if not r_min <= radius[0] <= r_max:
                continue
            density, velocity = sample_cubes(density_cube, velocity_cubes, radius, theta, phi, device)
            rows["time"].append(time)
            rows["dump"].append(dump)
            rows["position"].append(position[index])
            rows["density"].append(density[0])
            rows["velocity"].append(velocity[0])
            rows["theta"].append(theta[0])
            rows["phi"].append(phi[0])
    if not rows["time"]:
        raise ValueError("The probe never enters the radial extent of the cubes; check --perihelion-time.")
    return {key: np.asarray(value) for key, value in rows.items()}


##########################################################
# products

def clean_arrays(samples, key):
    time_unix = np.array([Time(t).unix for t in samples["time"]], dtype=np.float64)
    position = samples["position"].astype(np.float64)
    radius = np.linalg.norm(position, axis=-1)
    velocity = samples["velocity"].astype(np.float32)
    has_velocity = np.all(np.isfinite(velocity), axis=-1).astype(np.float32)
    n = time_unix.size
    return {
        "time_unix": time_unix,
        "time_iso": np.array([Time(t).isot[:19] for t in samples["time"]]),
        "density_cm3": samples["density"].astype(np.float32),
        "density_source": np.full(n, DENSITY_SOURCE_SYNTHETIC, dtype=np.float32),
        "density_time_delta_seconds": np.zeros(n, dtype=np.float32),
        "has_density": np.ones(n, dtype=np.float32),
        "has_velocity": has_velocity,
        "velocity_rtn_kms": velocity,
        "velocity_radial_kms": velocity[:, 0].copy(),
        "velocity_time_delta_seconds": np.zeros(n, dtype=np.float32),
        "inertial_coords_r_lat_lon": inertial_coords_from_hci(position).astype(np.float32),
        "radius_rsun": radius.astype(np.float32),
        "r_hat": (position / radius[:, None]).astype(np.float32),
        "position_time_delta_seconds": np.zeros(n, dtype=np.float32),
        "quality": np.ones(n, dtype=np.float32),
        "position_source": np.array("jpl_horizons shifted and rotated (sunerf.data.psi.synthetic_insitu)"),
        "position_coordinate_system": np.array("SunPy HeliocentricInertial spherical r_lat_lon"),
        # Simulation bookkeeping of the synthetic product.
        "dump": samples["dump"].astype(np.int32),
        "carrington_colatitude_rad": samples["theta"].astype(np.float32),
        "carrington_longitude_rad": samples["phi"].astype(np.float32),
        "probe_key": np.array(key),
    }


def gap_mask(time_unix, gap_fraction, gap_seconds, rng):
    """Rows inside randomly dropped windows of ``gap_seconds``."""
    if gap_fraction <= 0.0:
        return np.zeros(time_unix.shape, dtype=bool)
    window = np.floor((time_unix - time_unix.min()) / gap_seconds).astype(np.int64)
    dropped = rng.random(window.max() + 1) < gap_fraction
    return dropped[window]


def degrade_arrays(arrays, args, rng):
    """Apply density scatter, velocity errors and velocity gaps; returns a new dict."""
    noisy = {key: np.array(value, copy=True) for key, value in arrays.items()}
    n = noisy["time_unix"].size
    noisy["density_cm3"] = (noisy["density_cm3"] * np.exp(rng.normal(0.0, args.density_noise, n))).astype(np.float32)
    velocity = noisy["velocity_rtn_kms"] + rng.normal(0.0, args.velocity_noise_kms, (n, 3))
    gaps = gap_mask(noisy["time_unix"], args.gap_fraction, args.gap_duration.total_seconds(), rng)
    velocity[gaps] = np.nan
    noisy["velocity_rtn_kms"] = velocity.astype(np.float32)
    noisy["velocity_radial_kms"] = velocity[:, 0].astype(np.float32)
    noisy["has_velocity"] = np.all(np.isfinite(velocity), axis=-1).astype(np.float32)
    return noisy


def write_product(path: Path, arrays, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    BaseInSituPreparer.write_summary(str(path), meta)


def qa_plot(truth, noisy, args, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = np.array([datetime.fromtimestamp(t, timezone.utc).replace(tzinfo=None) for t in truth["time_unix"]])
    cme = args.cme_longitude_deg
    lon = np.rad2deg(truth["carrington_longitude_rad"])
    lat = 90.0 - np.rad2deg(truth["carrington_colatitude_rad"])
    fig, axs = plt.subplots(4, 1, figsize=(8, 11), sharex=True, dpi=150)
    axs[0].plot(times, truth["radius_rsun"], color="black")
    axs[0].set_ylabel("radius [R_sun]")
    axs[1].plot(times, (lon - cme + 180.0) % 360.0 - 180.0, color="C0", label="longitude - CME")
    axs[1].plot(times, lat, color="C1", label="latitude")
    axs[1].axhline(0.0, color="0.5", lw=0.5)
    axs[1].set_ylabel("Carrington [deg]")
    axs[1].legend()
    axs[2].plot(times, noisy["density_cm3"], ".", color="0.6", ms=3, label="synthetic")
    axs[2].plot(times, truth["density_cm3"], color="black", lw=1, label="truth")
    axs[2].set_yscale("log")
    axs[2].set_ylabel("n_e [cm^-3]")
    axs[2].legend()
    axs[3].plot(times, noisy["velocity_radial_kms"], ".", color="0.6", ms=3)
    axs[3].plot(times, truth["velocity_radial_kms"], color="black", lw=1)
    axs[3].set_ylabel("v_r [km/s]")
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for ax in axs:
        ax.axvspan(args.align_time, args.align_time, color="C3")
    fig.autofmt_xdate()
    fig.suptitle(f"{args.key}: {truth['time_unix'].size} samples inside the cubes")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    cubes = discover_cubes(args.density_dir)
    schedule = frame_schedule(cubes, load_dump_times(args.dump_times), args.pre_duration, args.pre_cadence)
    times = [time for time, _ in schedule]
    position, rotation = shifted_trajectory(times, args)
    samples = sample_schedule(schedule, position, args, device)
    truth = clean_arrays(samples, args.key)
    noisy = degrade_arrays(truth, args, np.random.default_rng(args.seed))

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "probe_key": args.key,
        "product": "synthetic in-situ samples of a virtual spacecraft inside PSI/MAS cubes",
        "density_dir": str(args.density_dir),
        "velocity_dir": None if args.velocity_dir is None else str(args.velocity_dir),
        "horizons_target": args.horizons_target,
        "reference_perihelion": args.reference_perihelion.isoformat(),
        "perihelion_time": args.perihelion_time.isoformat(),
        "align_time": args.align_time.isoformat(),
        "cme_longitude_deg": args.cme_longitude_deg,
        "longitude_offset_deg": args.longitude_offset_deg,
        "trajectory_rotation_deg": float(np.rad2deg(rotation)),
        "pre_duration_seconds": args.pre_duration.total_seconds(),
        "pre_cadence_seconds": args.pre_cadence.total_seconds(),
        "density_noise_lognormal_sigma": args.density_noise,
        "velocity_noise_kms": args.velocity_noise_kms,
        "gap_fraction": args.gap_fraction,
        "gap_duration_seconds": args.gap_duration.total_seconds(),
        "seed": args.seed,
        "n_valid": int(truth["time_unix"].size),
        "n_velocity": int(noisy["has_velocity"].sum()),
        "radius_range_rsun": [float(truth["radius_rsun"].min()), float(truth["radius_rsun"].max())],
        "start": truth["time_iso"][0].item(),
        "end": truth["time_iso"][-1].item(),
        "density_source_encoding": {"psi_synthetic": DENSITY_SOURCE_SYNTHETIC},
        "output_units": BaseInSituPreparer.output_units({"carrington_longitude_rad": "rad"}),
    }
    truth_path = args.out_file.with_name(args.out_file.stem + "_truth.npz")
    write_product(truth_path, truth, {**meta, "noise": "none"})
    write_product(args.out_file, noisy, meta)
    if args.plot_dir is not None:
        qa_plot(truth, noisy, args, args.plot_dir / f"{args.out_file.stem}_qa.png")
    print(json.dumps({k: meta[k] for k in ("n_valid", "n_velocity", "radius_range_rsun", "start", "end",
                                           "trajectory_rotation_deg")}, indent=2))
    print(f"Wrote {args.out_file} and {truth_path}")


if __name__ == "__main__":
    main()
