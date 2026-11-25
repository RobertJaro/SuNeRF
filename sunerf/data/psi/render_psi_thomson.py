#!/usr/bin/env python3
"""Render clean synthetic tB/pB observations of one observer from PSI/MAS density cubes.

Every cube in ``--density-dir`` yields one tB/pB pair at the observation time
that ``--dump-times`` (written by ``sunerf.data.psi.dump_times``) tabulates for
its dump, so the series keeps the cadence of the simulation.  The observer
is Earth, a Sun--Earth Lagrange point or an explicit Stonyhurst position. The
grid is a Sun-centred square or the ARC grid and footprint of a PUNCH PAM
mosaic supplied with ``--punch-pam``. Rays are built by the
training loader; detector effects are applied afterwards with ``degrade_psi``.

``--pre-duration`` prepends frames before the first dump.  They render the
first cube, which is assumed to be the relaxed steady state.  The cubes live in
the corotating Carrington frame, and rigid corotation of a steady state
satisfies the continuity equation with ``v = v_corotating + Omega x r``, so the
series can be extended to arbitrarily early times.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.coordinates import frames, get_earth
from sunpy.map import Map, make_fitswcs_header
from tqdm.auto import tqdm

from sunerf.data.coronagraph.prep_common import crop_to_field_limit, parse_duration, positive_float
from sunerf.data.coronagraph.prep_punch_pam import PUNCH_OCC_MIN, PUNCH_INVALID_FILE_FLAGS
from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.data.psi.density_cube import (
    MAS_RHO_TO_ELECTRON_CM3,
    PSIDensityCube,
    dump_index,
    read_psi_density,
)
from sunerf.data.psi.dump_times import load_dump_times
from sunerf.physics.thomson import electron_density_normalization_cm3
from sunerf.rendering.thomson import ThomsonScattering

PRODUCTS = ("tb", "pb")
# Ecliptic longitude relative to Earth in degrees and distance relative to Earth's.
LAGRANGE_POINTS = {
    "L1": (0.0, 0.99),
    "L2": (0.0, 1.01),
    "L3": (180.0, 1.0),
    "L4": (60.0, 1.0),
    "L5": (-60.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--density-dir", type=Path, required=True,
                        help="Directory with the PSI rho<dump>.hdf sequence.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory; files go to <out-dir>/{tb,pb}/.")
    parser.add_argument("--key", required=True, help="Observer name used in the file names.")
    parser.add_argument("--observer", nargs="+", required=True, metavar="VALUE",
                        help="Earth, Lagrange point (L1..L5), or Stonyhurst 'LON_DEG LAT_DEG DISTANCE_AU'.")
    parser.add_argument("--punch-pam", type=Path,
                        help="PUNCH L3 PAM FITS supplying spatial WCS and uncertainty footprint.")
    parser.add_argument("--inner-rsun", type=positive_float,
                        help="Smallest impact parameter. PAM defaults to the preparation occulter angle.")
    parser.add_argument("--outer-rsun", type=positive_float, required=True,
                        help="Largest rendered impact parameter (at most the cube radius).")
    parser.add_argument("--resolution", type=int, required=True, help="Image size in pixels.")
    parser.add_argument("--dump-times", type=Path, required=True,
                        help="JSON table of the dump observation times from sunerf.data.psi.dump_times.")
    parser.add_argument("--pre-duration", type=parse_duration, default=None,
                        help="Time span before the first dump rendered from the first cube, e.g. 2d.")
    parser.add_argument("--pre-cadence", type=parse_duration, default=None,
                        help="Cadence of the frames inside --pre-duration, e.g. 15m.")
    parser.add_argument("--n-samples", type=int, default=512, help="Samples per line of sight.")
    parser.add_argument("--ray-batch", type=int, default=16384, help="Rays rendered at once.")
    parser.add_argument("--density-unit-scale-cm3", type=positive_float,
                        default=MAS_RHO_TO_ELECTRON_CM3,
                        help="Electron density in cm^-3 of one cube density unit.")
    parser.add_argument("--device", default=None, help="Torch device; default picks cuda/mps/cpu.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.inner_rsun is None and args.punch_pam is None:
        parser.error("--inner-rsun is required unless --punch-pam is supplied")
    return args


def discover_cubes(density_dir: Path) -> dict[int, Path]:
    cubes = {dump_index(path): path for path in sorted(density_dir.glob("*.hdf"))}
    if not cubes:
        raise FileNotFoundError(f"No PSI density cubes (*.hdf) found in {density_dir}.")
    return dict(sorted(cubes.items()))


def frame_schedule(dumps, dump_times, pre_duration=None, pre_cadence=None):
    """Return chronological ``(time, dump)`` pairs; early frames reuse the first dump.

    ``dump_times`` maps every dump to its simulation observation time.
    """
    if (pre_duration is None) != (pre_cadence is None):
        raise ValueError("--pre-duration and --pre-cadence must be given together.")
    dumps = sorted(dumps)
    missing = [dump for dump in dumps if dump not in dump_times]
    if missing:
        raise KeyError(f"--dump-times holds no observation time for dumps {missing}.")
    times = [dump_times[dump].to_datetime() for dump in dumps]
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("Dump observation times must increase with the dump number.")
    first, start_time = dumps[0], times[0]
    schedule = []
    if pre_duration is not None:
        count = int(pre_duration / pre_cadence)
        schedule += [(start_time - step * pre_cadence, first) for step in range(count, 0, -1)]
    schedule += list(zip(times, dumps))
    return schedule


def observer_coordinate(observer, time) -> SkyCoord:
    """Stonyhurst position of Earth, a Lagrange point or 'lon_deg lat_deg distance_au'."""
    if len(observer) == 1 and observer[0].lower() == "earth":
        return get_earth(time)
    stonyhurst = frames.HeliographicStonyhurst(obstime=time)
    if len(observer) == 1 and observer[0].upper() in LAGRANGE_POINTS:
        longitude, distance_factor = LAGRANGE_POINTS[observer[0].upper()]
        return SkyCoord(
            lon=longitude * u.deg, lat=0 * u.deg, distance=get_earth(time).radius * distance_factor,
            frame=frames.HeliocentricEarthEcliptic(obstime=time),
        ).transform_to(stonyhurst)
    if len(observer) != 3:
        raise ValueError(
            f"--observer takes Earth, one of {sorted(LAGRANGE_POINTS)} or 'LON_DEG LAT_DEG DISTANCE_AU'."
        )
    longitude, latitude, distance = (float(value) for value in observer)
    return SkyCoord(lon=longitude * u.deg, lat=latitude * u.deg, radius=distance * u.AU, frame=stonyhurst)


def observation_map(observer: SkyCoord, time, outer_rsun, resolution, key) -> Map:
    """Empty Sun-centred map whose edge touches the ``outer_rsun`` impact parameter."""
    outer_fraction = (outer_rsun * u.R_sun / observer.radius).to_value(u.dimensionless_unscaled)
    if not 0 < outer_fraction < 1:
        raise ValueError("--outer-rsun must lie between the Sun and the observer.")
    # Gnomonic pixels are spaced in tan(elongation).
    half_width = (np.tan(np.arcsin(outer_fraction)) * u.rad).to(u.arcsec)
    scale = 2.0 * half_width / resolution / u.pixel
    center = SkyCoord(0 * u.arcsec, 0 * u.arcsec,
                      frame=frames.Helioprojective(observer=observer, obstime=time))
    data = np.full((resolution, resolution), np.nan, dtype=np.float32)
    header = make_fitswcs_header(
        data, center, scale=u.Quantity([scale, scale]), telescope="PSI-MAS", instrument=key
    )
    return Map(data, header)


def load_pam_footprint(path: Path) -> Map:
    """Read the angular grid and joint coverage of a Level 3 PAM mosaic."""
    with fits.open(path) as hdus:
        header = hdus[1].header
        if (header.get("TYPECODE"), header.get("OBSCODE")) != ("PA", "M"):
            raise ValueError(f"{path} is not a PUNCH PAM mosaic.")
        if any(header.get(flag, 0) for flag in PUNCH_INVALID_FILE_FLAGS):
            raise ValueError(f"{path} has PUNCH file-quality flags set.")
        uncertainty = hdus[2].data[:2]
        covered = np.any(np.isfinite(uncertainty) & (uncertainty != 0), axis=0)
        wcs = WCS(header, naxis=2)
        if tuple(wcs.wcs.ctype) != ("HPLN-ARC", "HPLT-ARC"):
            raise ValueError(f"{path} lacks the helioprojective ARC mosaic grid.")
        return Map(covered.astype(np.float32), wcs.to_header())


def pam_observation_map(template, observer, time, outer_rsun, resolution, key):
    """Place the angular footprint at the simulation observer and crop to the requested field."""
    ref = template.reference_coordinate
    center = SkyCoord(ref.Tx, ref.Ty, frame=frames.Helioprojective(observer=observer, obstime=time))
    header = make_fitswcs_header(
        template.data, center, scale=u.Quantity(template.scale),
        reference_pixel=u.Quantity(template.reference_pixel), rotation_matrix=template.rotation_matrix,
        projection_code="ARC", telescope="PSI-MAS", instrument=key,
    )
    fraction = (outer_rsun * u.R_sun / observer.radius).to_value(u.one)
    if not 0 < fraction < 1:
        raise ValueError("--outer-rsun must lie between the Sun and the observer.")
    cropped = crop_to_field_limit(Map(template.data, header), np.arcsin(fraction) * u.rad)
    return cropped.resample([resolution, resolution] * u.pixel, method="nearest")


def carrington_to_hci_longitude(time) -> float:
    """HCI longitude in radians of the Carrington prime meridian at ``time``."""
    meridian = SkyCoord(
        lon=0 * u.deg, lat=0 * u.deg, radius=1 * u.R_sun,
        frame=frames.HeliographicCarrington, observer="self", obstime=time,
    )
    return float(meridian.transform_to(frames.HeliocentricInertial(obstime=time)).lon.to_value(u.rad))


def line_of_sight_samples(rays_o, rays_d, sphere_radius, n_samples):
    """Sample each chord through the simulation sphere uniformly in scattering angle.

    ``z = t_ca + b tan(alpha)`` concentrates samples around the point of closest
    approach, where the density and the Thomson kernel peak.
    """
    closest = -np.einsum("ij,ij->i", rays_o, rays_d)
    impact = np.linalg.norm(rays_o + closest[:, None] * rays_d, axis=-1)
    half_chord = np.sqrt(np.clip(sphere_radius ** 2 - impact ** 2, 0.0, None))
    upper = np.arctan2(half_chord, impact)
    lower = np.maximum(-upper, np.arctan2(-closest, impact))  # never behind the observer
    alpha = lower[:, None] + (upper - lower)[:, None] * np.linspace(0.0, 1.0, n_samples)
    return closest[:, None] + impact[:, None] * np.tan(alpha)


@torch.no_grad()
def render_rays(cube, thomson, rays, longitude_offset, n_samples, ray_batch, device):
    """Return tB/pB in mean solar brightness for ``rays[N, 2, 3]`` in HCI R_sun."""
    rays = np.asarray(rays, dtype=np.float64)
    sphere_radius = cube.radial_range[1]
    brightness_per_model_unit = 1.0 / electron_density_normalization_cm3(1.0, 1.0)
    images = []
    for start in range(0, rays.shape[0], ray_batch):
        batch = slice(start, start + ray_batch)
        rays_o = torch.tensor(rays[batch, 0], dtype=torch.float32, device=device)
        rays_d = torch.tensor(rays[batch, 1], dtype=torch.float32, device=device)
        z_vals = line_of_sight_samples(rays[batch, 0], rays[batch, 1], sphere_radius, n_samples)
        z = torch.tensor(z_vals, dtype=torch.float32, device=device)
        points = rays_o[:, None] + z[..., None] * rays_d[:, None]
        radius = points.norm(dim=-1)
        theta = torch.acos((points[..., 2] / radius.clamp_min(1e-6)).clamp(-1.0, 1.0))
        phi = torch.atan2(points[..., 1], points[..., 0]) - longitude_offset
        density = cube(radius, theta, phi)
        image = thomson(density[..., None], z, rays_d, rays_o, points)["image"]
        images.append((image * brightness_per_model_unit).cpu().numpy())
    return np.concatenate(images, axis=0)


def render_map(s_map, cube, thomson, inner_rsun, outer_rsun, n_samples, ray_batch, device,
               footprint=None):
    geometry = MapDataLoader(Rs_per_ds=1.0, reference_frame="inertial").load(s_map)
    projected_radius = geometry["projected_radius"]
    valid = (projected_radius >= inner_rsun) & (projected_radius <= outer_rsun)
    if footprint is not None:
        valid &= footprint
    if not np.any(valid):
        raise ValueError("No pixels are covered within the requested field of view.")
    image = np.full((*projected_radius.shape, 2), np.nan, dtype=np.float32)
    image[valid] = render_rays(
        cube, thomson, geometry["rays"][valid], carrington_to_hci_longitude(s_map.date),
        n_samples, ray_batch, device,
    )
    return {"tb": image[..., 0], "pb": image[..., 1]}


def save_product(data, s_map: Map, path: Path, metadata: dict) -> None:
    meta = s_map.meta.copy()
    meta["bunit"] = "MSB"
    meta["synth"] = "CLEAR"
    meta["occmin"] = float(metadata["inner_rsun"])
    meta["occmax"] = float(metadata["outer_rsun"])
    meta["occunit"] = "R_sun"
    meta["sim_modl"] = "PSI-MAS"
    meta["sim_dump"] = int(metadata["dump"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Map(np.asarray(data, dtype=np.float32), meta).save(path, overwrite=True)


def main() -> None:
    args = parse_args()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(args.device)
    pam = load_pam_footprint(args.punch_pam) if args.punch_pam else None
    cubes = discover_cubes(args.density_dir)
    schedule = frame_schedule(
        cubes, load_dump_times(args.dump_times), args.pre_duration, args.pre_cadence
    )

    thomson = ThomsonScattering(Rs_per_ds=1.0).to(device)
    cube, loaded_dump = None, None
    manifest = []
    for index, (time, dump) in enumerate(tqdm(schedule, desc=f"Rendering {args.key}"), start=1):
        outputs = {
            product: args.out_dir / product / f"{args.key}_{product}{index:04d}.fts"
            for product in PRODUCTS
        }
        manifest.append({
            "index": index, "time": Time(time).isot, "dump": dump,
            **{product: str(path) for product, path in outputs.items()},
        })
        if not args.overwrite and all(path.exists() for path in outputs.values()):
            continue
        if dump != loaded_dump:
            cube = PSIDensityCube(
                *read_psi_density(cubes[dump]),
                density_unit_scale_cm3=args.density_unit_scale_cm3,
            ).to(device)
            loaded_dump = dump
            if args.outer_rsun > cube.radial_range[1]:
                raise ValueError(
                    f"--outer-rsun {args.outer_rsun} exceeds the cube radius {cube.radial_range[1]:.2f}."
                )
        observer = observer_coordinate(args.observer, time)
        inner_rsun = args.inner_rsun
        footprint = None
        if pam is None:
            s_map = observation_map(observer, time, args.outer_rsun, args.resolution, args.key)
        else:
            s_map = pam_observation_map(pam, observer, time, args.outer_rsun, args.resolution, args.key)
            footprint = s_map.data == 1
            s_map.meta["pam_file"] = args.punch_pam.name
            if inner_rsun is None:
                inner_rsun = observer.radius.to_value(u.R_sun) * np.sin(PUNCH_OCC_MIN * u.arcsec).value
        s_map.meta["sim_file"] = cubes[dump].name
        s_map.meta["los_rmax"] = cube.radial_range[1]  # R_sun
        s_map.meta["ne_scale"] = args.density_unit_scale_cm3
        images = render_map(
            s_map, cube, thomson, inner_rsun, args.outer_rsun,
            args.n_samples, args.ray_batch, device, footprint=footprint,
        )
        metadata = {"inner_rsun": inner_rsun, "outer_rsun": args.outer_rsun, "dump": dump}
        for product, path in outputs.items():
            save_product(images[product], s_map, path, metadata)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Rendered clean PSI observations: {args.out_dir}")


if __name__ == "__main__":
    main()
