"""Forward-synthesize prepared EUV observations from PSI/MAS plasma cubes.

The PSI data path contains the density and temperature cubes in separate
``rho/`` and ``t/`` folders. Every snapshot pair is rendered as one frame: the
regular SuNeRF ray, sampling, and radiative-transfer code images the snapshot
for the requested observer, and the images are written in the
prepared-EUV-v2 FITS contract. ``sunerf-plasma`` therefore reconstructs the
synthetic observations through exactly the same path as real data, with the
simulation cubes as ground truth.

The HDF5 cubes carry no time information (no attributes, no time axis), so the
date of one reference snapshot and the snapshot cadence must be supplied.

One call renders one instrument at one observer location
(:func:`render_observer`); a multi-view data set is a sequence of such calls.
The location is a Sun-Earth Lagrange point (``L1`` ... ``L5``), a planet
(``mercury`` ... ``neptune``), or explicit Stonyhurst coordinates
``longitude_deg,latitude_deg,distance_au``.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from astropy import constants as const
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst, get_earth
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm

from sunerf.data.euv.observation import PREPARED_EUV_SCHEMA
from sunerf.data.euv.prepare import _atomic_write_fits
from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.data.psi.spherical_grid import _frame_id, load_psi_grid, pair_psi_files
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.plasma import PlasmaRadiativeTransfer, init_absorption_model
from sunerf.response import load_response_artifact

# MAS code units (Mikic et al.): density 1e8 cm^-3, temperature 2.807e7 K.
MAS_DENSITY_UNIT_CM3 = 1.0e8
MAS_TEMPERATURE_UNIT_K = 2.807e7
DEFAULT_CUTOFF = {"T_cut_K": 4.0e5, "delta_T_K": 5.0e4}
INSTRUMENTS = {
    "AIA": {
        "response": "builtin:aia",
        "channels": ("A94", "A131", "A171", "A193", "A211", "A335"),
        "header": {"telescop": "SDO/AIA", "instrume": "AIA_SYNTHETIC", "quality": 0},
        "cmap": "sdoaia{wavelength}",
    },
    "EUVI-A": {
        "response": "builtin:euvi_a",
        "channels": ("171", "195", "284"),
        "header": {"telescop": "STEREO", "instrume": "SECCHI", "detector": "EUVI",
                   "obsrvtry": "STEREO_A"},
        "cmap": "euvi{wavelength}",
    },
    "EUVI-B": {
        "response": "builtin:euvi_b",
        "channels": ("171", "195", "284"),
        "header": {"telescop": "STEREO", "instrume": "SECCHI", "detector": "EUVI",
                   "obsrvtry": "STEREO_B"},
        "cmap": "euvi{wavelength}",
    },
}
PLANETS = ("mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune")
LAGRANGE_POINTS = ("L1", "L2", "L3", "L4", "L5")
OBSERVER_KEYS = LAGRANGE_POINTS + PLANETS
# Sun-Earth L1/L2 lie at (mu / 3)^(1/3) ~ 1 % of the Sun-Earth distance.
_COLLINEAR_LAGRANGE_FRACTION = 0.01
_SIDEREAL_YEAR_DAYS = 365.256363


def _channel_wavelength(channel) -> int:
    return int("".join(character for character in str(channel) if character.isdigit()))


def _earth_orbit_point(time, phase_deg):
    """Point on Earth's orbit ``phase_deg`` ahead of Earth, seen at ``time``.

    The point is where Earth is ``phase_deg / 360`` of a year later. It is fixed
    in the inertial frame and expressed in Stonyhurst coordinates of ``time``.
    """
    later = time + timedelta(days=_SIDEREAL_YEAR_DAYS * phase_deg / 360.0)
    inertial = get_earth(later).transform_to(frames.HeliocentricInertial(obstime=later))
    point = SkyCoord(
        inertial.lon, inertial.lat, inertial.distance,
        frame=frames.HeliocentricInertial(obstime=time),
    )
    return point.transform_to(frames.HeliographicStonyhurst(obstime=time))


def observer_coordinate(location, time):
    """Stonyhurst coordinate of a named or explicit observer location."""
    text = str(location).strip()
    key = text.upper() if text.upper() in LAGRANGE_POINTS else text.lower()
    if key in PLANETS:
        return get_body_heliographic_stonyhurst(key, time)
    if key in ("L1", "L2"):
        earth = get_earth(time)
        sign = -1.0 if key == "L1" else 1.0
        return SkyCoord(
            earth.lon, earth.lat, earth.radius * (1.0 + sign * _COLLINEAR_LAGRANGE_FRACTION),
            frame=frames.HeliographicStonyhurst(obstime=time),
        )
    if key in ("L3", "L4", "L5"):
        return _earth_orbit_point(time, {"L3": 180.0, "L4": 60.0, "L5": -60.0}[key])
    try:
        longitude, latitude, distance = (float(value) for value in text.split(","))
    except ValueError as error:
        raise ValueError(
            f"observer location {location!r} must be one of {OBSERVER_KEYS} or "
            "'longitude_deg,latitude_deg,distance_au' in Stonyhurst coordinates"
        ) from error
    if not -90.0 <= latitude <= 90.0 or not np.isfinite([longitude, distance]).all() or distance <= 0:
        raise ValueError(f"invalid explicit observer location {location!r}")
    return SkyCoord(
        longitude * u.deg, latitude * u.deg, distance * u.AU,
        frame=frames.HeliographicStonyhurst(obstime=time),
    )


def frame_times(frame_ids, *, reference_date, cadence_seconds, reference_frame_id=None):
    """Observation time of every snapshot from the explicit reference and cadence."""
    if cadence_seconds <= 0:
        raise ValueError("cadence_seconds must be positive")
    reference_frame_id = frame_ids[0] if reference_frame_id is None else int(reference_frame_id)
    if reference_frame_id not in frame_ids:
        raise ValueError(f"reference frame {reference_frame_id} is not among {frame_ids}")
    return {
        frame_id: reference_date
        + timedelta(seconds=(frame_id - reference_frame_id) * cadence_seconds)
        for frame_id in frame_ids
    }


def _blank_map(coordinate, time, *, resolution, fov_rsun, wavelength):
    # The field of view is defined in solar radii so that every observer frames
    # the same region regardless of its distance (Mercury ... Neptune).
    angular_radius = np.arcsin((1.0 * u.R_sun / coordinate.radius).decompose().value)
    fov_arcsec = fov_rsun * (angular_radius * u.rad).to_value(u.arcsec)
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, obstime=time, observer=coordinate,
        frame=frames.Helioprojective,
    )
    scale = [fov_arcsec / resolution, fov_arcsec / resolution] * u.arcsec / u.pix
    data = np.zeros((resolution, resolution), dtype=np.float32)
    header = make_fitswcs_header(
        data, reference, scale=scale, wavelength=wavelength * u.AA, exposure=1.0 * u.s,
    )
    return Map(data, header)


@torch.no_grad()
def render_observation(
    rendering, instrument, rays, time, *, batch_size, device, progress=None,
):
    """Render one observer; returns ``(channel, y, x)`` and the ray validity.

    ``progress`` is an optional tqdm bar that is advanced by the rendered rays.
    """
    shape = rays.shape[:2]
    flat_rays = torch.as_tensor(rays.reshape(-1, 2, 3))
    finite = torch.isfinite(flat_rays).all(dim=(1, 2))
    images, validity = [], []
    for start in range(0, flat_rays.shape[0], batch_size):
        batch_rays = flat_rays[start:start + batch_size].to(device)
        batch = {instrument: {
            "rays": batch_rays,
            "time": torch.full((batch_rays.shape[0], 1), float(time), device=device),
            "instrument": instrument,
        }}
        output = rendering(batch, shuffle=False, diagnostics=False)
        images.append(output["model_out"][instrument]["image"].cpu())
        validity.append(output["ray_valid"].cpu())
        if progress is not None:
            progress.update(batch_rays.shape[0])
    image = torch.cat(images).numpy()
    valid = (torch.cat(validity) & finite).numpy()
    image = np.where(valid[:, None], image, np.nan)
    image = image.reshape(*shape, -1).transpose(2, 0, 1)
    return image.astype(np.float32), valid.reshape(shape)


def _write_channel(path, blank_map, image, valid, *, instrument, channel, renderer, provenance):
    specification = INSTRUMENTS[instrument]
    response_provenance = renderer.response_provenance
    meta = blank_map.meta.copy()
    meta.update(specification["header"])
    measurement_unit = (u.Unit(renderer.response_unit) / u.cm**5).to_string()
    pixel_solid_angle = float(
        (blank_map.scale[0] * blank_map.scale[1] * u.pix**2).to_value(u.sr)
    )
    meta.update({
        "wavelnth": _channel_wavelength(channel),
        "waveunit": "angstrom",
        "bunit": measurement_unit,
        "exptime": 1.0,
        "prepschm": PREPARED_EUV_SCHEMA,
        "schemav": 2,
        "cal_id": f"sunerf-psi-synthetic:{renderer.response_id}"[:68],
        "senscon": response_provenance["sensitivity_convention"],
        "maskext": "VALID_MASK",
        "natpxsr": float(response_provenance.get(
            "native_pixel_solid_angle_sr", pixel_solid_angle
        )),
        "prepxsr": pixel_solid_angle,
        "radsem": response_provenance["measurement_semantics"],
        "geomsem": "sample_interp_no_solid_angle_conversion",
        "prepdate": datetime.now(timezone.utc).isoformat(),
        "srcfile": provenance["source"][:68],
        "psiframe": provenance["frame_id"],
        "history": [
            "SuNeRF synthetic prepared-EUV-v2 rendered from a PSI/MAS cube",
            f"temperature cutoff: {json.dumps(provenance['temperature_cutoff'])}",
            f"absorption: {json.dumps(provenance['absorption'])}",
        ],
    })
    image = np.where(valid, image, np.nan).astype(np.float32)
    _atomic_write_fits(Map(image, meta), valid, path, overwrite=True)


def channel_colormap(instrument, channel):
    """Instrument colormap of one channel (SunPy), or a neutral fallback."""
    import matplotlib
    import sunpy.visualization.colormaps  # noqa: F401  (registers the solar colormaps)

    name = INSTRUMENTS[instrument].get("cmap", "").format(wavelength=_channel_wavelength(channel))
    return name if name in matplotlib.colormaps else "gray"


def write_overview(path, image, valid, *, instrument, channels, title, fov_rsun, unit):
    """One JPEG with every rendered channel of a frame, in the instrument colormaps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    n_channels = len(channels)
    figure, axes = plt.subplots(
        1, n_channels, figsize=(3.0 * n_channels, 3.4), squeeze=False, facecolor="black",
    )
    extent = [-0.5 * fov_rsun, 0.5 * fov_rsun, -0.5 * fov_rsun, 0.5 * fov_rsun]
    for axis, channel, data in zip(axes[0], channels, image):
        data = np.where(valid, data, np.nan)
        finite = data[np.isfinite(data)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        colormap = matplotlib.colormaps[channel_colormap(instrument, channel)].with_extremes(
            bad="black"
        )
        axis.imshow(
            data, origin="lower", extent=extent, cmap=colormap,
            norm=PowerNorm(gamma=0.35, vmin=0.0, vmax=max(vmax, np.finfo(np.float32).tiny)),
            interpolation="nearest",
        )
        axis.set_title(
            f"{_channel_wavelength(channel)} \u00c5   99.5%: {vmax:.3g} {unit}",
            color="white", fontsize=8,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, color="white", fontsize=9)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.86, wspace=0.02)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130, facecolor="black", pil_kwargs={"quality": 90})
    plt.close(figure)
    return path


def render_observer(
    psi_data,
    out_path,
    *,
    instrument,
    location,
    reference_date,
    cadence_seconds,
    reference_frame_id=None,
    frame_ids=None,
    response=None,
    temperature_cutoff=DEFAULT_CUTOFF,
    absorption_artifact="builtin:h_he_photoionization",
    resolution=256,
    fov_rsun=3.1,
    min_radius=1.0,
    max_radius=1.5,
    n_samples=256,
    n_hierarchical_samples=256,
    density_unit_cm3=MAS_DENSITY_UNIT_CM3,
    temperature_unit_K=MAS_TEMPERATURE_UNIT_K,
    batch_size=8192,
    device=None,
    light_travel_time=True,
    overview=True,
    show_progress=True,
):
    """Render every PSI snapshot pair for one instrument at one observer location.

    ``psi_data`` is the directory that contains the ``rho`` and ``t`` folders.
    One prepared-EUV FITS file per snapshot and channel is written directly into
    ``out_path``, together with ``observer.json``. ``temperature_cutoff`` and
    ``absorption_artifact`` may be ``None`` to disable them.

    With ``light_travel_time`` the FITS timestamp is the detector time: the
    snapshot (emission) time plus the Sun-observer light travel time. A
    reconstruction with ``module.light_travel_time: true`` then maps every
    observer back onto the common snapshot time.

    ``overview`` writes one ``<time>_overview.jpg`` per frame next to the FITS
    files with all rendered channels in their instrument colormaps.
    """
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument must be one of {sorted(INSTRUMENTS)}; received {instrument!r}")
    if isinstance(reference_date, str):
        reference_date = datetime.fromisoformat(reference_date)
    device = torch.device(
        device if device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    specification = INSTRUMENTS[instrument]
    channels = specification["channels"]
    response = specification["response"] if response is None else response
    observer_coordinate(location, reference_date)  # reject an invalid location early

    # One frame per snapshot pair found below <psi_data>/rho and <psi_data>/t.
    pairs = pair_psi_files(psi_data, frame_ids=frame_ids)
    snapshot_ids = [_frame_id(density_path) for density_path, _ in pairs]
    times = frame_times(
        snapshot_ids, reference_date=reference_date, cadence_seconds=cadence_seconds,
        reference_frame_id=reference_frame_id,
    )

    log_T_range = np.asarray(load_response_artifact(response).log_temperature, dtype=np.float32)
    response_config = {
        "artifact": response, "channels": list(channels), "learnable": False, "Rs_per_ds": 1.0,
    }
    if temperature_cutoff is not None:
        response_config["temperature_cutoff"] = dict(temperature_cutoff)
    absorption_config = None if absorption_artifact is None else {
        "type": "photoionization",
        "artifact": absorption_artifact,
        "hydrogen_density_convention": "fully_ionized_proxy",
    }
    absorption_model = None if absorption_config is None else init_absorption_model(
        absorption_config, instrument_key=instrument, channels=list(channels),
    )
    renderer = PlasmaRadiativeTransfer(
        response_config, log_T_range, absorption_model=absorption_model
    ).to(device).eval()

    out_path = Path(out_path)
    ray_loader = MapDataLoader(Rs_per_ds=1.0, max_radius=max_radius)
    records = []
    progress = tqdm(
        total=len(pairs) * resolution * resolution, unit="ray", unit_scale=True,
        desc=f"{instrument} @ {location}", disable=not show_progress,
    )
    for index, ((density_path, _), snapshot_id) in enumerate(zip(pairs, snapshot_ids)):
        progress.set_postfix_str(f"snapshot {index + 1}/{len(pairs)} #{snapshot_id}")
        time = times[snapshot_id]
        # Each snapshot is its own static grid observed at its own time.
        grid = load_psi_grid(
            psi_data,
            log_T=log_T_range,
            density_unit_scale_cm3=density_unit_cm3,
            temperature_unit_scale_K=temperature_unit_K,
            reference_frame_id=snapshot_id,
            longitude_frame="carrington",
            frame_ids=[snapshot_id],
            ref_date=time,
            min_radius=min_radius,
            max_radius=max_radius,
        )
        rendering = BasicRenderingModule(
            model=grid.model,
            rendering_modules={instrument: renderer},
            Rs_per_ds=1.0,
            seconds_per_dt=86400.0,
            sampling_config={
                "type": "spherical", "min_distance": min_radius, "max_distance": max_radius,
                "n_samples": n_samples, "perturb": False, "radial_weighting": True,
            },
            hierarchical_sampling_config={
                "type": "hierarchical", "n_samples": n_hierarchical_samples, "perturb": False,
            },
        ).to(device).eval()
        snapshot_time = time
        if light_travel_time:
            # One fixed-point step is exact to well below a second because the
            # observer moves ~1e-4 AU during the light travel time.
            delay = observer_coordinate(location, time).radius.to(u.m) / const.c
            time = snapshot_time + timedelta(seconds=float(delay.to_value(u.s)))
        coordinate = observer_coordinate(location, time)
        blank = _blank_map(
            coordinate, time, resolution=resolution, fov_rsun=fov_rsun,
            wavelength=_channel_wavelength(channels[0]),
        )
        image, valid = render_observation(
            rendering, instrument, ray_loader.load(blank)["rays"], 0.0,
            batch_size=batch_size, device=device, progress=progress,
        )
        provenance = {
            "source": Path(density_path).name,
            "frame_id": int(snapshot_id),
            "temperature_cutoff": temperature_cutoff,
            "absorption": absorption_config,
        }
        stamp = time.strftime("%Y%m%dT%H%M%S")
        for channel_index, channel in enumerate(channels):
            channel_map = _blank_map(
                coordinate, time, resolution=resolution, fov_rsun=fov_rsun,
                wavelength=_channel_wavelength(channel),
            )
            _write_channel(
                out_path / f"{stamp}_{_channel_wavelength(channel)}.prepared.fits",
                channel_map, image[channel_index], valid, instrument=instrument,
                channel=channel, renderer=renderer, provenance=provenance,
            )
        if overview:
            write_overview(
                out_path / f"{stamp}_overview.jpg", image, valid,
                instrument=instrument, channels=channels, fov_rsun=fov_rsun,
                unit=(u.Unit(renderer.response_unit) / u.cm**5).to_string(),
                title=(
                    f"{instrument} @ {location}   PSI snapshot #{snapshot_id}   "
                    f"{time.isoformat(timespec='seconds')}   "
                    f"lon {coordinate.lon.to_value(u.deg):.1f}\u00b0  "
                    f"lat {coordinate.lat.to_value(u.deg):.1f}\u00b0  "
                    f"{coordinate.radius.to_value(u.AU):.3f} AU"
                ),
            )
        records.append({
            "frame_id": int(snapshot_id),
            "time": time.isoformat(),
            "snapshot_time": snapshot_time.isoformat(),
            "stonyhurst_longitude_deg": float(coordinate.lon.to_value(u.deg)),
            "stonyhurst_latitude_deg": float(coordinate.lat.to_value(u.deg)),
            "distance_au": float(coordinate.radius.to_value(u.AU)),
            "carrington_longitude_deg": float(blank.carrington_longitude.to_value(u.deg)),
            "median_valid": [float(np.nanmedian(image[i])) for i in range(len(channels))],
        })
        tqdm.write(
            f"[{instrument} @ {location}] snapshot {index + 1}/{len(pairs)} #{snapshot_id} "
            f"{stamp} lon={records[-1]['stonyhurst_longitude_deg']:.1f} "
            f"lat={records[-1]['stonyhurst_latitude_deg']:.1f} "
            f"d={records[-1]['distance_au']:.3f} AU"
        )
        del rendering, grid
    progress.close()

    with open(out_path / "observer.json", "w", encoding="utf-8") as stream:
        json.dump({
            "instrument": instrument,
            "location": str(location),
            "channels": list(channels),
            "light_travel_time": bool(light_travel_time),
            "time_source": {
                "reference_date": reference_date.isoformat(),
                "cadence_seconds": float(cadence_seconds),
                "reference_frame_id": int(
                    snapshot_ids[0] if reference_frame_id is None else reference_frame_id
                ),
                "note": "PSI HDF5 cubes contain no time information",
            },
            "observations": records,
        }, stream, indent=2)
    return records


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--psi-data", required=True,
        help="Directory that contains the 'rho' and 't' folders with paired HDF5 snapshots.",
    )
    parser.add_argument(
        "--out-path", required=True,
        help="Directory that receives the prepared FITS files of this observer.",
    )
    parser.add_argument("--instrument", required=True, choices=list(INSTRUMENTS))
    parser.add_argument(
        "--location", required=True,
        help=(
            f"Observer location: one of {', '.join(OBSERVER_KEYS)} or Stonyhurst "
            "'longitude_deg,latitude_deg,distance_au'."
        ),
    )
    parser.add_argument(
        "--reference-date", required=True,
        help="ISO date of the reference snapshot; the HDF5 cubes contain no time information.",
    )
    parser.add_argument(
        "--cadence-seconds", type=float, required=True,
        help="Simulation time between consecutive snapshot numbers.",
    )
    parser.add_argument(
        "--reference-frame-id", type=int, default=None,
        help="Snapshot number observed at --reference-date (default: the first snapshot).",
    )
    parser.add_argument("--frame-ids", type=int, nargs="+", default=None,
                        help="Render only these snapshot numbers (default: every pair).")
    parser.add_argument("--response", default=None,
                        help="Response table (path or builtin:<name>); default: packaged table.")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument(
        "--fov-rsun", type=float, default=3.1,
        help="Full width of the field of view in solar radii (same for every observer).",
    )
    parser.add_argument("--min-radius", type=float, default=1.0)
    parser.add_argument("--max-radius", type=float, default=1.5)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--n-hierarchical-samples", type=int, default=256)
    parser.add_argument("--density-unit-cm3", type=float, default=MAS_DENSITY_UNIT_CM3)
    parser.add_argument("--temperature-unit-k", type=float, default=MAS_TEMPERATURE_UNIT_K)
    parser.add_argument("--no-temperature-cutoff", action="store_true")
    parser.add_argument("--absorption-artifact", default="builtin:h_he_photoionization")
    parser.add_argument("--no-absorption", action="store_true")
    parser.add_argument(
        "--no-light-travel-time", action="store_true",
        help="Stamp the snapshot time instead of the delayed detector time.",
    )
    parser.add_argument("--no-overview", action="store_true",
                        help="Do not write the per-frame overview JPEG.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    render_observer(
        args.psi_data,
        args.out_path,
        instrument=args.instrument,
        location=args.location,
        reference_date=args.reference_date,
        cadence_seconds=args.cadence_seconds,
        reference_frame_id=args.reference_frame_id,
        frame_ids=args.frame_ids,
        response=args.response,
        temperature_cutoff=None if args.no_temperature_cutoff else DEFAULT_CUTOFF,
        absorption_artifact=None if args.no_absorption else args.absorption_artifact,
        resolution=args.resolution,
        fov_rsun=args.fov_rsun,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        n_samples=args.n_samples,
        n_hierarchical_samples=args.n_hierarchical_samples,
        density_unit_cm3=args.density_unit_cm3,
        temperature_unit_K=args.temperature_unit_k,
        batch_size=args.batch_size,
        device=args.device,
        light_travel_time=not args.no_light_travel_time,
        overview=not args.no_overview,
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
