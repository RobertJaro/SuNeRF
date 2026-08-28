#!/usr/bin/env python3
"""Shared utilities for coronagraph preprocessing scripts."""

import argparse
import datetime as dt
import os
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy import log as sunpy_log
from sunpy.coordinates import frames
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

from sunerf.data.ray_sampling import hpc_angular_separation, hpc_impact_parameter


DEFAULT_SOLAR_SYSTEM_OBJECTS = (
    "venus",
    "mercury",
    "moon",
)
FILENAME_TS_PATTERNS = (
    re.compile(r"(?P<ts>\d{8}T\d{6})"),
    re.compile(r"(?P<ts>\d{8}_\d{6})"),
    re.compile(r"(?P<ts>\d{14})"),
)


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def parse_duration(value: str) -> dt.timedelta:
    match = re.fullmatch(r"(?i)\s*(\d+)\s*([smhd])\s*", value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid duration '{value}'. Use formats like 30s, 15m, 1h."
        )
    quantity = int(match.group(1))
    unit = match.group(2).lower()
    seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return dt.timedelta(seconds=quantity * seconds_per_unit)


def parse_time(value: str) -> dt.datetime:
    try:
        return Time(value).to_datetime()
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid time '{value}': {exc}") from exc


def datetime_from_filename(file_path: str):
    name = os.path.basename(file_path)
    for pattern in FILENAME_TS_PATTERNS:
        match = pattern.search(name)
        if match is None:
            continue
        stamp = match.group("ts")
        if "T" in stamp:
            return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%S")
        if "_" in stamp:
            return dt.datetime.strptime(stamp, "%Y%m%d_%H%M%S")
        return dt.datetime.strptime(stamp, "%Y%m%d%H%M%S")
    return None


def get_observation_time(file_path: str):
    header = fits.getheader(file_path)
    for key in ("DATE-OBS", "DATE_OBS", "DATE-BEG", "DATE-AVG", "DATE-END"):
        value = header.get(key)
        if value in (None, ""):
            continue
        try:
            return Time(value).to_datetime()
        except Exception:
            continue
    fallback = datetime_from_filename(file_path)
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Could not determine observation time for {file_path}")


def _get_item_observation_time(item):
    if isinstance(item, (str, os.PathLike)):
        return get_observation_time(item)
    return min(get_observation_time(file_path) for file_path in item)


def select_items_by_time(items, start=None, end=None, cadence=None, get_time=_get_item_observation_time):
    items = list(items)
    if start is None and end is None and cadence is None:
        return items
    if start is not None and end is not None and start >= end:
        raise ValueError("Start time must be earlier than end time.")
    if cadence is not None and cadence.total_seconds() <= 0:
        raise ValueError("Cadence must be positive.")

    with ThreadPoolExecutor() as executor:
        observation_times = list(
            tqdm(
                executor.map(get_time, items),
                total=len(items),
                desc="Loading observation times",
            )
        )

    timed_items = sorted(zip(observation_times, items), key=lambda item: item[0])
    timed_items = [
        (obs_time, item)
        for obs_time, item in timed_items
        if (start is None or obs_time >= start) and (end is None or obs_time < end)
    ]
    if cadence is None or not timed_items:
        return [item for _, item in timed_items]

    sampled = []
    next_time = timed_items[0][0]
    for obs_time, item in timed_items:
        if obs_time < next_time:
            continue
        sampled.append(item)
        next_time = obs_time + cadence
    return sampled


def sample_files_at_cadence(files, cadence: dt.timedelta):
    return select_items_by_time(files, cadence=cadence)


def add_cadence_argument(parser):
    parser.add_argument(
        "--cadence",
        type=parse_duration,
        default=None,
        help="Optional preprocessing cadence such as 15m, 1h, or 1d.",
    )


def add_common_prep_arguments(parser, *, include_max_radius=False):
    parser.add_argument(
        "--start",
        type=parse_time,
        default=None,
        help="Inclusive preprocessing start time.",
    )
    parser.add_argument(
        "--end",
        type=parse_time,
        default=None,
        help="Exclusive preprocessing end time.",
    )
    parser.add_argument(
        "--occ_min",
        type=float,
        default=None,
        help="Minimum occulter radius in arcseconds.",
    )
    parser.add_argument(
        "--occ_max",
        type=float,
        default=None,
        help="Maximum occulter radius in arcseconds.",
    )
    if include_max_radius:
        parser.add_argument(
            "--max_radius",
            type=float,
            default=None,
            help="Maximum projected radius in solar radii.",
        )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs. By default existing outputs are skipped.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize to (width height) in pixels.",
    )
    parser.add_argument(
        "--value_min",
        type=float,
        default=None,
        help="Optional minimum allowed data value; smaller values are set to NaN.",
    )
    parser.add_argument(
        "--value_max",
        type=float,
        default=None,
        help="Optional maximum allowed data value; larger values are set to NaN.",
    )
    parser.add_argument(
        "--filter_bright_objects",
        action="store_true",
        default=False,
        help=(
            "Mask bright background stars/planets by fitting each image's radial "
            "brightness profile and setting outliers to NaN."
        ),
    )
    parser.add_argument(
        "--filter_bright_background_objects",
        dest="filter_bright_objects",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bright_object_threshold",
        type=positive_float,
        default=10.0,
        help=(
            "Mask pixels this many times brighter than the fitted radial background "
            "when --filter_bright_objects is enabled."
        ),
    )
    parser.add_argument(
        "--bright_background_threshold",
        dest="bright_object_threshold",
        type=positive_float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--remove_solar_system_objects",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Locate projected solar-system objects in the image and apply "
            "circular NaN masks around them."
        ),
    )
    parser.add_argument(
        "--solar_system_objects",
        nargs="+",
        default=DEFAULT_SOLAR_SYSTEM_OBJECTS,
        help="Solar-system body names to mask when --remove_solar_system_objects is enabled.",
    )
    parser.add_argument(
        "--solar_system_object_mask_radius",
        type=positive_float,
        default=1000.0,
        help="Circular helioprojective mask radius in arcseconds for non-lunar objects.",
    )
    parser.add_argument(
        "--moon_mask_radius",
        type=positive_float,
        default=2000.0,
        help="Circular helioprojective mask radius in arcseconds for the Moon.",
    )


def common_kwargs_from_args(args):
    return {
        "occ_min": args.occ_min * u.arcsec if args.occ_min is not None else None,
        "occ_max": args.occ_max * u.arcsec if args.occ_max is not None else None,
        "max_radius": (
            args.max_radius * u.solRad
            if hasattr(args, "max_radius") and args.max_radius is not None
            else None
        ),
        "resize": args.resize,
        "value_min": getattr(args, "value_min", None),
        "value_max": getattr(args, "value_max", None),
        "filter_bright_objects": args.filter_bright_objects,
        "bright_object_threshold": args.bright_object_threshold,
        "remove_solar_system_objects": args.remove_solar_system_objects,
        "solar_system_objects": args.solar_system_objects,
        "solar_system_object_mask_radius": args.solar_system_object_mask_radius,
        "moon_mask_radius": args.moon_mask_radius,
    }


def ensure_tb_pb_output_dirs(out_path):
    tb_out_path = os.path.join(out_path, "tB")
    pb_out_path = os.path.join(out_path, "pB")
    os.makedirs(tb_out_path, exist_ok=True)
    os.makedirs(pb_out_path, exist_ok=True)
    return tb_out_path, pb_out_path


def should_write_output(path, overwrite=False):
    """Return whether an output may be written without violating overwrite policy."""
    return overwrite or not os.path.exists(path)


class MapPreprocessor:
    """Applies shared map-level preprocessing after instrument-specific loading."""

    def __init__(
        self,
        occ_min=None,
        occ_max=None,
        max_radius=None,
        resize=None,
        value_min=None,
        value_max=None,
        filter_bright_objects=False,
        bright_object_threshold=10.0,
        remove_solar_system_objects=True,
        solar_system_objects=DEFAULT_SOLAR_SYSTEM_OBJECTS,
        solar_system_object_mask_radius=1000.0,
        moon_mask_radius=2000.0,
    ):
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.max_radius = max_radius
        self.resize = resize
        self.value_min = value_min
        self.value_max = value_max
        self.filter_bright_objects = filter_bright_objects
        self.bright_object_threshold = bright_object_threshold
        self.remove_solar_system_objects = remove_solar_system_objects
        self.solar_system_objects = tuple(solar_system_objects or DEFAULT_SOLAR_SYSTEM_OBJECTS)
        self.solar_system_object_mask_radius = solar_system_object_mask_radius
        self.moon_mask_radius = moon_mask_radius

    def prepare_map(self, s_map):
        if self._needs_radial_preprocessing:
            s_map = prep_coronagraph_map(
                s_map,
                occ_min=self.occ_min,
                occ_max=self.occ_max,
                max_radius=self.max_radius,
                filter_bright_objects=self.filter_bright_objects,
                bright_object_threshold=self.bright_object_threshold,
            )
        if self.remove_solar_system_objects:
            s_map = mask_solar_system_objects(
                s_map,
                objects=self.solar_system_objects,
                object_mask_radius=self.solar_system_object_mask_radius,
                moon_mask_radius=self.moon_mask_radius,
            )
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.value_min is not None:
            s_map.data[s_map.data < self.value_min] = np.nan
        if self.value_max is not None:
            s_map.data[s_map.data > self.value_max] = np.nan
        return s_map

    @property
    def _needs_radial_preprocessing(self):
        return any(
            option is not None
            for option in (self.occ_min, self.occ_max, self.max_radius)
        ) or self.filter_bright_objects


def fit_radial_background(radius, brightness, degree=3):
    """Fit expected positive brightness as a function of projected radius."""
    valid = (
        np.isfinite(radius)
        & np.isfinite(brightness)
        & (radius > 0)
        & (brightness > 0)
    )
    if np.count_nonzero(valid) < degree + 1:
        return None

    radius_values = radius[valid]
    if np.min(radius_values) == np.max(radius_values):
        return None

    log_brightness = np.log(brightness[valid])
    return np.polyfit(radius_values, log_brightness, deg=degree)


def mask_bright_background_objects(data, radius, threshold=10.0, degree=3):
    """Mask compact objects that are much brighter than the fitted radial background."""
    coeffs = fit_radial_background(radius, data, degree=degree)
    if coeffs is None:
        return data

    expected_brightness = np.exp(np.polyval(coeffs, radius))
    bright = (
        np.isfinite(data)
        & np.isfinite(expected_brightness)
        & (expected_brightness > 0)
        & (data > threshold * expected_brightness)
    )
    data[bright] = np.nan
    return data


def prep_coronagraph_map(
    s_map,
    occ_min=None,
    occ_max=None,
    max_radius=None,
    filter_bright_objects=False,
    bright_object_threshold=10.0,
):
    """Apply common coronagraph radial masks and optional bright-object filtering."""
    data = np.array(s_map.data, dtype=float, copy=True)
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    angular_radius = hpc_angular_separation(coords.Tx, coords.Ty)

    if occ_min is not None:
        data[angular_radius <= occ_min] = np.nan
    if occ_max is not None:
        data[angular_radius >= occ_max] = np.nan

    projected_radius = None
    if max_radius is not None:
        projected_radius = hpc_impact_parameter(
            coords.Tx, coords.Ty, s_map.dsun
        ).to_value(u.R_sun)
        data[projected_radius >= max_radius.to_value(u.solRad)] = np.nan
    if filter_bright_objects:
        if projected_radius is None:
            projected_radius = hpc_impact_parameter(
                coords.Tx, coords.Ty, s_map.dsun
            ).to_value(u.R_sun)
        data = mask_bright_background_objects(
            data,
            projected_radius,
            threshold=bright_object_threshold,
        )

    return Map(data, s_map.meta)


def _hpc_circle_mask(coordinates, center, radius):
    radius = u.Quantity(radius, u.arcsec)
    separation = hpc_angular_separation(
        coordinates.Tx,
        coordinates.Ty,
        center_Tx=center.Tx,
        center_Ty=center.Ty,
    )
    return separation <= radius


def mask_solar_system_objects(
    s_map,
    objects=DEFAULT_SOLAR_SYSTEM_OBJECTS,
    object_mask_radius=1000.0,
    moon_mask_radius=2000.0,
):
    observer = s_map.observer_coordinate
    hpc_frame = frames.Helioprojective(observer=observer, obstime=s_map.date)

    # Fetch the projected HPC location of each requested body.
    object_locations = []
    for body in objects:
        body_name = body.lower()
        try:
            previous_log_level = sunpy_log.level
            sunpy_log.setLevel("WARNING")
            try:
                coord = get_body_heliographic_stonyhurst(
                    body_name,
                    s_map.date,
                    observer=observer,
                )
            finally:
                sunpy_log.setLevel(previous_log_level)
            hpc_coord = SkyCoord(coord.transform_to(hpc_frame))
        except Exception as exc:
            print(
                f"[solar-system-mask] Failed to project {body_name} "
                f"at {s_map.date.isot}: {exc}",
                flush=True,
            )
            continue
        object_locations.append((body_name, hpc_coord))

    # Build one combined mask for all bodies whose centers fall inside the FOV.
    mask = np.zeros(s_map.data.shape, dtype=bool)
    coordinates = None
    height, width = s_map.data.shape
    for body_name, hpc_coord in object_locations:
        x, y = s_map.world_to_pixel(hpc_coord)
        x_value = x.to_value(u.pixel)
        y_value = y.to_value(u.pixel)
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        if not (0 <= x_value < width and 0 <= y_value < height):
            continue

        mask_radius = moon_mask_radius if body_name == "moon" else object_mask_radius
        if coordinates is None:
            coordinates = all_coordinates_from_map(s_map)
        object_mask = _hpc_circle_mask(coordinates, hpc_coord, mask_radius)
        mask |= object_mask
        print(
            f"[solar-system-mask] {body_name} is in the FOV at {s_map.date.isot}: "
            f"HPC=({hpc_coord.Tx.to_value(u.arcsec):.1f}, "
            f"{hpc_coord.Ty.to_value(u.arcsec):.1f}) arcsec, "
            f"pixel=({x_value:.1f}, {y_value:.1f}), "
            f"mask_radius={u.Quantity(mask_radius, u.arcsec).to_value(u.arcsec):.1f} arcsec, "
            f"mask_pixels={np.count_nonzero(object_mask)}",
            flush=True,
        )

    # Apply the combined mask to the frame once.
    data = np.array(s_map.data, dtype=float, copy=True)
    newly_masked = np.count_nonzero(mask & np.isfinite(data))
    data[mask] = np.nan
    if np.any(mask):
        print(
            f"[solar-system-mask] Applied combined mask: "
            f"mask_pixels={np.count_nonzero(mask)}, newly_masked={newly_masked}",
            flush=True,
        )

    return Map(data, s_map.meta)
