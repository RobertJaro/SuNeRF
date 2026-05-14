#!/usr/bin/env python3
"""Shared utilities for coronagraph preprocessing scripts."""

import argparse
import datetime as dt
import os
import re

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm


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


def sample_files_at_cadence(files, cadence: dt.timedelta):
    if cadence.total_seconds() <= 0:
        raise ValueError("Cadence must be positive.")

    timed_files = []
    for file_path in tqdm(files, desc="Loading observation times"):
        obs_time = get_observation_time(file_path)
        timed_files.append((obs_time, file_path))
    timed_files.sort(key=lambda item: item[0])

    sampled = []
    next_time = timed_files[0][0]
    last_added_path = None

    for obs_time, file_path in timed_files:
        if obs_time < next_time:
            continue
        if file_path != last_added_path:
            sampled.append(file_path)
            last_added_path = file_path
        next_time = obs_time + cadence

    return sampled


def add_common_prep_arguments(parser, *, include_max_radius=False, include_clip=False, include_value_limits=False):
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
        "--no_overwrite",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=None,
        help="Optional resize to (width height) in pixels.",
    )
    if include_clip:
        parser.add_argument(
            "--clip_min",
            type=float,
            default=None,
            help="Optional minimum value to clip data to.",
        )
        parser.add_argument(
            "--clip_max",
            type=float,
            default=None,
            help="Optional maximum value to clip data to.",
        )
    if include_value_limits:
        parser.add_argument(
            "--value_min",
            type=float,
            default=None,
            help="Optional minimum allowed data value.",
        )
        parser.add_argument(
            "--value_max",
            type=float,
            default=None,
            help="Optional maximum allowed data value.",
        )
    parser.add_argument(
        "--filter_bright_objects",
        action="store_true",
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
        "clip_min": getattr(args, "clip_min", None),
        "clip_max": getattr(args, "clip_max", None),
        "value_min": getattr(args, "value_min", None),
        "value_max": getattr(args, "value_max", None),
        "filter_bright_objects": args.filter_bright_objects,
        "bright_object_threshold": args.bright_object_threshold,
    }


def ensure_tb_pb_output_dirs(out_path):
    tb_out_path = os.path.join(out_path, "tB")
    pb_out_path = os.path.join(out_path, "pB")
    os.makedirs(tb_out_path, exist_ok=True)
    os.makedirs(pb_out_path, exist_ok=True)
    return tb_out_path, pb_out_path


class MapPreprocessor:
    """Applies shared map-level preprocessing after instrument-specific loading."""

    def __init__(
        self,
        occ_min=None,
        occ_max=None,
        max_radius=None,
        resize=None,
        clip_min=None,
        clip_max=None,
        value_min=None,
        value_max=None,
        filter_bright_objects=False,
        bright_object_threshold=10.0,
    ):
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.max_radius = max_radius
        self.resize = resize
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.value_min = value_min
        self.value_max = value_max
        self.filter_bright_objects = filter_bright_objects
        self.bright_object_threshold = bright_object_threshold

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
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.clip_min is not None or self.clip_max is not None:
            s_map.data[:] = np.clip(s_map.data, a_min=self.clip_min, a_max=self.clip_max)
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
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)

    if occ_min is not None:
        data[radius <= occ_min] = np.nan
    if occ_max is not None:
        data[radius >= occ_max] = np.nan

    projected_radius = None
    if max_radius is not None:
        projected_radius = (radius / s_map.rsun_obs).to_value(1)
        data[projected_radius >= max_radius.to_value(u.solRad)] = np.nan
    if filter_bright_objects:
        if projected_radius is None:
            projected_radius = (radius / s_map.rsun_obs).to_value(1)
        data = mask_bright_background_objects(
            data,
            projected_radius,
            threshold=bright_object_threshold,
        )

    return Map(data, s_map.meta)
