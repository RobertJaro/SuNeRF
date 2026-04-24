#!/usr/bin/env python3
"""

Batch-preprocess FITS images.

For each FITS file matched by --data_path, the script:
1) Loads the file as a SunPy Map.
2) Computes helioprojective coordinates for each pixel.
3) Masks (sets to NaN):
   - Pixels inside the occulter radius (default: 5500 arcsec).
   - Non-positive values (<= 0).
4) Saves the result to --output_path with the same basename.

Example
-------
python prep_coronagraph.py --data_path '/path/to/data/*.fits' --output_path '/path/to/output/' --occ_min 5500
"""

import argparse
import datetime as dt
import multiprocessing
import os
import re
from glob import glob

import numpy as np
from astropy import units as u
from astropy.time import Time
from sunpy.sun import constants
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm


def _load_map(file_path):
    """
    Load a Metis FITS file as a SunPy Map.

    Parameters
    ----------
    file_path : str
        Path to the FITS file.

    Returns
    -------
    sunpy.map.Map
        SunPy map constructed from FITS data and header.
    """
    try:
        data = fits.getdata(file_path)
        header = fits.getheader(file_path)
        if 'rsun_ref' not in header:
            header['rsun_ref'] = constants.radius.to_value(u.m)  # Add solar radius if missing
    except Exception as e:
        raise RuntimeError(f"Error loading FITS file {file_path}: {e}")
    return Map(data, header)


FILENAME_TS_PATTERNS = (
    re.compile(r"(?P<ts>\d{8}T\d{6})"),
    re.compile(r"(?P<ts>\d{8}_\d{6})"),
    re.compile(r"(?P<ts>\d{14})"),
)


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


def _datetime_from_filename(file_path: str):
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


def _get_observation_time(file_path: str):
    header = fits.getheader(file_path)
    for key in ("DATE-OBS", "DATE_OBS", "DATE-BEG", "DATE-AVG", "DATE-END"):
        value = header.get(key)
        if value in (None, ""):
            continue
        try:
            return Time(value).to_datetime()
        except Exception:
            continue
    fallback = _datetime_from_filename(file_path)
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Could not determine observation time for {file_path}")


def sample_files_at_cadence(files, cadence: dt.timedelta):
    if cadence.total_seconds() <= 0:
        raise ValueError("Cadence must be positive.")

    timed_files = []
    for file_path in tqdm(files, desc="Loading observation times"):
        obs_time = _get_observation_time(file_path)
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


def mask_radial_line(data, center, angle_deg, halfwidth_deg=2.0, r_min=0.0, r_max=np.inf, fill=np.nan):
    """
    Mask pixels near a radial line (through `center`) at a given position angle.

    Parameters
    ----------
    data : ndarray (H, W)
        Image.
    center : tuple[float, float]
        (cx, cy) in pixel coordinates, with x=column, y=row.
    angle_deg : float
        Line angle in degrees, measured CCW from +x axis (image columns).
        If you want solar-style PA (CCW from +y / north), see note below.
    halfwidth_deg : float
        Half-width of the masked angular band (degrees).
    r_min, r_max : float
        Optional radial limits (pixels) to only mask between these radii.
    fill : float
        Value to write into masked pixels (e.g., np.nan or 0).

    Returns
    -------
    out : ndarray
        Copy of `data` with masked pixels set to `fill`.
    mask : ndarray (bool)
        True where pixels were masked.
    """
    out = np.array(data, copy=True)
    cy, cx = center[1], center[0]  # keep user convention: (cx, cy)
    y, x = np.indices(out.shape)
    dx = x - cx
    dy = y - cy

    theta = np.arctan2(dy, dx)  # [-pi, pi], 0 along +x
    theta0 = np.deg2rad(angle_deg)

    # Smallest signed angular difference in [-pi, pi]
    dtheta = np.arctan2(np.sin(theta - theta0), np.cos(theta - theta0))

    r = np.hypot(dx, dy)

    mask = (np.abs(dtheta) <= np.deg2rad(halfwidth_deg)) & (r >= r_min) & (r <= r_max)
    out[mask] = fill
    return out, mask

def _prep_coronagraph_map(s_map, occ_min=None, occ_max=None, max_radius=None):
    """
    Preprocess a coronagraph map by masking the occulter and invalid values.

    Masking rules
    -------------
    - Pixels with helioprojective radius <= `occ_rad` are set to NaN.
    - Pixels with values <= 0 are set to NaN.

    Parameters
    ----------
    s_map : sunpy.map.Map
        Input coronagraph map.
    occ_rad : astropy.units.Quantity, optional
        Occulter radius in angular units (default: 5500 arcsec).

    Returns
    -------
    sunpy.map.Map
        New map with masked data and original metadata.
    """
    data = np.array(s_map.data, dtype=float, copy=True)
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)

    if occ_min is not None:
        data[radius <= occ_min] = np.nan
    if occ_max is not None:
        data[radius >= occ_max] = np.nan
    if max_radius is not None:
        projected_radius = (radius / s_map.rsun_obs).to_value(1)
        data[projected_radius >= max_radius.to_value(u.solRad)] = np.nan

    return Map(data, s_map.meta)


class CoronagraphPrep:
    """
    Callable helper for multiprocessing conversion of coronagraph FITS files.

    Parameters
    ----------
    out_path : str
        Output directory.
    overwrite : bool, optional
        If False, existing outputs are skipped. Default is True.
    occ_rad : astropy.units.Quantity, optional
        Occulter radius used for masking. Default is 5500 arcsec.
    """

    def __init__(self, out_path, overwrite=True, occ_min=None, occ_max=None, resize=None, clip_max=None, map_prep_func=None):
        self.out_path = out_path
        self.overwrite = overwrite
        self.occ_min = occ_min
        self.occ_max = occ_max
        self.resize = resize
        self.clip_max = clip_max
        self.map_prep_func = map_prep_func

    def convert(self, file_path):
        """
        Load, preprocess, and save one coronagraph FITS file.

        Parameters
        ----------
        file_path : str
            Input FITS file path.

        Returns
        -------
        str
            Output file path.
        """
        out_path = os.path.join(self.out_path, os.path.basename(file_path))
        if os.path.exists(out_path) and not self.overwrite:
            print(f"File {out_path} already exists. Skipping.")
            return out_path

        s_map = _load_map(file_path)
        if self.map_prep_func is not None:
            s_map = self.map_prep_func(s_map)
        s_map = _prep_coronagraph_map(s_map, occ_min=self.occ_min, occ_max=self.occ_max)
        if self.resize is not None:
            s_map = s_map.resample(self.resize * u.pixel)
        if self.clip_max is not None:
            s_map.data[:] = np.clip(s_map.data, a_max=self.clip_max, a_min=None)
        s_map.save(out_path, overwrite=True)
        return out_path


def main():
    """
    CLI entry point for batch preprocessing.
    """
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data_path", type=str, required=True, help="Glob pattern for FITS files.")
    p.add_argument("--out_path", type=str, required=True, help="Output directory for preprocessed maps.")
    p.add_argument(
        "--occ_min",
        type=float,
        default=None,
        help="Minimum occulter radius in arcseconds.",
    )
    p.add_argument(
        "--occ_max",
        type=float,
        default=None,
        help="Maximum occulter radius in arcseconds.",
    )
    p.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Skip outputs that already exist.",
    )
    p.add_argument('--clip_max', type=float, default=None,
                   help='Optional maximum value to clip data to.')
    p.add_argument('--num_workers', type=int, default=os.cpu_count(),)
    p.add_argument('--resize', type=int, nargs=2, default=None,
                   help='Optional resize to (width height) in pixels.')
    p.add_argument(
        "--cadence",
        type=parse_duration,
        default=None,
        help="Optional fixed sampling cadence like 15m, 1h, or 30s.",
    )
    args = p.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    files = sorted(glob(args.data_path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.data_path}")
    if args.cadence is not None:
        original_count = len(files)
        files = sample_files_at_cadence(files, args.cadence)
        print(
            f"Cadence sampling kept {len(files)} of {original_count} files "
            f"at {args.cadence} spacing."
        )

    prepper = CoronagraphPrep(
        args.out_path,
        overwrite=not args.no_overwrite,
        occ_min=args.occ_min * u.arcsec if args.occ_min is not None else None,
        occ_max=args.occ_max * u.arcsec if args.occ_max is not None else None,
        resize=args.resize,
        clip_max=args.clip_max,
    )

    with multiprocessing.Pool(args.num_workers) as p:
        out_files = [
            f
            for f in tqdm(
                p.imap(prepper.convert, files),
                total=len(files),
                desc="Preprocessing maps",
            )
        ]

    print(f"Preprocessed {len(out_files)} maps saved to {args.out_path}.")


if __name__ == "__main__":
    main()
