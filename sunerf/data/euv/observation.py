"""Common contract for calibrated EUV observations.

The routines in this module deliberately do not perform instrument calibration.
They are the boundary between instrument-specific preparation and SuNeRF: a
prepared FITS product is validated, assigned stable channel metadata, and
converted to one consistent in-memory layout before rays are constructed.
"""

from __future__ import annotations

import multiprocessing
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from sunpy.map import Map
from tqdm import tqdm

from sunerf.configuration import canonical_channel_id, canonical_channel_mapping
from sunerf.response import SENSITIVITY_CONVENTIONS


PREPARED_EUV_SCHEMA = 'sunerf.prepared_euv.v2'


def load_prepared_map(path: str | Path):
    """Load the primary prepared image, never the VALID_MASK image extension."""
    return load_prepared_observation(path)[0]


def load_prepared_observation(path: str | Path, *, required_mask=False, saturation_dn=None):
    """Load the prepared image and its VALID_MASK extension in one file open.

    Returns ``(map, valid_mask)``; the mask is all-true when the file carries
    no mask extension and ``required_mask`` is false. Reading both HDUs here
    avoids a second open of every file on a network file system.

    With ``saturation_dn`` (detector DN, ``instruments[].saturation_dn``) the
    pixels whose ``PEAK_DN`` reaches it, i.e. saturated native pixels with their
    column bleed and interpolation reach, are invalid as well. A file without
    ``PEAK_DN`` is loaded without a saturation mask.
    """
    with fits.open(path, memmap=False) as hdul:
        data = np.array(hdul[0].data, copy=True)
        header = hdul[0].header.copy()
        if data.ndim != 2:
            raise ValueError(f'Prepared EUV primary HDU must be a 2D image: {path}.')
        valid_mask = _valid_mask_from_hdul(hdul, path, data.shape, required=required_mask)
        if saturation_dn is not None:
            valid_mask = valid_mask & ~saturated_pixels(hdul, path, data.shape, saturation_dn)
    _ensure_pc_matrix(header)
    return Map(data, header), valid_mask


PEAK_DN_EXTENSION = 'PEAK_DN'


def saturated_pixels(hdul, path, shape, saturation_dn):
    """Pixels whose peak detector DN (``PEAK_DN``) reach ``saturation_dn``.

    A product without ``PEAK_DN`` (prepared before the layer existed, or EUVI
    without DSATVAL) has no peak information, so none of its pixels are masked.
    """
    if PEAK_DN_EXTENSION not in hdul:
        # The message names no file, so the default warning filter reports it
        # once per process instead of once per frame.
        warnings.warn(
            f'Prepared EUV products without a {PEAK_DN_EXTENSION} extension are not masked for '
            'instruments[].saturation_dn; prepare them again with python -m '
            'sunerf.data.euv.prepare --overwrite to mask their saturated pixels.',
            stacklevel=2,
        )
        return np.zeros(tuple(shape), dtype=bool)
    peak = np.asarray(hdul[PEAK_DN_EXTENSION].data, dtype=np.float64)
    if peak.shape != tuple(shape):
        raise ValueError(
            f'{PEAK_DN_EXTENSION} shape {peak.shape} does not match image shape {tuple(shape)}.'
        )
    with np.errstate(invalid='ignore'):
        return np.isfinite(peak) & (peak >= float(saturation_dn))


def _ensure_pc_matrix(header: fits.Header) -> None:
    """Write PC keys from CROTA2 so sunpy never takes its CROTA branch.

    sunpy's EUIMap declares ``_rotation_matrix_from_crota`` as a property while
    ``GenericMap.rotation_matrix`` calls it, which fails on any prepared EUI
    file that carries only CROTA2 (sunpy 6 through 8).
    """
    if any(key in header for key in ('PC1_1', 'PC1_2', 'PC2_1', 'PC2_2', 'CD1_1', 'CD2_2')):
        return
    angle = np.deg2rad(float(header.get('CROTA2', header.get('CROTA', 0.0)) or 0.0))
    ratio = float(header.get('CDELT2', 1.0)) / float(header.get('CDELT1', 1.0))
    header['PC1_1'] = float(np.cos(angle))
    header['PC1_2'] = float(-ratio * np.sin(angle))
    header['PC2_1'] = float(np.sin(angle) / ratio)
    header['PC2_2'] = float(np.cos(angle))


def _utc_naive(value: Any) -> datetime:
    """Return a comparable UTC-naive ``datetime`` from common time values."""
    if hasattr(value, 'datetime'):
        value = value.datetime
    if not isinstance(value, datetime):
        value = parse(str(value))
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _header_time(header: fits.Header) -> datetime:
    for key in ('DATE-OBS', 'DATE_OBS', 'T_OBS', 'DATE-AVG'):
        if header.get(key):
            return _utc_naive(header[key])
    raise ValueError('Prepared EUV FITS file is missing an observation time.')


def _channel_number(value: Any) -> int:
    """Normalize numeric and prefixed channel labels (for example ``A171``)."""
    if isinstance(value, u.Quantity):
        value = value.to_value(u.AA)
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        digits = ''.join(character for character in str(value) if character.isdigit())
        if not digits:
            raise ValueError(f'Cannot determine EUV channel from {value!r}.')
        return int(digits)


def _header_channel(header: fits.Header) -> int:
    for key in ('WAVELNTH', 'WAVELENGTH', 'WAVE_LEN'):
        if header.get(key) is not None:
            return _channel_number(header[key])
    raise ValueError('Prepared EUV FITS file is missing WAVELNTH metadata.')


def _prepared_channel_and_time(path: str) -> tuple[int, datetime]:
    header = fits.getheader(path)
    return _header_channel(header), _header_time(header)


def discover_prepared_files(
        files: Sequence[str | Path], channel_ids: Sequence[int | str], *, workers: int = 0
) -> tuple[dict[int | str, list[str]], dict[int | str, list[datetime]]]:
    """Group prepared FITS paths by requested channel using FITS metadata.

    File-name parsing was historically different for every instrument and could
    silently put a file into the wrong channel. The FITS channel and time are the
    authoritative values at this boundary. Each header read opens one file, so
    ``workers`` > 1 reads them in a process pool (network file systems).
    """
    requested = tuple(channel_ids)
    requested_by_number = {_channel_number(channel): channel for channel in requested}
    if len(requested_by_number) != len(requested):
        raise ValueError(f'Channel identifiers are not unique: {requested!r}.')

    paths = [str(path) for path in files]
    description = 'Reading prepared FITS headers'
    if workers > 1 and len(paths) > 1:
        with multiprocessing.Pool(min(workers, len(paths))) as pool:
            headers = list(tqdm(
                pool.imap(_prepared_channel_and_time, paths, chunksize=16),
                total=len(paths), desc=description,
            ))
    else:
        headers = [_prepared_channel_and_time(path) for path in tqdm(paths, desc=description)]

    file_dict = {channel: [] for channel in requested}
    date_dict = {channel: [] for channel in requested}
    for path, (number, time) in zip(paths, headers):
        channel = requested_by_number.get(number)
        if channel is None:
            continue
        file_dict[channel].append(path)
        date_dict[channel].append(time)

    missing = [channel for channel in requested if not file_dict[channel]]
    if missing:
        raise ValueError(f'No prepared EUV files found for channels {missing!r}.')
    return file_dict, date_dict


def _fits_valid_mask(path, shape, *, required):
    path = Path(path)
    if not path.is_file():
        if required:
            raise ValueError(
                f'Strict prepared-v2 ingestion requires a readable FITS source: {path}.'
            )
        return np.ones(shape, dtype=bool)
    with fits.open(path, memmap=False) as hdul:
        return _valid_mask_from_hdul(hdul, path, shape, required=required)


def _valid_mask_from_hdul(hdul, path, shape, *, required):
    extension_name = str(hdul[0].header.get('MASKEXT', '')).strip()
    if required and extension_name != 'VALID_MASK':
        raise ValueError(
            f'Prepared-v2 FITS {path} must declare MASKEXT=VALID_MASK.'
        )
    if not extension_name and 'VALID_MASK' in hdul:
        extension_name = 'VALID_MASK'
    if not extension_name:
        return np.ones(shape, dtype=bool)
    if extension_name not in hdul:
        raise ValueError(
            f'Prepared FITS {path} declares missing mask extension {extension_name!r}.'
        )
    values = np.asarray(hdul[extension_name].data)
    if values.shape != tuple(shape):
        raise ValueError(
            f'VALID_MASK shape {values.shape} does not match image shape {tuple(shape)}.'
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isin(values, (0, 1))):
        raise ValueError('VALID_MASK must contain only binary 0/1 values.')
    return values.astype(bool)


def match_prepared_channels(
        file_dict: Mapping[Any, Sequence[str]],
        date_dict: Mapping[Any, Sequence[datetime]],
        *,
        tolerance: timedelta = timedelta(minutes=2),
        channel_order: Sequence[Any] | None = None,
) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], np.ndarray]:
    """Group prepared files of one instrument into multi-channel observations.

    Files of all channels are walked in time order. A file joins the current
    observation when its channel is still free there and it lies within
    ``tolerance`` of the observation's first file; otherwise it starts a new
    observation. Every input file is used at most once, and an observation is
    kept as soon as one channel is present: channels without a file within
    the tolerance are ``None`` in the returned per-channel arrays and are
    loaded as fully invalid (NaN) frames that the loss ignores.

    Returns ``(matched_files, matched_dates, observation_times)``; the
    observation time is the mean time of the channels present.
    """
    channels = tuple(file_dict) if channel_order is None else tuple(channel_order)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError('channel_order must contain at least one unique channel.')
    if set(channels) != set(file_dict) or set(channels) != set(date_dict):
        raise ValueError('file_dict, date_dict, and channel_order must contain identical channels.')
    tolerance_seconds = float(tolerance.total_seconds())
    if tolerance_seconds < 0:
        raise ValueError('Matching tolerance must be non-negative.')

    entries = []
    for channel_index, channel in enumerate(channels):
        paths = [str(value) for value in file_dict[channel]]
        dates = [_utc_naive(value) for value in date_dict[channel]]
        if len(paths) != len(dates):
            raise ValueError(f'Channel {channel!r} has different file and date counts.')
        if len(set(paths)) != len(paths):
            raise ValueError(f'Channel {channel!r} contains duplicate source paths.')
        entries.extend(zip(dates, [channel_index] * len(paths), paths))
    if not entries:
        raise ValueError('No prepared files to match.')
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    groups = []
    anchor = None
    current = None
    for date, channel_index, path in entries:
        if (
            current is None
            or (date - anchor).total_seconds() > tolerance_seconds
            or current[channel_index] is not None
        ):
            anchor = date
            current = [None] * len(channels)
            groups.append(current)
        current[channel_index] = (path, date)

    matched_files = {
        channel: np.array(
            [None if group[index] is None else group[index][0] for group in groups],
            dtype=object,
        )
        for index, channel in enumerate(channels)
    }
    matched_dates = {
        channel: np.array(
            [None if group[index] is None else group[index][1] for group in groups],
            dtype=object,
        )
        for index, channel in enumerate(channels)
    }
    observation_times = np.array(
        [observation_time([entry[1] for entry in group if entry is not None]) for group in groups],
        dtype=object,
    )
    return matched_files, matched_dates, observation_times


def observation_time(channel_times: Sequence[datetime | None]) -> datetime:
    """Mean UTC-naive time of the channels present in one observation."""
    present = [_utc_naive(value) for value in channel_times if value is not None]
    if not present:
        raise ValueError('An observation needs at least one channel time.')
    origin = present[0]
    offset = float(np.mean([(value - origin).total_seconds() for value in present]))
    return origin + timedelta(seconds=offset)



def normalize_image_scaling(
        scaling: float | Sequence[float] | Mapping[Any, float], channel_ids: Sequence[Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize image divisors and return serializable metadata."""
    channels = tuple(channel_ids)
    canonical_channels = tuple(canonical_channel_id(channel) for channel in channels)
    if len(set(canonical_channels)) != len(canonical_channels):
        raise ValueError(f'Image scaling channel_ids contain aliases or duplicates: {channels!r}.')
    metadata = None
    if (
        isinstance(scaling, Mapping)
        and scaling.get('schema') == 'sunerf.image_scaling.v1'
    ):
        scaling_channels = tuple(
            canonical_channel_id(channel)
            for channel in scaling.get('channel_ids', ())
        )
        if (
            scaling.get('operation') != 'divide'
            or scaling.get('inverse_operation') != 'multiply'
            or scaling_channels != canonical_channels
        ):
            raise ValueError('Stored image scaling metadata is inconsistent with channel_ids.')
        divisors = np.asarray(scaling.get('divisor', ()), dtype=np.float32)
        metadata = dict(scaling)
    elif isinstance(scaling, Mapping):
        try:
            canonical_scaling = canonical_channel_mapping(scaling)
        except ValueError as error:
            raise ValueError(f'Image scaling {error}.') from error
        if set(canonical_scaling) != set(canonical_channels):
            raise ValueError(
                'Image scaling mapping keys must match channel_ids exactly after '
                f'canonicalization; expected {canonical_channels!r}, got '
                f'{tuple(canonical_scaling)!r}.'
            )
        divisors = np.array(
            [canonical_scaling[channel] for channel in canonical_channels],
            dtype=np.float32,
        )
    elif np.ndim(scaling) == 0:
        divisors = np.full(len(channels), float(scaling), dtype=np.float32)
    else:
        divisors = np.asarray(scaling, dtype=np.float32)
        if divisors.shape != (len(channels),):
            raise ValueError(
                f'Image scaling must have {len(channels)} entries, got {divisors.shape}.'
            )
    if divisors.shape != (len(channels),):
        raise ValueError(
            f'Image scaling must have {len(channels)} entries, got {divisors.shape}.'
        )
    if not np.all(np.isfinite(divisors)) or np.any(divisors <= 0):
        raise ValueError('Every image scaling divisor must be finite and positive.')
    if metadata is None:
        metadata = {
            'schema': 'sunerf.image_scaling.v1',
            'operation': 'divide',
            'channel_ids': [str(channel) for channel in channels],
            'divisor': divisors.astype(float).tolist(),
            'inverse_operation': 'multiply',
        }
    return divisors, metadata


def observation_scaling_statistics(
        image: np.ndarray, valid_mask: np.ndarray, percentile: float = 99.5
) -> np.ndarray:
    """Percentile of absolute valid pixels per channel of one (y, x, channel) image.

    Channels without any valid finite pixel yield NaN.
    """
    image = np.asarray(image)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if image.ndim != 3 or valid_mask.shape != image.shape:
        raise ValueError(
            'Image-scaling statistics expect matching (y, x, channel) image and valid_mask.'
        )
    percentile = float(percentile)
    if not np.isfinite(percentile) or not 0 < percentile <= 100:
        raise ValueError('Image-scaling percentile must lie in (0, 100].')
    statistics = np.full(image.shape[-1], np.nan, dtype=np.float64)
    for channel_index in range(image.shape[-1]):
        selected = valid_mask[..., channel_index] & np.isfinite(image[..., channel_index])
        if np.any(selected):
            values = np.abs(image[..., channel_index][selected])
            statistics[channel_index] = np.percentile(values, percentile)
    return statistics


def reduce_scaling_statistics(
        statistics: np.ndarray,
        channel_ids: Sequence[Any],
        *,
        percentile: float = 99.5,
        temporal_reduction: str = 'median',
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reduce (observation, channel) statistics to one divisor per channel.

    The temporal median prevents one flare or corrupt exposure from setting the
    conditioning scale for the whole reconstruction.
    """
    channels = tuple(channel_ids)
    statistics = np.asarray(statistics, dtype=np.float64)
    if statistics.ndim != 2 or statistics.shape[-1] != len(channels):
        raise ValueError(
            'Image-scaling reduction expects (observation, channel) statistics.'
        )
    if temporal_reduction != 'median':
        raise ValueError("Image-scaling temporal_reduction must be 'median'.")

    divisors = []
    observation_counts = []
    for channel_index, channel in enumerate(channels):
        channel_statistics = statistics[:, channel_index]
        channel_statistics = channel_statistics[np.isfinite(channel_statistics)]
        if channel_statistics.size == 0:
            raise ValueError(
                f'Cannot estimate image scaling for channel {channel!r}: no valid pixels.'
            )
        divisor = float(np.median(channel_statistics))
        if not np.isfinite(divisor) or divisor <= 0:
            raise ValueError(
                f'Estimated image scaling for channel {channel!r} is not positive.'
            )
        divisors.append(divisor)
        observation_counts.append(int(channel_statistics.size))

    divisors, metadata = normalize_image_scaling(divisors, channels)
    metadata['estimator'] = {
        'strategy': 'robust_percentile',
        'percentile': float(percentile),
        'pixel_statistic': 'absolute_valid_magnitude',
        'temporal_reduction': temporal_reduction,
        'observation_count_by_channel': observation_counts,
    }
    return divisors, metadata


def estimate_image_scaling(
        images: np.ndarray,
        valid_mask: np.ndarray,
        channel_ids: Sequence[Any],
        *,
        percentile: float = 99.5,
        temporal_reduction: str = 'median',
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one robust magnitude divisor per channel of an image stack.

    Each observation contributes its percentile of absolute valid pixels; see
    ``reduce_scaling_statistics`` for the temporal reduction.
    """
    images = np.asarray(images)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    channels = tuple(channel_ids)
    if images.ndim != 4 or images.shape[-1] != len(channels):
        raise ValueError(
            'Image-scaling estimation expects (observation, y, x, channel) images.'
        )
    if valid_mask.shape != images.shape:
        raise ValueError('Image-scaling valid_mask must exactly match the image stack.')
    statistics = np.stack([
        observation_scaling_statistics(image, mask, percentile)
        for image, mask in zip(images, valid_mask)
    ]) if len(images) else np.empty((0, len(channels)))
    return reduce_scaling_statistics(
        statistics, channels, percentile=percentile,
        temporal_reduction=temporal_reduction,
    )


def _map_unit(s_map) -> str:
    unit = getattr(s_map, 'unit', None)
    if unit is not None:
        return str(unit)
    return str(s_map.meta.get('BUNIT', 'unknown'))


def _pixel_solid_angle(s_map) -> float:
    scale = s_map.scale
    return float(abs(
        scale[0].to_value(u.rad / u.pix) * scale[1].to_value(u.rad / u.pix)
    ))


def _sample_hpc_coordinates(s_map):
    ny, nx = s_map.data.shape
    x = np.array([0, nx - 1, 0, nx - 1, 0.5 * (nx - 1)]) * u.pix
    y = np.array([0, 0, ny - 1, ny - 1, 0.5 * (ny - 1)]) * u.pix
    coordinates = s_map.pixel_to_world(x, y)
    return np.stack([
        coordinates.Tx.to_value(u.arcsec), coordinates.Ty.to_value(u.arcsec)
    ], axis=-1)


def validate_map_alignment(
        reference_map, candidate_map, *, tolerance_arcsec=0.25,
        observer_angular_tolerance_deg=0.1, observer_distance_tolerance_au=0.01,
):
    """Validate pixel grids and observer identity before sharing one ray bundle.

    The default observer tolerance is intentionally generous for a roughly
    two-minute same-spacecraft sequence (0.1 degree and 0.01 AU), while still
    rejecting distinct heliospheric viewpoints such as STEREO-A and STEREO-B.
    """
    if candidate_map.data.shape != reference_map.data.shape:
        raise ValueError(
            f'Prepared EUV channel shape {candidate_map.data.shape} does not match '
            f'reference shape {reference_map.data.shape}.'
        )
    reference_ctype = tuple(str(value).upper() for value in reference_map.wcs.wcs.ctype)
    candidate_ctype = tuple(str(value).upper() for value in candidate_map.wcs.wcs.ctype)
    if candidate_ctype != reference_ctype:
        raise ValueError(
            f'Prepared EUV WCS axis order {candidate_ctype!r} does not match '
            f'{reference_ctype!r}.'
        )
    reference_coordinates = _sample_hpc_coordinates(reference_map)
    candidate_coordinates = _sample_hpc_coordinates(candidate_map)
    if not np.allclose(
        candidate_coordinates, reference_coordinates, rtol=0.0,
        atol=float(tolerance_arcsec), equal_nan=False,
    ):
        max_offset = float(np.max(np.abs(candidate_coordinates - reference_coordinates)))
        raise ValueError(
            f'Prepared EUV channels are not co-registered: sampled WCS offset '
            f'{max_offset:.6g} arcsec exceeds {tolerance_arcsec} arcsec.'
        )
    reference_observer = reference_map.observer_coordinate
    candidate_observer = candidate_map.observer_coordinate
    reference_lon = reference_observer.lon.to_value(u.deg)
    candidate_lon = candidate_observer.lon.to_value(u.deg)
    lon_delta = (candidate_lon - reference_lon + 180.0) % 360.0 - 180.0
    lat_delta = (
        candidate_observer.lat - reference_observer.lat
    ).to_value(u.deg)
    mean_latitude = 0.5 * (
        candidate_observer.lat + reference_observer.lat
    ).to_value(u.rad)
    angular_offset = float(np.hypot(lon_delta * np.cos(mean_latitude), lat_delta))
    reference_distance = reference_observer.radius.to_value(u.AU)
    candidate_distance = candidate_observer.radius.to_value(u.AU)
    distance_offset = float(abs(candidate_distance - reference_distance))
    if (
        angular_offset > float(observer_angular_tolerance_deg)
        or distance_offset > float(observer_distance_tolerance_au)
    ):
        raise ValueError(
            'Prepared EUV channels have incompatible observer viewpoints: '
            f'angular offset {angular_offset:.6g} deg and distance offset '
            f'{distance_offset:.6g} AU exceed tolerances '
            f'({observer_angular_tolerance_deg} deg, '
            f'{observer_distance_tolerance_au} AU). Cross-spacecraft channels '
            'cannot share one ray construction.'
        )


@dataclass(frozen=True)
class PreparedEUVObservation:
    """Validated, channel-first EUV observation consumed by SuNeRF.

    Images remain in their calibrated instrument-native units. Machine-learning
    normalization is represented separately by ``image_scaling`` at dataset
    construction time.

    A channel without a prepared file at this time step is still part of the
    observation: its image is NaN, its valid mask is false and its per-channel
    metadata (``wcs``, ``channel_times``, units, ...) is ``None``.
    """

    instrument_id: str
    channel_ids: tuple[str, ...]
    image: np.ndarray
    valid_mask: np.ndarray
    wcs: tuple[Any, ...]
    channel_times: tuple[datetime, ...]
    measurement_units: tuple[str, ...]
    pixel_solid_angle_sr: tuple[float, ...]
    native_pixel_solid_angle_sr: tuple[float, ...]
    measurement_semantics: tuple[str | None, ...]
    source_paths: tuple[str, ...]
    sensitivity_conventions: tuple[str | None, ...]
    variance: np.ndarray | None = None
    calibration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        image = np.asarray(self.image, dtype=np.float32)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        if image.ndim != 3:
            raise ValueError(f'image must have shape (channel, y, x), got {image.shape}.')
        if valid_mask.shape != image.shape:
            raise ValueError('valid_mask must have the same shape as image.')
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError('channel_ids must be unique and ordered.')
        for name in (
            'channel_ids', 'wcs', 'channel_times', 'measurement_units',
            'pixel_solid_angle_sr', 'native_pixel_solid_angle_sr',
            'measurement_semantics', 'source_paths', 'sensitivity_conventions',
        ):
            if len(getattr(self, name)) != image.shape[0]:
                raise ValueError(f'{name} must contain one value per image channel.')
        if self.variance is not None:
            variance = np.asarray(self.variance, dtype=np.float32)
            if variance.shape != image.shape:
                raise ValueError('variance must have the same shape as image.')
            if np.any(variance[valid_mask] < 0) or not np.all(np.isfinite(variance[valid_mask])):
                raise ValueError('Variance must be finite and non-negative at valid pixels.')
            object.__setattr__(self, 'variance', variance)
        if not np.all(np.isfinite(image[valid_mask])):
            raise ValueError('Valid EUV pixels must contain finite measurements.')
        if any(
            value is not None and (not np.isfinite(value) or value <= 0)
            for values in (
                self.pixel_solid_angle_sr,
                self.native_pixel_solid_angle_sr,
            )
            for value in values
        ):
            raise ValueError('Pixel solid angles must be finite and positive per channel.')
        if all(value is None for value in self.channel_times):
            raise ValueError('An observation needs at least one present channel.')
        if any(
            (value is None) != (time is None)
            for values in (
                self.wcs, self.measurement_units, self.pixel_solid_angle_sr,
                self.native_pixel_solid_angle_sr, self.source_paths,
            )
            for value, time in zip(values, self.channel_times)
        ):
            raise ValueError('Per-channel metadata must be None exactly for missing channels.')
        if np.any(valid_mask[[time is None for time in self.channel_times]]):
            raise ValueError('Missing channels cannot have valid pixels.')
        object.__setattr__(self, 'image', image)
        object.__setattr__(self, 'valid_mask', valid_mask)
        object.__setattr__(
            self, 'channel_times',
            tuple(None if value is None else _utc_naive(value) for value in self.channel_times),
        )

    @property
    def channel_present(self) -> tuple[bool, ...]:
        return tuple(value is not None for value in self.channel_times)

    @property
    def obstime(self) -> datetime:
        return observation_time(self.channel_times)

    @property
    def time_offsets_seconds(self) -> tuple[float | None, ...]:
        center = self.obstime
        return tuple(
            None if value is None else (value - center).total_seconds()
            for value in self.channel_times
        )

    def metadata(self) -> dict[str, Any]:
        return {
            'schema': 'sunerf.prepared_euv_observation.v1',
            'instrument_id': self.instrument_id,
            'channel_ids': list(self.channel_ids),
            'channel_present': list(self.channel_present),
            'channel_times': [
                None if value is None else value.isoformat() for value in self.channel_times
            ],
            'time_offsets_seconds': list(self.time_offsets_seconds),
            'measurement_units': list(self.measurement_units),
            'pixel_solid_angle_sr': list(self.pixel_solid_angle_sr),
            'native_pixel_solid_angle_sr': list(self.native_pixel_solid_angle_sr),
            'measurement_semantics': list(self.measurement_semantics),
            'source_paths': list(self.source_paths),
            'sensitivity_conventions': list(self.sensitivity_conventions),
            'calibration': dict(self.calibration),
        }


class PreparedEUVAdapter:
    """Small instrument adapter for already-calibrated FITS products."""

    instrument_id = 'EUV'

    def validate_map(self, s_map, *, strict_metadata=False):
        exposure = s_map.meta.get('EXPTIME', s_map.meta.get('XPOSURE'))
        if strict_metadata and exposure is None:
            raise ValueError(f'{self.instrument_id} prepared image is missing exposure metadata.')
        if exposure is not None and (not np.isfinite(float(exposure)) or float(exposure) <= 0):
            raise ValueError(f'{self.instrument_id} exposure time must be finite and positive.')
        if strict_metadata and _map_unit(s_map) == 'unknown':
            raise ValueError(f'{self.instrument_id} prepared image is missing BUNIT.')
        sensitivity_convention = str(s_map.meta.get('SENSCON', '')).strip()
        if strict_metadata and sensitivity_convention not in SENSITIVITY_CONVENTIONS:
            raise ValueError(
                f'{self.instrument_id} prepared image must declare SENSCON as '
                f'one of {sorted(SENSITIVITY_CONVENTIONS)}.'
            )
        if strict_metadata and s_map.meta.get('PREPSCHM') != PREPARED_EUV_SCHEMA:
            raise ValueError(
                f'{self.instrument_id} prepared image has unsupported PREPSCHM '
                f'{s_map.meta.get("PREPSCHM")!r}; expected {PREPARED_EUV_SCHEMA!r}.'
            )
        measurement_semantics = str(s_map.meta.get('RADSEM', '')).strip()
        if strict_metadata and measurement_semantics not in {
            'surface_brightness', 'per_native_pixel'
        }:
            raise ValueError(
                f'{self.instrument_id} prepared image must declare RADSEM as '
                'surface_brightness or per_native_pixel.'
            )
        native_solid_angle = s_map.meta.get('NATPXSR')
        if strict_metadata and (
            native_solid_angle is None
            or not np.isfinite(float(native_solid_angle))
            or float(native_solid_angle) <= 0
        ):
            raise ValueError(
                f'{self.instrument_id} prepared image must declare a finite positive NATPXSR.'
            )

    def prepare(
            self,
            maps: Sequence[Any],
            channel_ids: Sequence[Any],
            *,
            instrument_id: str | None = None,
            source_paths: Sequence[str] | None = None,
            alignment_tolerance_arcsec: float = 0.25,
            strict_metadata: bool = False,
            valid_masks: Sequence[np.ndarray] | None = None,
    ) -> PreparedEUVObservation:
        """Validate the channel maps and assemble one observation.

        ``valid_masks`` are the VALID_MASK extensions already read together
        with the maps (``load_prepared_observation``); without them every
        source file is opened again for its mask. A ``None`` map is a channel
        without a prepared file at this time step; it becomes a NaN frame with
        an all-false valid mask. At least one channel must be present.

        Pixels that are not finite or not positive are invalid: they are NaN
        in the returned image and false in the valid mask. Which pixels of an
        observation are supervised is decided per channel downstream, so a
        pixel stays in training as long as one channel is valid there.
        """
        if len(maps) != len(channel_ids) or not maps:
            raise ValueError('maps and channel_ids must have the same non-zero length.')
        if valid_masks is not None and len(valid_masks) != len(maps):
            raise ValueError('valid_masks must have one entry per channel.')
        prepared_maps = [
            None if value is None else (value if hasattr(value, 'wcs') else load_prepared_map(value))
            for value in maps
        ]
        if all(s_map is None for s_map in prepared_maps):
            raise ValueError('An observation needs at least one present channel.')
        if source_paths is None:
            source_paths = ['unknown'] * len(prepared_maps)
        if len(source_paths) != len(prepared_maps):
            raise ValueError('source_paths must have one entry per channel.')
        source_paths = tuple(
            None if s_map is None else str(value)
            for value, s_map in zip(source_paths, prepared_maps)
        )

        reference_map = next(s_map for s_map in prepared_maps if s_map is not None)
        for channel, s_map in zip(channel_ids, prepared_maps):
            if s_map is None:
                continue
            self.validate_map(s_map, strict_metadata=strict_metadata)
            actual_channel = _channel_number(s_map.wavelength)
            expected_channel = _channel_number(channel)
            if actual_channel != expected_channel:
                raise ValueError(
                    f'Channel order mismatch: expected {channel!r}, FITS map reports '
                    f'{actual_channel!r}.'
                )
            validate_map_alignment(
                reference_map, s_map, tolerance_arcsec=alignment_tolerance_arcsec
            )

        shape = reference_map.data.shape
        image = np.stack([
            np.full(shape, np.nan, dtype=np.float32) if s_map is None
            else np.asarray(s_map.data, dtype=np.float32)
            for s_map in prepared_maps
        ], axis=0)
        # Non-positive EUV values carry no emission information for the
        # optically thin forward model (zero or negative after calibration,
        # saturation bleed, or fill values) and are treated as invalid.
        with np.errstate(invalid='ignore'):
            valid_mask = np.isfinite(image) & (image > 0)
        for channel_index, (s_map, source_path) in enumerate(
                zip(prepared_maps, source_paths)
        ):
            if s_map is None:
                continue
            if valid_masks is not None:
                file_mask = np.asarray(valid_masks[channel_index], dtype=bool)
                if file_mask.shape != s_map.data.shape:
                    raise ValueError(
                        f'VALID_MASK shape {file_mask.shape} does not match image shape '
                        f'{s_map.data.shape}.'
                    )
            else:
                file_mask = _fits_valid_mask(
                    source_path, s_map.data.shape, required=strict_metadata
                )
            valid_mask[channel_index] &= file_mask

        image = np.where(valid_mask, image, np.nan)

        def per_channel(function):
            return tuple(None if s_map is None else function(s_map) for s_map in prepared_maps)

        return PreparedEUVObservation(
            instrument_id=self.instrument_id if instrument_id is None else str(instrument_id),
            channel_ids=tuple(str(channel) for channel in channel_ids),
            image=image,
            valid_mask=valid_mask,
            wcs=per_channel(lambda s_map: s_map.wcs),
            channel_times=per_channel(lambda s_map: _utc_naive(s_map.date)),
            measurement_units=per_channel(_map_unit),
            pixel_solid_angle_sr=per_channel(_pixel_solid_angle),
            native_pixel_solid_angle_sr=per_channel(
                lambda s_map: float(s_map.meta.get('NATPXSR', _pixel_solid_angle(s_map)))
            ),
            measurement_semantics=per_channel(lambda s_map: s_map.meta.get('RADSEM')),
            source_paths=source_paths,
            sensitivity_conventions=per_channel(lambda s_map: s_map.meta.get('SENSCON')),
            calibration={
                'prepared_contract': 'sunerf.prepared_euv_observation.v1',
                'strict_metadata': bool(strict_metadata),
            },
        )


class AIAPreparedAdapter(PreparedEUVAdapter):
    instrument_id = 'AIA'

    def validate_map(self, s_map, *, strict_metadata=False):
        super().validate_map(s_map, strict_metadata=strict_metadata)
        quality = s_map.meta.get('QUALITY')
        if strict_metadata and quality is None:
            raise ValueError('AIA prepared image is missing QUALITY metadata.')
        if quality is not None and int(quality) != 0:
            raise ValueError(f'AIA QUALITY must be zero, got {quality!r}.')


class EUVIPreparedAdapter(PreparedEUVAdapter):
    instrument_id = 'EUVI'


class EUIPreparedAdapter(PreparedEUVAdapter):
    instrument_id = 'EUI'

    def validate_map(self, s_map, *, strict_metadata=False):
        super().validate_map(s_map, strict_metadata=strict_metadata)
        image_type = str(s_map.meta.get('IMGTYPE', '')).lower()
        if image_type and 'image' not in image_type:
            raise ValueError(f'EUI IMGTYPE does not describe an image: {image_type!r}.')


class GenericPreparedEUVAdapter(PreparedEUVAdapter):
    """Compatibility adapter for simulation products such as PSI."""

    instrument_id = 'SYNTHETIC-EUV'


_ADAPTERS = {
    'AIA': AIAPreparedAdapter,
    'EUVI': EUVIPreparedAdapter,
    'EUI': EUIPreparedAdapter,
    'PSI': GenericPreparedEUVAdapter,
}


def get_prepared_euv_adapter(instrument: str) -> PreparedEUVAdapter:
    try:
        return _ADAPTERS[instrument.upper()]()
    except KeyError as exc:
        raise ValueError(f'Unknown prepared EUV instrument {instrument!r}.') from exc


def prepare_aia_observation(maps, channel_ids, **kwargs):
    """Validate calibrated AIA rate maps and return the common contract."""
    return AIAPreparedAdapter().prepare(maps, channel_ids, **kwargs)


def prepare_euvi_observation(maps, channel_ids, **kwargs):
    """Validate documented SECCHI-prepped EUVI maps and return the contract."""
    return EUVIPreparedAdapter().prepare(maps, channel_ids, **kwargs)


def prepare_eui_observation(maps, channel_ids, **kwargs):
    """Validate calibrated EUI (preferably L2) maps and return the contract."""
    return EUIPreparedAdapter().prepare(maps, channel_ids, **kwargs)
