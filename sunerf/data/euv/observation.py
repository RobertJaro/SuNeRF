"""Common contract for calibrated EUV observations.

The routines in this module deliberately do not perform instrument calibration.
They are the boundary between instrument-specific preparation and SuNeRF: a
prepared FITS product is validated, assigned stable channel metadata, and
converted to one consistent in-memory layout before rays are constructed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sunpy.map import Map

from sunerf.configuration import canonical_channel_id, canonical_channel_mapping
from sunerf.response import SENSITIVITY_CONVENTIONS


PREPARED_EUV_SCHEMA = 'sunerf.prepared_euv.v2'


def load_prepared_map(path: str | Path):
    """Load the primary prepared image, never the VALID_MASK image extension."""
    with fits.open(path, memmap=False) as hdul:
        data = np.array(hdul[0].data, copy=True)
        header = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f'Prepared EUV primary HDU must be a 2D image: {path}.')
    return Map(data, header)


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


def discover_prepared_files(
        files: Sequence[str | Path], channel_ids: Sequence[int | str]
) -> tuple[dict[int | str, list[str]], dict[int | str, list[datetime]]]:
    """Group prepared FITS paths by requested channel using FITS metadata.

    File-name parsing was historically different for every instrument and could
    silently put a file into the wrong channel. The FITS channel and time are the
    authoritative values at this boundary.
    """
    requested = tuple(channel_ids)
    requested_by_number = {_channel_number(channel): channel for channel in requested}
    if len(requested_by_number) != len(requested):
        raise ValueError(f'Channel identifiers are not unique: {requested!r}.')

    file_dict = {channel: [] for channel in requested}
    date_dict = {channel: [] for channel in requested}
    for path_value in files:
        path = str(path_value)
        header = fits.getheader(path)
        channel = requested_by_number.get(_header_channel(header))
        if channel is None:
            continue
        file_dict[channel].append(path)
        date_dict[channel].append(_header_time(header))

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


def _is_better(candidate_count, candidate_cost, count, cost):
    return candidate_count > count or (
        candidate_count == count and candidate_cost < cost - 1e-12
    )


def _optimal_unique_time_matches(reference_dates, candidate_dates, tolerance_seconds):
    """Maximum-cardinality, minimum-offset ordered one-to-one time matching."""
    reference_seconds = np.array([
        (value - reference_dates[0]).total_seconds() for value in reference_dates
    ], dtype=np.float64)
    candidate_seconds = np.array([
        (value - reference_dates[0]).total_seconds() for value in candidate_dates
    ], dtype=np.float64)
    n_reference = len(reference_seconds)
    n_candidate = len(candidate_seconds)

    # The exact dynamic program is intentionally bounded. Very large archives
    # should be cadence-filtered before ingestion; the linear fallback still
    # guarantees unique, chronological, tolerance-respecting matches.
    if n_reference * n_candidate > 4_000_000:
        matches = {}
        candidate_index = 0
        for reference_index, reference_time in enumerate(reference_seconds):
            while (
                candidate_index < n_candidate
                and candidate_seconds[candidate_index] < reference_time - tolerance_seconds
            ):
                candidate_index += 1
            if (
                candidate_index < n_candidate
                and candidate_seconds[candidate_index] <= reference_time + tolerance_seconds
            ):
                matches[reference_index] = candidate_index
                candidate_index += 1
        return matches

    counts = np.zeros((n_reference + 1, n_candidate + 1), dtype=np.int32)
    costs = np.zeros((n_reference + 1, n_candidate + 1), dtype=np.float64)
    directions = np.zeros((n_reference + 1, n_candidate + 1), dtype=np.uint8)
    # directions: 1=skip reference, 2=skip candidate, 3=match
    for reference_index in range(1, n_reference + 1):
        for candidate_index in range(1, n_candidate + 1):
            count = counts[reference_index - 1, candidate_index]
            cost = costs[reference_index - 1, candidate_index]
            direction = 1

            left_count = counts[reference_index, candidate_index - 1]
            left_cost = costs[reference_index, candidate_index - 1]
            if _is_better(left_count, left_cost, count, cost):
                count, cost, direction = left_count, left_cost, 2

            offset = abs(
                reference_seconds[reference_index - 1]
                - candidate_seconds[candidate_index - 1]
            )
            if offset <= tolerance_seconds:
                match_count = counts[reference_index - 1, candidate_index - 1] + 1
                match_cost = costs[reference_index - 1, candidate_index - 1] + offset
                if _is_better(match_count, match_cost, count, cost):
                    count, cost, direction = match_count, match_cost, 3

            counts[reference_index, candidate_index] = count
            costs[reference_index, candidate_index] = cost
            directions[reference_index, candidate_index] = direction

    matches = {}
    reference_index, candidate_index = n_reference, n_candidate
    while reference_index and candidate_index:
        direction = directions[reference_index, candidate_index]
        if direction == 3:
            matches[reference_index - 1] = candidate_index - 1
            reference_index -= 1
            candidate_index -= 1
        elif direction == 2:
            candidate_index -= 1
        else:
            reference_index -= 1
    return matches


def _global_unique_time_matches(reference_dates, candidate_dates, tolerance_seconds):
    """Select complete tuples with maximum cardinality and minimum total offset.

    Each non-reference channel receives its own one-to-one assignment, while a
    shared binary variable decides whether a reference row is retained.  This
    avoids the order-dependent data loss caused by sequential pairwise
    matching.  The sparse mixed-integer problem is only needed when at least
    one channel cannot match every reference row; the common complete-cadence
    case stays on the faster ordered dynamic program.
    """
    channels = tuple(candidate_dates)
    n_reference = len(reference_dates)
    full_matches = {
        channel: _optimal_unique_time_matches(
            reference_dates, candidate_dates[channel], tolerance_seconds
        )
        for channel in channels
    }
    if all(len(matches) == n_reference for matches in full_matches.values()):
        return np.arange(n_reference, dtype=np.int64), full_matches

    origin = reference_dates[0]
    reference_seconds = np.asarray(
        [(value - origin).total_seconds() for value in reference_dates],
        dtype=np.float64,
    )
    candidate_seconds = {
        channel: np.asarray(
            [(value - origin).total_seconds() for value in candidate_dates[channel]],
            dtype=np.float64,
        )
        for channel in channels
    }

    # Variables are the retained-reference indicators followed by all eligible
    # channel/reference/candidate edges.  Scaling the total edge cost below one
    # makes cardinality the strict primary objective without a second solve.
    edges = []
    denominator = max(float(tolerance_seconds), 1.0) * max(
        (n_reference + 1) * len(channels), 1
    )
    for channel_index, channel in enumerate(channels):
        for reference_index, reference_time in enumerate(reference_seconds):
            candidate_indices = np.flatnonzero(
                np.abs(candidate_seconds[channel] - reference_time)
                <= tolerance_seconds
            )
            for candidate_index in candidate_indices:
                offset = abs(
                    reference_time - candidate_seconds[channel][candidate_index]
                )
                edges.append(
                    (
                        channel_index,
                        reference_index,
                        int(candidate_index),
                        float(offset),
                    )
                )

    n_variables = n_reference + len(edges)
    objective = np.zeros(n_variables, dtype=np.float64)
    objective[:n_reference] = -1.0
    for edge_index, (*_, offset) in enumerate(edges, start=n_reference):
        objective[edge_index] = offset / denominator

    equality_rows = len(channels) * n_reference
    candidate_row_offsets = {}
    row_count = equality_rows
    for channel_index, channel in enumerate(channels):
        for candidate_index in range(len(candidate_seconds[channel])):
            candidate_row_offsets[(channel_index, candidate_index)] = row_count
            row_count += 1

    rows = []
    columns = []
    values = []
    for channel_index in range(len(channels)):
        for reference_index in range(n_reference):
            rows.append(channel_index * n_reference + reference_index)
            columns.append(reference_index)
            values.append(-1.0)
    for variable_index, (channel_index, reference_index, candidate_index, _) in enumerate(
        edges, start=n_reference
    ):
        rows.extend(
            [
                channel_index * n_reference + reference_index,
                candidate_row_offsets[(channel_index, candidate_index)],
            ]
        )
        columns.extend([variable_index, variable_index])
        values.extend([1.0, 1.0])

    matrix = coo_matrix(
        (values, (rows, columns)), shape=(row_count, n_variables), dtype=np.float64
    ).tocsr()
    lower = np.concatenate(
        [np.zeros(equality_rows), np.full(row_count - equality_rows, -np.inf)]
    )
    upper = np.concatenate(
        [np.zeros(equality_rows), np.ones(row_count - equality_rows)]
    )
    result = milp(
        objective,
        integrality=np.ones(n_variables, dtype=np.uint8),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=LinearConstraint(matrix, lower, upper),
        options={'presolve': True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f'Global EUV time matching failed: {result.message}')

    selected = np.flatnonzero(result.x[:n_reference] > 0.5).astype(np.int64)
    assignments = {channel: {} for channel in channels}
    for variable_index, (channel_index, reference_index, candidate_index, _) in enumerate(
        edges, start=n_reference
    ):
        if result.x[variable_index] > 0.5:
            assignments[channels[channel_index]][reference_index] = candidate_index
    if any(set(matches) != set(selected) for matches in assignments.values()):
        raise RuntimeError('Global EUV time matching returned an incomplete assignment.')
    return selected, assignments


def match_prepared_channels(
        file_dict: Mapping[Any, Sequence[str]],
        date_dict: Mapping[Any, Sequence[datetime]],
        *,
        tolerance: timedelta = timedelta(minutes=2),
        channel_order: Sequence[Any] | None = None,
) -> tuple[dict[Any, np.ndarray], dict[Any, np.ndarray], Any]:
    """Create complete, unique multi-channel observations.

    The least-populated channel is used as the reference. Every input file can
    appear in at most one output observation, and incomplete reference times are
    removed. Inputs are copied and never reordered or mutated in place.
    """
    channels = tuple(file_dict) if channel_order is None else tuple(channel_order)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError('channel_order must contain at least one unique channel.')
    if set(channels) != set(file_dict) or set(channels) != set(date_dict):
        raise ValueError('file_dict, date_dict, and channel_order must contain identical channels.')

    sorted_files = {}
    sorted_dates = {}
    for channel in channels:
        paths = np.asarray(file_dict[channel], dtype=object)
        dates = np.asarray([_utc_naive(value) for value in date_dict[channel]], dtype=object)
        if len(paths) != len(dates):
            raise ValueError(f'Channel {channel!r} has different file and date counts.')
        if len(paths) == 0:
            raise ValueError(f'Channel {channel!r} has no observations.')
        if len(set(map(str, paths))) != len(paths):
            raise ValueError(f'Channel {channel!r} contains duplicate source paths.')
        order = np.argsort(dates, kind='stable')
        sorted_files[channel] = paths[order]
        sorted_dates[channel] = dates[order]

    reference_channel = min(channels, key=lambda channel: len(sorted_dates[channel]))
    remaining_channels = tuple(
        channel for channel in channels if channel != reference_channel
    )
    tolerance_seconds = float(tolerance.total_seconds())
    if tolerance_seconds < 0:
        raise ValueError('Matching tolerance must be non-negative.')
    active_reference_indices, candidate_assignments = _global_unique_time_matches(
        sorted_dates[reference_channel],
        {channel: sorted_dates[channel] for channel in remaining_channels},
        tolerance_seconds,
    )
    if not len(active_reference_indices):
        raise ValueError(f'No complete observations remain within {tolerance}.')
    assignments = {reference_channel: active_reference_indices}
    assignments.update({
        channel: np.asarray(
            [matches[int(index)] for index in active_reference_indices],
            dtype=np.int64,
        )
        for channel, matches in candidate_assignments.items()
    })

    matched_files = {
        channel: sorted_files[channel][assignments[channel]].copy()
        for channel in channels
    }
    matched_dates = {
        channel: sorted_dates[channel][assignments[channel]].copy()
        for channel in channels
    }
    for channel in channels:
        if len(set(map(str, matched_files[channel]))) != len(matched_files[channel]):
            raise RuntimeError(f'Internal error: channel {channel!r} reused an input file.')
    return matched_files, matched_dates, reference_channel


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
            not np.isfinite(value) or value <= 0
            for values in (
                self.pixel_solid_angle_sr,
                self.native_pixel_solid_angle_sr,
            )
            for value in values
        ):
            raise ValueError('Pixel solid angles must be finite and positive per channel.')
        object.__setattr__(self, 'image', image)
        object.__setattr__(self, 'valid_mask', valid_mask)
        object.__setattr__(
            self, 'channel_times', tuple(_utc_naive(value) for value in self.channel_times)
        )

    @property
    def obstime(self) -> datetime:
        timestamps = np.array([
            value.replace(tzinfo=timezone.utc).timestamp() for value in self.channel_times
        ])
        return datetime.fromtimestamp(float(np.mean(timestamps)), tz=timezone.utc).replace(tzinfo=None)

    @property
    def time_offsets_seconds(self) -> tuple[float, ...]:
        center = self.obstime
        return tuple((value - center).total_seconds() for value in self.channel_times)

    def metadata(self) -> dict[str, Any]:
        return {
            'schema': 'sunerf.prepared_euv_observation.v1',
            'instrument_id': self.instrument_id,
            'channel_ids': list(self.channel_ids),
            'channel_times': [value.isoformat() for value in self.channel_times],
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
    ) -> PreparedEUVObservation:
        if len(maps) != len(channel_ids) or not maps:
            raise ValueError('maps and channel_ids must have the same non-zero length.')
        prepared_maps = [
            value if hasattr(value, 'wcs') else load_prepared_map(value)
            for value in maps
        ]
        source_paths = tuple(
            str(value) for value in (
                source_paths if source_paths is not None else ['unknown'] * len(prepared_maps)
            )
        )
        if len(source_paths) != len(prepared_maps):
            raise ValueError('source_paths must have one entry per channel.')

        reference_map = prepared_maps[0]
        for channel, s_map in zip(channel_ids, prepared_maps):
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

        image = np.stack([
            np.asarray(s_map.data, dtype=np.float32) for s_map in prepared_maps
        ], axis=0)
        valid_mask = np.isfinite(image)
        for channel_index, (s_map, source_path) in enumerate(
                zip(prepared_maps, source_paths)
        ):
            valid_mask[channel_index] &= _fits_valid_mask(
                source_path, s_map.data.shape, required=strict_metadata
            )
        return PreparedEUVObservation(
            instrument_id=self.instrument_id if instrument_id is None else str(instrument_id),
            channel_ids=tuple(str(channel) for channel in channel_ids),
            image=image,
            valid_mask=valid_mask,
            wcs=tuple(s_map.wcs for s_map in prepared_maps),
            channel_times=tuple(_utc_naive(s_map.date) for s_map in prepared_maps),
            measurement_units=tuple(_map_unit(s_map) for s_map in prepared_maps),
            pixel_solid_angle_sr=tuple(_pixel_solid_angle(s_map) for s_map in prepared_maps),
            native_pixel_solid_angle_sr=tuple(
                float(s_map.meta.get('NATPXSR', _pixel_solid_angle(s_map)))
                for s_map in prepared_maps
            ),
            measurement_semantics=tuple(
                s_map.meta.get('RADSEM') for s_map in prepared_maps
            ),
            source_paths=source_paths,
            sensitivity_conventions=tuple(
                s_map.meta.get('SENSCON') for s_map in prepared_maps
            ),
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
