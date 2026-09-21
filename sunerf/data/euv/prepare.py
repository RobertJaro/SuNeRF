"""Unified raw-to-prepared EUV data pipeline.

The three supported adapters have deliberately different calibration contracts:

* ``aia`` applies the documented aiapy sequence (updated pointing, registration,
  time-dependent degradation correction, exposure normalization).  Both aiapy
  input tables must be supplied as local files so a run never depends on a
  mutable network resource.
* ``euvi`` accepts only an externally calibrated SECCHI/EUVI product.  The
  external product and calibration identifiers are required and this module
  performs geometry operations only.
* ``eui`` accepts calibrated Solar Orbiter/EUI level-2 data, validates the level,
  and performs geometry operations only.

All adapters then use the same crop, reprojection, resampling, radial masking,
and FITS writer. Detector values are never clipped or normalized and invalid
pixels remain NaN with a matching ``VALID_MASK`` FITS extension.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time
from astropy.wcs import WCS
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header

from sunerf.data.euv.observation import PREPARED_EUV_SCHEMA

MAX_PREP_WORKERS = 8
SENSITIVITY_CONVENTIONS = frozenset(
    {'native_epoch', 'reference_epoch', 'static_assumed'}
)


@dataclass(frozen=True)
class GeometryConfig:
    """Instrument-independent geometry applied after radiometric calibration.

    ``shape`` is ``(ny, nx)``. ``hpc_bounds_arcsec`` is
    ``(xmin, ymin, xmax, ymax)``.  A reprojection reference fixes both the WCS
    and shape and therefore cannot be combined with ``shape`` or explicit HPC
    bounds. ``max_radius_rsun`` crops to a bounding square and masks the pixels
    outside the requested apparent solar radius.
    """

    shape: tuple[int, int] | None = None
    hpc_bounds_arcsec: tuple[float, float, float, float] | None = None
    max_radius_rsun: float | None = None
    reproject_reference: str | Path | None = None
    north_up: bool = True
    interpolation_order: int = 3

    def __post_init__(self):
        if self.shape is not None and (
            len(self.shape) != 2 or any(int(value) <= 0 for value in self.shape)
        ):
            raise ValueError('Geometry shape must be a positive (ny, nx) pair.')
        if self.hpc_bounds_arcsec is not None:
            if len(self.hpc_bounds_arcsec) != 4:
                raise ValueError('HPC bounds must be (xmin, ymin, xmax, ymax).')
            xmin, ymin, xmax, ymax = self.hpc_bounds_arcsec
            if xmin >= xmax or ymin >= ymax:
                raise ValueError('HPC bounds must have increasing x and y limits.')
            if self.shape is not None and any(int(value) < 2 for value in self.shape):
                raise ValueError(
                    'Exact HPC bounds require at least two pixels along each output axis.'
                )
        if self.max_radius_rsun is not None and self.max_radius_rsun <= 0:
            raise ValueError('max_radius_rsun must be positive.')
        if self.reproject_reference is not None and (
            self.shape is not None or self.hpc_bounds_arcsec is not None
        ):
            raise ValueError(
                'A reprojection reference defines the shape and bounds; do not '
                'also set shape or hpc_bounds_arcsec.'
            )
        if self.interpolation_order not in range(6):
            raise ValueError('interpolation_order must be between 0 and 5.')


@dataclass(frozen=True)
class PreparationResult:
    """One prepared FITS product."""

    output_path: Path


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _nonempty(value: Any, name: str) -> str:
    value = '' if value is None else str(value).strip()
    if not value:
        raise ValueError(f'{name} must be a non-empty identifier.')
    return value


def _map_channel(s_map) -> str:
    try:
        value = s_map.wavelength.to_value(u.AA)
    except Exception:
        value = s_map.meta.get('wavelnth') or s_map.meta.get('wavelength')
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError) as error:
        raise ValueError('The input FITS file has no numeric EUV wavelength.') from error


def _validated_sensitivity_convention(value, *, context):
    if value not in SENSITIVITY_CONVENTIONS:
        raise ValueError(
            f'{context} sensitivity_convention must be one of '
            f'{sorted(SENSITIVITY_CONVENTIONS)}, got {value!r}.'
        )
    return value


def _measurement_unit(s_map) -> str:
    value = s_map.meta.get('bunit')
    if value is None or not str(value).strip():
        raise ValueError('The calibrated input must declare a non-empty BUNIT.')
    value = str(value).strip()
    # SECCHI_PREP writes the historical, non-FITS token ``PhotonFlux`` after
    # converting detector DN to photons and normalizing to one second. Publish
    # the same quantity using an Astropy/FITS-compatible unit.
    if re.sub(r'[^a-z0-9]+', '', value.lower()) == 'photonflux':
        return 'ph / s'
    try:
        u.Unit(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'The calibrated input BUNIT {value!r} is not an Astropy-compatible unit.'
        ) from error
    return value


def _pixel_solid_angle_sr(s_map) -> float:
    return float(abs(
        s_map.scale[0].to_value(u.rad / u.pix)
        * s_map.scale[1].to_value(u.rad / u.pix)
    ))


def _geometry_radiometry_provenance(native_map, prepared_map):
    return {
        'native_pixel_solid_angle_sr': _pixel_solid_angle_sr(native_map),
        'prepared_pixel_solid_angle_sr': _pixel_solid_angle_sr(prepared_map),
        'geometry_resampling_semantics': (
            'interpolate_calibrated_surface_brightness_samples_without_'
            'solid_angle_conversion'
        ),
        'solid_angle_conversion_applied': False,
        'native_pixel_semantics_preserved': True,
    }


def _aia_rate_unit(input_unit: str) -> str:
    """Return a canonical per-second unit, rejecting double normalization."""
    try:
        parsed = u.Unit(input_unit)
    except (TypeError, ValueError) as error:
        raise ValueError(f'AIA BUNIT {input_unit!r} cannot be parsed by Astropy.') from error
    if any(base == u.s and power < 0 for base, power in zip(parsed.bases, parsed.powers)):
        raise ValueError(
            f'AIA level-1 BUNIT {input_unit!r} is already rate-normalized; '
            'refusing to divide by EXPTIME twice.'
        )
    return (parsed / u.s).to_string()


def _aia_level1_input_unit(s_map) -> str:
    """Validate an AIA level-1 detector image and resolve its count unit."""
    identity = ' '.join(
        str(s_map.meta.get(key, ''))
        for key in ('telescop', 'instrume', 'detector')
    ).lower()
    if 'aia' not in identity:
        raise ValueError('AIA preparation requires an input positively identified as AIA.')
    level = next(
        (
            s_map.meta.get(key)
            for key in ('lvl_num', 'level', 'data_lev', 'proclev')
            if s_map.meta.get(key) is not None
        ),
        None,
    )
    try:
        is_level_one = float(str(level).lower().removeprefix('l')) == 1.0
    except (TypeError, ValueError):
        is_level_one = False
    if not is_level_one:
        raise ValueError(f'AIA preparation requires level-1 input, got {level!r}.')
    unit = s_map.meta.get('bunit')
    # JSOC AIA level-1 detector products commonly omit BUNIT. Their data
    # numbers are defined as detector DN by the series contract.
    return 'DN' if unit is None or not str(unit).strip() else str(unit).strip()


def _exposure_seconds(s_map) -> float:
    value = s_map.meta.get('exptime')
    try:
        seconds = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('The input FITS file must declare a numeric EXPTIME.') from error
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'EXPTIME must be positive and finite, got {value!r}.')
    return seconds


def _copy_map(s_map, data=None, meta=None):
    return Map(
        np.array(s_map.data if data is None else data, dtype=np.float64, copy=True),
        s_map.meta.copy() if meta is None else meta,
    )


def _source_valid_mask(source_path: str | Path, shape: tuple[int, ...]) -> np.ndarray:
    with fits.open(source_path, memmap=False) as hdul:
        if 'VALID_MASK' in hdul:
            mask = np.asarray(hdul['VALID_MASK'].data, dtype=bool)
            if mask.shape != shape:
                raise ValueError(
                    f'VALID_MASK shape {mask.shape} does not match image shape {shape}.'
                )
            return mask
    return np.ones(shape, dtype=bool)


def _submap_pair(data_map, mask_map, bounds):
    xmin, ymin, xmax, ymax = bounds
    bottom_left = SkyCoord(
        xmin * u.arcsec, ymin * u.arcsec, frame=data_map.coordinate_frame
    )
    top_right = SkyCoord(
        xmax * u.arcsec, ymax * u.arcsec, frame=data_map.coordinate_frame
    )
    return (
        data_map.submap(bottom_left=bottom_left, top_right=top_right),
        mask_map.submap(bottom_left=bottom_left, top_right=top_right),
    )


_OBSERVATION_METADATA_KEYS = (
    'bunit',
    'date-obs',
    'date_obs',
    'date-avg',
    'detector',
    'exptime',
    'instrume',
    'lvl_num',
    'obsrvtry',
    'quality',
    'telescop',
    'waveunit',
    'wavelnth',
    'wavelength',
)


def _restore_observation_metadata(source_map, target_map):
    """Restore non-WCS observation identity dropped by reprojection."""
    meta = target_map.meta.copy()
    for key in _OBSERVATION_METADATA_KEYS:
        value = source_map.meta.get(key)
        if value is not None:
            meta[key] = value
    return _copy_map(target_map, meta=meta)


def apply_common_geometry(
    s_map,
    valid_mask: np.ndarray | None = None,
    geometry: GeometryConfig | None = None,
    *,
    already_north_up: bool = False,
):
    """Apply the shared geometry path and return ``(map, valid_mask)``.

    The mask follows every geometric operation using nearest-neighbour
    interpolation. Invalid detector samples and interpolation footprints are
    represented as NaN in the returned map rather than as artificial zeros.
    """
    geometry = geometry or GeometryConfig()
    data = np.asarray(s_map.data, dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(data.shape, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != data.shape:
        raise ValueError(
            f'valid_mask shape {valid_mask.shape} does not match image shape {data.shape}.'
        )
    valid_mask &= np.isfinite(data)
    data_map = _copy_map(s_map, np.where(valid_mask, data, np.nan))
    mask_map = _copy_map(s_map, valid_mask.astype(np.float32))

    if geometry.north_up and not already_north_up:
        data_map = data_map.rotate(
            recenter=True,
            missing=np.nan,
            order=geometry.interpolation_order,
            clip=False,
        )
        mask_map = mask_map.rotate(recenter=True, missing=0, order=0, clip=False)

    exact_hpc_grid = (
        geometry.hpc_bounds_arcsec is not None and geometry.shape is not None
    )
    if exact_hpc_grid:
        xmin, ymin, xmax, ymax = geometry.hpc_bounds_arcsec
        ny, nx = (int(value) for value in geometry.shape)
        reference = SkyCoord(
            0.5 * (xmin + xmax) * u.arcsec,
            0.5 * (ymin + ymax) * u.arcsec,
            frame=data_map.coordinate_frame,
        )
        scale = u.Quantity(
            [(xmax - xmin) / (nx - 1), (ymax - ymin) / (ny - 1)],
            u.arcsec / u.pix,
        )
        target_header = make_fitswcs_header((ny, nx), reference, scale=scale)
        target_wcs = WCS(target_header)
        reproject_kwargs = {'shape_out': (ny, nx)}
        data_map = data_map.reproject_to(target_wcs, **reproject_kwargs)
        mask_map = mask_map.reproject_to(
            target_wcs, order='nearest-neighbor', **reproject_kwargs
        )
    elif geometry.hpc_bounds_arcsec is not None:
        data_map, mask_map = _submap_pair(
            data_map, mask_map, geometry.hpc_bounds_arcsec
        )
    elif geometry.max_radius_rsun is not None and geometry.reproject_reference is None:
        radius = geometry.max_radius_rsun * data_map.rsun_obs.to_value(u.arcsec)
        data_map, mask_map = _submap_pair(
            data_map, mask_map, (-radius, -radius, radius, radius)
        )

    if exact_hpc_grid:
        pass
    elif geometry.reproject_reference is not None:
        reference = Map(str(geometry.reproject_reference))
        reproject_kwargs = {'shape_out': reference.data.shape}
        data_map = data_map.reproject_to(reference.wcs, **reproject_kwargs)
        mask_map = mask_map.reproject_to(
            reference.wcs, order='nearest-neighbor', **reproject_kwargs
        )
    elif geometry.shape is not None:
        ny, nx = (int(value) for value in geometry.shape)
        dimensions = u.Quantity([nx, ny], u.pix)
        data_map = data_map.resample(dimensions, method='linear')
        mask_map = mask_map.resample(dimensions, method='nearest')

    final_mask = np.asarray(mask_map.data) >= 0.5
    final_mask &= np.isfinite(data_map.data)
    if geometry.max_radius_rsun is not None:
        coordinates = all_coordinates_from_map(data_map)
        radius = np.hypot(coordinates.Tx, coordinates.Ty) / data_map.rsun_obs
        final_mask &= np.asarray(radius.to_value(u.one)) <= geometry.max_radius_rsun
    if not final_mask.any():
        raise ValueError('Geometry and masks removed every valid image pixel.')
    prepared = _copy_map(data_map, np.where(final_mask, data_map.data, np.nan))
    return _restore_observation_metadata(s_map, prepared), final_mask


def _load_aiapy_calibration():
    if sys.version_info < (3, 12):
        raise ImportError(
            'AIA preparation with pinned aiapy 0.12.1 requires Python >=3.12; '
            'the SuNeRF core remains compatible with Python >=3.10.'
        )
    try:
        from aiapy.calibrate import correct_degradation, register, update_pointing
    except ImportError as error:
        raise ImportError(
            'AIA preparation requires pinned aiapy 0.12.1; on Python >=3.12 '
            'install `sunerf[euv-prep]`.'
        ) from error
    installed_version = importlib.metadata.version('aiapy')
    if installed_version != '0.12.1':
        raise ImportError(
            f'AIA preparation supports aiapy 0.12.1 exactly, found {installed_version}; '
            'install `sunerf[euv-prep]` on Python >=3.12.'
        )
    return update_pointing, register, correct_degradation


def _file_cache_key(path: str | Path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _cached_file_sha256(path: str, mtime_ns: int, size: int) -> str:
    del mtime_ns, size
    return _sha256(path)


@lru_cache(maxsize=8)
def _cached_pointing_table(path: str, mtime_ns: int, size: int) -> QTable:
    del mtime_ns, size
    table = QTable.read(path)
    required = {'T_START', 'T_STOP'}
    if not required.issubset(table.colnames):
        raise ValueError(
            f'AIA pointing table {path!s} is missing columns {sorted(required - set(table.colnames))}.'
        )
    for column in required:
        if not isinstance(table[column], Time):
            table[column] = Time(table[column], scale='utc')
    return table


@lru_cache(maxsize=8)
def _cached_correction_table(path: str, mtime_ns: int, size: int) -> QTable:
    del mtime_ns, size
    table = QTable.read(path)
    required = {'DATE', 'WAVELNTH', 'EFF_AREA'}
    if not required.issubset(table.colnames):
        raise ValueError(
            f'AIA correction table {path!r} is missing columns '
            f'{sorted(required - set(table.colnames))}.'
        )
    return table


def _load_pointing_table(path: str | Path) -> QTable:
    """Load a pinned pointing table from the content-aware in-process cache."""
    table = _cached_pointing_table(*_file_cache_key(path))
    return table.copy(copy_data=True)


def _load_aia_inputs(correction_path, pointing_path):
    correction_key = _file_cache_key(correction_path)
    pointing_key = _file_cache_key(pointing_path)
    pointing_table = _load_pointing_table(pointing_path)
    correction_table = _cached_correction_table(*correction_key).copy(copy_data=True)
    return {
        'pointing_table': pointing_table,
        'pointing_sha256': _cached_file_sha256(*pointing_key),
        'correction_table': correction_table,
        'correction_sha256': _cached_file_sha256(*correction_key),
    }


def prepare_aia_map(
    s_map,
    *,
    correction_table_path: str | Path,
    pointing_table_path: str | Path,
    geometry: GeometryConfig | None = None,
    valid_mask: np.ndarray | None = None,
):
    """Calibrate one AIA level-1 map using only pinned local aiapy inputs."""
    correction_table_path = Path(correction_table_path)
    pointing_table_path = Path(pointing_table_path)
    if not correction_table_path.is_file():
        raise FileNotFoundError(f'AIA correction table not found: {correction_table_path}')
    if not pointing_table_path.is_file():
        raise FileNotFoundError(f'AIA pointing table not found: {pointing_table_path}')
    if int(s_map.meta.get('quality', -1)) != 0:
        raise ValueError(f'AIA QUALITY must be zero, got {s_map.meta.get("quality")!r}.')
    exposure = _exposure_seconds(s_map)
    input_unit = _aia_level1_input_unit(s_map)
    output_unit = _aia_rate_unit(input_unit)
    update_pointing, register, correct_degradation = _load_aiapy_calibration()
    pinned_inputs = _load_aia_inputs(correction_table_path, pointing_table_path)

    raw_data = np.asarray(s_map.data)
    raw_mask = np.isfinite(raw_data)
    if valid_mask is not None:
        supplied_mask = np.asarray(valid_mask, dtype=bool)
        if supplied_mask.shape != raw_data.shape:
            raise ValueError(
                f'valid_mask shape {supplied_mask.shape} does not match '
                f'AIA image shape {raw_data.shape}.'
            )
        raw_mask &= supplied_mask
    mask_map = _copy_map(s_map, raw_mask.astype(np.float32))

    calibrated = update_pointing(
        _copy_map(s_map), pointing_table=pinned_inputs['pointing_table']
    )
    mask_map = update_pointing(
        mask_map, pointing_table=pinned_inputs['pointing_table']
    )
    calibrated = register(calibrated, missing=np.nan, order=3)
    mask_map = register(mask_map, missing=0, order=0)
    registered_mask = np.asarray(mask_map.data) >= 0.5
    if registered_mask.shape != calibrated.data.shape:
        raise ValueError('AIA registered data and mask shapes do not match.')
    calibrated = correct_degradation(
        calibrated, correction_table=pinned_inputs['correction_table']
    )
    meta = calibrated.meta.copy()
    meta['bunit'] = output_unit
    meta['expnorm'] = True
    calibrated = _copy_map(calibrated, calibrated.data / exposure, meta)
    prepared, final_mask = apply_common_geometry(
        calibrated, registered_mask, geometry, already_north_up=True
    )
    # Reprojection constructs a fresh WCS header and may discard non-WCS
    # metadata. Restore the calibrated measurement contract explicitly so the
    # prepared product can be validated and stamped independently of the
    # geometry implementation used by SunPy/reproject.
    prepared_meta = prepared.meta.copy()
    prepared_meta['bunit'] = output_unit
    prepared_meta['expnorm'] = True
    prepared = _copy_map(prepared, meta=prepared_meta)
    provenance = {
        'adapter': 'aia',
        'calibration_id': f'aiapy:pinned:{pinned_inputs["correction_sha256"][:16]}',
        'calibration_steps': [
            'aiapy.update_pointing',
            'aiapy.register',
            'aiapy.correct_degradation',
            'divide_by_exposure_seconds',
        ],
        'correction_table_path': str(correction_table_path.resolve()),
        'correction_table_sha256': pinned_inputs['correction_sha256'],
        'pointing_table_path': str(pointing_table_path.resolve()),
        'pointing_table_sha256': pinned_inputs['pointing_sha256'],
        'sensitivity_convention': 'reference_epoch',
        'measurement_semantics': 'per_native_pixel',
        **_geometry_radiometry_provenance(calibrated, prepared),
    }
    return prepared, final_mask, provenance


def _validate_euvi_spacecraft(s_map, spacecraft: str):
    spacecraft = spacecraft.upper()
    if spacecraft not in {'A', 'B'}:
        raise ValueError("EUVI spacecraft must be 'A' or 'B'.")
    identity = ' '.join(
        str(s_map.meta.get(key, ''))
        for key in ('obsrvtry', 'telescop', 'detector', 'instrume')
    ).lower()
    compact = re.sub(r'[^a-z0-9]+', '', identity)
    words = set(re.findall(r'[a-z0-9]+', identity))
    if 'secchi' not in compact and 'euvi' not in compact:
        raise ValueError('EUVI input header must positively identify SECCHI/EUVI.')
    expected_long = ('stereoa', 'ahead') if spacecraft == 'A' else ('stereob', 'behind')
    expected_short = 'sta' if spacecraft == 'A' else 'stb'
    if not any(token in compact for token in expected_long) and expected_short not in words:
        raise ValueError(
            f'EUVI header does not positively identify STEREO-{spacecraft}.'
        )


def _product_levels_match(header_level, product_level) -> bool:
    normalized_header = re.sub(r'[^a-z0-9.]+', '', str(header_level).lower())
    normalized_product = re.sub(r'[^a-z0-9.]+', '', str(product_level).lower())
    if normalized_header == normalized_product:
        return True
    header_numbers = re.findall(r'\d+(?:\.\d+)?', normalized_header)
    product_numbers = re.findall(r'\d+(?:\.\d+)?', normalized_product)
    return (
        len(header_numbers) == 1
        and len(product_numbers) == 1
        and float(header_numbers[0]) == float(product_numbers[0])
    )


def prepare_euvi_map(
    s_map,
    *,
    spacecraft: str,
    product_level: str,
    sensitivity_convention: str,
    geometry: GeometryConfig | None = None,
    valid_mask: np.ndarray | None = None,
):
    """Validate an externally calibrated SECCHI/EUVI product; no radiometry."""
    product_level = _nonempty(product_level, 'product_level')
    sensitivity_convention = _validated_sensitivity_convention(
        sensitivity_convention, context='EUVI preparation'
    )
    if not product_level.upper().startswith(('SECCHI', 'EUVI')):
        raise ValueError(
            'EUVI product_level must name a documented SECCHI/EUVI calibrated product.'
        )
    _validate_euvi_spacecraft(s_map, spacecraft.upper())
    measurement_unit = _measurement_unit(s_map)
    _exposure_seconds(s_map)
    for key in ('level', 'lvl_num', 'data_lev', 'proclev'):
        header_level = s_map.meta.get(key)
        if header_level is not None and str(header_level).strip():
            if not _product_levels_match(header_level, product_level):
                raise ValueError(
                    f'EUVI header {key.upper()}={header_level!r} does not match '
                    f'product_level={product_level!r}.'
                )
            break
    calibrated_meta = s_map.meta.copy()
    calibrated_meta['bunit'] = measurement_unit
    calibrated = _copy_map(s_map, meta=calibrated_meta)
    prepared, final_mask = apply_common_geometry(calibrated, valid_mask, geometry)
    provenance = {
        'adapter': 'euvi',
        'spacecraft': spacecraft.upper(),
        'external_product_level': product_level,
        'calibration_steps': ['external_SECCHI_calibration', 'geometry_only_in_SuNeRF'],
        'sensitivity_convention': sensitivity_convention,
        'measurement_semantics': 'per_native_pixel',
        **_geometry_radiometry_provenance(calibrated, prepared),
    }
    return prepared, final_mask, provenance


def _eui_level(s_map) -> str:
    for key in ('level', 'lvl_num', 'data_lev', 'proclev'):
        value = s_map.meta.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError('EUI input is missing calibrated product-level metadata.')


def _is_level_two(value: str) -> bool:
    compact = value.lower().replace('_', '').replace('-', '').replace(' ', '')
    return compact in {'2', '2.0', 'l2', 'level2'} or compact.startswith('l2')


def prepare_eui_map(
    s_map,
    *,
    calibration_id: str,
    sensitivity_convention: str,
    geometry: GeometryConfig | None = None,
    valid_mask: np.ndarray | None = None,
):
    """Validate a calibrated EUI L2 product; do not recalibrate it."""
    calibration_id = _nonempty(calibration_id, 'calibration_id')
    sensitivity_convention = _validated_sensitivity_convention(
        sensitivity_convention, context='EUI preparation'
    )
    identity = ' '.join(
        str(s_map.meta.get(key, ''))
        for key in ('obsrvtry', 'telescop', 'detector', 'instrume')
    ).lower()
    compact_identity = re.sub(r'[^a-z0-9]+', '', identity)
    if 'eui' not in compact_identity or not any(
        token in compact_identity for token in ('solarorbiter', 'solo')
    ):
        raise ValueError(
            'EUI input header must positively identify Solar Orbiter/EUI.'
        )
    level = _eui_level(s_map)
    if not _is_level_two(level):
        raise ValueError(f'EUI input must be calibrated level 2, got {level!r}.')
    _measurement_unit(s_map)
    _exposure_seconds(s_map)
    header_calibration_id = s_map.meta.get('cal_id') or s_map.meta.get('vers_cal')
    if header_calibration_id and str(header_calibration_id).strip() != calibration_id:
        raise ValueError(
            f'EUI calibration metadata {header_calibration_id!r} does not match '
            f'{calibration_id!r}.'
        )
    prepared, final_mask = apply_common_geometry(
        _copy_map(s_map), valid_mask, geometry
    )
    provenance = {
        'adapter': 'eui',
        'calibration_id': calibration_id,
        'external_product_level': level,
        'calibration_steps': ['validated_EUI_L2', 'geometry_only_in_SuNeRF'],
        'sensitivity_convention': sensitivity_convention,
        'measurement_semantics': 'per_native_pixel',
        **_geometry_radiometry_provenance(s_map, prepared),
    }
    return prepared, final_mask, provenance


def _stamp_metadata(s_map, source_path, provenance):
    source_sha256 = _sha256(source_path)
    meta = s_map.meta.copy()
    meta['prepschm'] = PREPARED_EUV_SCHEMA
    meta['schemav'] = 2
    if 'calibration_id' in provenance:
        meta['cal_id'] = provenance['calibration_id']
    meta['senscon'] = provenance['sensitivity_convention']
    meta['src_sha'] = source_sha256
    meta['srcfile'] = Path(source_path).name
    meta['maskext'] = 'VALID_MASK'
    meta['natpxsr'] = provenance['native_pixel_solid_angle_sr']
    meta['prepxsr'] = provenance['prepared_pixel_solid_angle_sr']
    meta['radsem'] = provenance['measurement_semantics']
    meta['geomsem'] = 'sample_interp_no_solid_angle_conversion'
    meta['prepdate'] = datetime.now(timezone.utc).isoformat()
    meta['bunit'] = _measurement_unit(s_map)
    history = list(meta.get('history', [])) if isinstance(meta.get('history'), list) else []
    history.append('SuNeRF prepared-EUV-v2; invalid pixels are NaN and VALID_MASK=0')
    for step in provenance['calibration_steps']:
        history.append(f'SuNeRF calibration provenance: {step}')
    meta['history'] = history
    return _copy_map(s_map, meta=meta), source_sha256


def _atomic_write_fits(s_map, valid_mask, output_path, *, overwrite=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f'Prepared output already exists: {output_path}; pass overwrite=True explicitly.'
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{output_path.name}.', suffix='.fits', dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.unlink()
        s_map.save(temporary_path, filetype='fits', overwrite=True)
        with fits.open(temporary_path, mode='append', memmap=False) as hdul:
            mask_hdu = fits.ImageHDU(
                np.asarray(valid_mask, dtype=np.uint8), name='VALID_MASK'
            )
            mask_hdu.header['BUNIT'] = '1'
            mask_hdu.header['MASKTRUE'] = 1
            mask_hdu.header['MASKFALS'] = 0
            hdul.append(mask_hdu)
            hdul.flush(output_verify='exception')
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _prepare_file(
    source_path,
    output_path,
    map_preparer: Callable,
    *,
    geometry=None,
    overwrite=False,
    **adapter_kwargs,
):
    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    s_map = Map(str(source_path))
    source_mask = _source_valid_mask(source_path, s_map.data.shape)
    prepared, valid_mask, provenance = map_preparer(
        s_map,
        geometry=geometry,
        valid_mask=source_mask,
        **adapter_kwargs,
    )
    prepared, _ = _stamp_metadata(prepared, source_path, provenance)
    _atomic_write_fits(prepared, valid_mask, output_path, overwrite=overwrite)
    return PreparationResult(Path(output_path))


def prepare_aia_file(source_path, output_path, **kwargs) -> PreparationResult:
    return _prepare_file(source_path, output_path, prepare_aia_map, **kwargs)


def prepare_euvi_file(source_path, output_path, **kwargs) -> PreparationResult:
    return _prepare_file(source_path, output_path, prepare_euvi_map, **kwargs)


def prepare_eui_file(source_path, output_path, **kwargs) -> PreparationResult:
    return _prepare_file(source_path, output_path, prepare_eui_map, **kwargs)


def _bounded_workers(workers: int | None, task_count: int) -> int:
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    if workers < 0:
        raise ValueError('workers must be non-negative.')
    if task_count == 0:
        return 0
    return max(1, min(int(workers) or 1, task_count, MAX_PREP_WORKERS))


def prepare_euv_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    adapter: str,
    geometry: GeometryConfig | None = None,
    workers: int | None = None,
    overwrite: bool = False,
    **adapter_kwargs,
) -> list[PreparationResult]:
    """Prepare calibrated FITS files for one instrument."""
    sources = [Path(path) for path in input_paths]
    if not sources:
        raise ValueError('No EUV input files were selected.')
    if len({str(path.resolve()) for path in sources}) != len(sources):
        raise ValueError('The EUV input list contains duplicate source paths.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preparers = {
        'aia': prepare_aia_file,
        'euvi': prepare_euvi_file,
        'eui': prepare_eui_file,
    }
    try:
        preparer = preparers[adapter.lower()]
    except KeyError as error:
        raise ValueError(f'Unsupported EUV adapter {adapter!r}.') from error

    requests = []
    destinations = set()
    for source in sources:
        destination = output_dir / f'{source.stem}.prepared.fits'
        if destination in destinations:
            raise ValueError(f'Output filename collision for {destination.name}.')
        destinations.add(destination)
        requests.append((source, destination))

    def run_one(request):
        source, destination = request
        return preparer(
            source,
            destination,
            geometry=geometry,
            overwrite=overwrite,
            **adapter_kwargs,
        )

    worker_count = _bounded_workers(workers, len(requests))
    if worker_count == 1:
        results = [run_one(request) for request in requests]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(run_one, requests))
    return results


def _expand_inputs(values: Sequence[str]) -> list[Path]:
    paths = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(Path(match) for match in (matches or [value]) if Path(match).is_file())
    if not paths:
        raise ValueError('No input files matched --input.')
    return paths


def _add_common_arguments(parser):
    parser.add_argument('--input', nargs='+', required=True, help='FITS paths or glob patterns.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--shape', type=int, nargs=2, metavar=('NY', 'NX'))
    parser.add_argument(
        '--hpc-bounds',
        type=float,
        nargs=4,
        metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
    )
    parser.add_argument('--max-radius-rsun', type=float)
    parser.add_argument('--reproject-reference')
    parser.add_argument('--no-north-up', action='store_true')
    parser.add_argument('--interpolation-order', type=int, default=3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m sunerf.data.euv.prepare',
        description='Create strict SuNeRF prepared-EUV-v2 FITS products.',
    )
    subparsers = parser.add_subparsers(dest='adapter', required=True)
    aia = subparsers.add_parser('aia', help='Calibrate AIA L1 data with pinned aiapy tables.')
    _add_common_arguments(aia)
    aia.add_argument('--correction-table', required=True)
    aia.add_argument('--pointing-table', required=True)

    euvi = subparsers.add_parser(
        'euvi', help='Ingest an externally calibrated SECCHI/EUVI product.'
    )
    _add_common_arguments(euvi)
    euvi.add_argument('--spacecraft', choices=('A', 'B'), required=True)
    euvi.add_argument('--product-level', required=True)
    euvi.add_argument(
        '--sensitivity-convention',
        choices=sorted(SENSITIVITY_CONVENTIONS),
        required=True,
        help='Calibration-epoch convention already applied by the external product.',
    )

    eui = subparsers.add_parser('eui', help='Ingest a calibrated Solar Orbiter/EUI L2 product.')
    _add_common_arguments(eui)
    eui.add_argument('--calibration-id', required=True)
    eui.add_argument(
        '--sensitivity-convention',
        choices=sorted(SENSITIVITY_CONVENTIONS),
        required=True,
        help='Calibration-epoch convention already applied by the level-2 product.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    geometry = GeometryConfig(
        shape=tuple(args.shape) if args.shape else None,
        hpc_bounds_arcsec=tuple(args.hpc_bounds) if args.hpc_bounds else None,
        max_radius_rsun=args.max_radius_rsun,
        reproject_reference=args.reproject_reference,
        north_up=not args.no_north_up,
        interpolation_order=args.interpolation_order,
    )
    adapter_kwargs = {}
    if args.adapter == 'aia':
        adapter_kwargs.update(
            correction_table_path=args.correction_table,
            pointing_table_path=args.pointing_table,
        )
    elif args.adapter == 'euvi':
        adapter_kwargs.update(
            spacecraft=args.spacecraft,
            product_level=args.product_level,
            sensitivity_convention=args.sensitivity_convention,
        )
    else:
        adapter_kwargs.update(
            calibration_id=args.calibration_id,
            sensitivity_convention=args.sensitivity_convention,
        )
    results = prepare_euv_files(
        _expand_inputs(args.input),
        args.output_dir,
        adapter=args.adapter,
        geometry=geometry,
        workers=args.workers,
        overwrite=args.overwrite,
        **adapter_kwargs,
    )
    print(f'Prepared {len(results)} files with schema {PREPARED_EUV_SCHEMA}.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
