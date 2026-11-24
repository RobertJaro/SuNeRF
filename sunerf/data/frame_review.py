"""Find and remove invalid frames of a FITS image sequence.

The data stay on the server; only compact previews and CSV files travel:

``scan`` (server)
    Reads every file once and computes per-frame statistics: header quality,
    valid-pixel fraction, saturated pixels, intensity percentiles, and
    temporal-consistency scores against the neighbouring frames of the same
    instrument and channel. Frames are classified as ``reject`` (a clear
    defect), ``warn`` (worth a look, e.g. a flare, saturated pixels or a short
    automatic exposure) or ``ok``. Saturation never rejects a frame; its pixels
    are masked while the data load. Writes a review directory with ``frames.csv`` (one row
    per file), ``scan.json``, ``thumbs/`` (JPEG previews on one fixed
    intensity scale per channel) and ``index.html``; no FITS data.

``review`` (local, on a copy of the review directory)
    Serves the review page on localhost and opens it. Every decision is saved
    immediately to ``decisions.csv`` in the review directory (columns
    ``path, decision, auto_status, group_name, time, reasons``; ``decision`` is
    ``reject`` or ``keep``). Reopening continues from it. A saturation bar
    selects the level for ``instruments[].saturation_dn`` of the plasma config
    (applied by ``run_plasma`` while loading the data) from the per-frame peak
    detector DN (recommended: 0.8 of the plateau of saturated frames) and saves
    it to ``saturation.csv``.

``apply`` (server)
    Walks a review directory, or a folder of them, and moves the rejected files
    of each into a ``rejected/`` subdirectory next to them: those marked
    ``reject`` in its ``decisions.csv``, or its automatic rejections from
    ``frames.csv`` when it was not reviewed. Dataset globs such as
    ``prepared/aia/*.prepared.fits`` no longer match them, so training ignores
    them without a configuration change. A single CSV with ``path`` and
    ``decision`` (or ``status``) columns, e.g. one edited in a spreadsheet, can
    also be applied directly.

``restore`` (server)
    Moves rejected files back.

The scan is instrument agnostic: it groups files by observatory, detector and
channel from the FITS header (``OBSRVTRY``, ``DETECTOR``/``INSTRUME``/``TELESCOP``,
``WAVELNTH``/``FILTER``), reads raw, calibrated and SuNeRF-prepared EUV products
(including the ``VALID_MASK`` extension), and works for any 2D image sequence.

For coronagraph images, scan with --cme. Painting or duplicating a region
saves a boolean NumPy file under masks/ named after the FITS file, with True
at excluded pixels in FITS array order. By default, brush edits, copies and
clears apply to both files of a tB/pB observation. Uncheck "Apply to both tB
and pB" to edit one channel. Press M to enable editing and copy the most
recent earlier mask on the same grid, when available. Sync masks/ alongside the
separate decisions CSV. Applying the review directory installs the masks in
.review-masks/ beside the observations. Training loads them by filename and
applies them in memory. Applying a CSV directly moves whole-frame rejects only.

CME quality scores use the measured brightness. For tB previews, the scan
subtracts the pixelwise minimum of UTC daily mean brightness on each detector
grid. Only automatically kept frames contribute (including warnings), and
invalid or manually masked pixels are omitted from the means. The same
background is used to display rejected frames for reassessment.

Saturation is counted from the data when they are detector counts (``BUNIT``
without a per-second unit) and the detector saturation level is known
(AIA, SECCHI) or given with ``--saturation-dn``; otherwise from the header
keyword ``NSATPIX``. Prepared products are exposure normalized and degradation
corrected, so their own values cannot be compared with the detector limit;
``--source-dir`` points at the files they were prepared from (header
``SRCFILE``), whose ``NSATPIX``, ``QUALITY`` and counts are used instead.

Example::

    # server
    python -m sunerf.data.frame_review scan "prepared/aia/*.prepared.fits" \\
        --source-dir raw/aia --output review/aia --workers 32
    # local
    rsync -av server:/path/review/aia .
    python -m sunerf.data.frame_review review aia
    rsync -av aia/decisions.csv server:/path/review/aia/
    # server: every review directory under review/
    python -m sunerf.data.frame_review apply review
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import re
import shutil
import socket
import sys
from tempfile import NamedTemporaryFile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
from astropy.io import fits
from astropy.time import Time

from sunerf.data.review_masks import decode_brush, mask_path, preview_png, read_mask, write_mask

REJECTED_DIRECTORY = 'rejected'
REJECTED_LOG = 'rejected.log'
THUMBNAIL_SIZE = 192
# Neighbouring frames of the same channel on each side for the temporal scores.
NEIGHBOURS = 3
# Groups with fewer frames get no temporal scores.
MIN_GROUP_SIZE = 5
# Robust z-scores (median / 1.4826 MAD over the channel) of the temporal scores.
REJECT_Z = 6.0
WARN_Z = 4.0
# Fraction of the channel's median valid-pixel fraction.
VALID_FRACTION_REJECT = 0.9
VALID_FRACTION_WARN = 0.98
# Fraction of the native detector pixels that are saturated (1e-6 is ~17 AIA
# pixels; a few isolated saturated pixels are common).
SATURATED_FRACTION_WARN = 1.0e-6
# Fraction of the channel's median exposure time (automatic exposure control).
SHORT_EXPOSURE_WARN = 0.5
# Frames of different channels within this interval share one review row.
ROW_TOLERANCE_SECONDS = 120.0
# Detector saturation in DN; level-1 values are dark subtracted, so a margin.
SATURATION_DN = {'AIA': 16383.0, 'SECCHI': 16383.0, 'EUVI': 16383.0}
SATURATION_MARGIN = 0.9

# Saturation-level selection. The peak of every frame is measured in the
# detector-DN convention of ``instruments[].saturation_dn``: level-1 DN as they
# are, SECCHI_PREP rates on the 14-bit scale of DSATVAL, other rates times the
# exposure. Prepared products carry it in their header (PEAKDN, PEAKNPX). Saturated frames
# share one ceiling and their brightest pixels form a plateau at it.
DETECTOR_FULL_SCALE_DN = 16383.0
# Pixels within this fraction of a frame's peak form its plateau; a plateau
# needs PLATEAU_MIN_PIXELS.
PLATEAU_TOLERANCE = 1.0e-3
PLATEAU_MIN_PIXELS = 3
# Frames whose plateau lies within this fraction of the highest peak of the
# channel share its ceiling; CEILING_MIN_FRAMES of them identify saturation.
CEILING_TOLERANCE = 0.01
CEILING_MIN_FRAMES = 2
# Recommended level: this fraction of the saturation, below the nonlinear range
# (13100 DN for the 16375 DN AIA level-1 ceiling), rounded down to 3 digits.
RECOMMENDED_FRACTION = 0.8

TIME_KEYS = ('DATE-OBS', 'DATE_OBS', 'DATE-AVG', 'T_OBS')
EXPOSURE_KEYS = ('EXPTIME', 'XPOSURE')
CHANNEL_KEYS = ('WAVELNTH', 'WAVELENGTH', 'FILTER')
HEADER_COLUMNS = ('QUALITY', 'NSATPIX', 'AECTYPE', 'IMG_TYPE')

METRICS = (
    ('level', 'Median log10 intensity'),
    ('peak', '99.9th percentile log10 intensity'),
    ('valid_fraction', 'Valid-pixel fraction'),
    ('saturated_fraction', 'Saturated-pixel fraction'),
    ('peak_dn', 'Peak detector DN (saturation_dn units)'),
    ('exposure', 'Exposure time [s]'),
    ('structure', 'Median |log difference| to neighbours'),
    ('level_z', 'Level jump z-score'),
    ('peak_z', 'Peak jump z-score'),
    ('structure_z', 'Structure z-score'),
)


# ----------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------

def _header_value(header, keys):
    for key in keys:
        value = header.get(key)
        if value is not None and str(value).strip() != '':
            return value
    return None


def _image_hdu(hdul):
    """First 2D image HDU that is not a mask, and the VALID_MASK if present."""
    image = None
    for hdu in hdul:
        if hdu.name == 'VALID_MASK' or not hdu.is_image or hdu.data is None:
            continue
        if np.ndim(hdu.data) == 2:
            image = hdu
            break
    if image is None:
        raise ValueError('no 2D image HDU')
    mask = None
    if 'VALID_MASK' in hdul:
        mask = np.asarray(hdul['VALID_MASK'].data, dtype=bool)
        if mask.shape != image.data.shape:
            mask = None
    return image, mask


def _parse_time(header):
    value = _header_value(header, TIME_KEYS)
    if value is None:
        raise ValueError(f'no observation time ({"/".join(TIME_KEYS)})')
    text = str(value).strip().replace('_TAI', '').rstrip('Z')
    return Time(text, format='isot', scale='utc') if 'T' in text else Time(text, scale='utc')


def _instrument(header):
    name = str(_header_value(header, ('DETECTOR', 'INSTRUME', 'TELESCOP')) or 'unknown').strip()
    # AIA names its four telescopes AIA_1 .. AIA_4; the channel identifies them.
    name = re.sub(r'^AIA_\d$', 'AIA', name)
    observatory = str(header.get('OBSRVTRY') or '').strip()
    if observatory and observatory.lower() not in name.lower():
        name = f'{observatory} {name}'
    return name


def _channel(header):
    value = _header_value(header, CHANNEL_KEYS)
    if value is None:
        return ''
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return str(value).strip()


def _float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _is_rate(header):
    """True for exposure-normalized data (a per-second unit or EXPNORM)."""
    unit = str(header.get('BUNIT') or '').replace(' ', '').lower()
    if header.get('EXPNORM') is True:
        return True
    return bool(re.search(r'/s\b|/s$|s-1|s\^-1|s\*\*-1|/sec', unit))


def _saturation_level(header, override):
    if override is not None:
        return float(override)
    text = ' '.join(str(header.get(key) or '') for key in ('INSTRUME', 'TELESCOP', 'DETECTOR')).upper()
    for name, level in SATURATION_DN.items():
        if name in text:
            return level
    return None


def _saturated_mask(data, header, valid, saturation_dn):
    """Saturated pixels of detector-count data, or None when not determinable."""
    if _is_rate(header):
        return None
    level = _saturation_level(header, saturation_dn)
    if level is None:
        return None
    with np.errstate(invalid='ignore'):
        return valid & (data >= SATURATION_MARGIN * level)


def _block_reduce(data, valid, size):
    """Mean of the valid pixels in square blocks, at most ``size`` per side."""
    factor = max(1, int(math.ceil(max(data.shape) / size)))
    rows = int(math.ceil(data.shape[0] / factor)) * factor
    cols = int(math.ceil(data.shape[1] / factor)) * factor
    total = np.zeros((rows, cols), dtype=np.float64)
    count = np.zeros((rows, cols), dtype=np.float64)
    total[:data.shape[0], :data.shape[1]] = np.where(valid, data, 0.0)
    count[:data.shape[0], :data.shape[1]] = valid
    shape = (rows // factor, factor, cols // factor, factor)
    total = total.reshape(shape).sum(axis=(1, 3))
    count = count.reshape(shape).sum(axis=(1, 3))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan), factor


def _block_any(mask, factor):
    rows = int(math.ceil(mask.shape[0] / factor)) * factor
    cols = int(math.ceil(mask.shape[1] / factor)) * factor
    padded = np.zeros((rows, cols), dtype=bool)
    padded[:mask.shape[0], :mask.shape[1]] = mask
    return padded.reshape(rows // factor, factor, cols // factor, factor).any(axis=(1, 3))


def _detector_dn(data, header):
    """Values in the ``instruments[].saturation_dn`` convention, or None.

    Level-1 detector counts stay as they are (AIA), SECCHI_PREP rates are put
    on the 14-bit scale of their saturation value DSATVAL (EUVI), and other
    rates are multiplied by the exposure time (EUI level 2). SuNeRF-prepared
    products are degradation corrected and give None: their peak is the header
    PEAKDN written by the preparation.
    """
    if header.get('PREPSCHM'):
        return None
    if not _is_rate(header):
        return data
    saturation = _float_or_nan(header.get('DSATVAL'))
    if np.isfinite(saturation) and saturation > 0:
        return data / saturation * DETECTOR_FULL_SCALE_DN
    exposure = _float_or_nan(_header_value(header, EXPOSURE_KEYS))
    if np.isfinite(exposure) and exposure > 0:
        return data * exposure
    return None


def _peak_statistics(data, header, valid):
    """Peak detector DN of a frame and the number of pixels on its plateau."""
    if header.get('PEAKDN') is not None:  # prepared on the native grid
        return float(header['PEAKDN']), int(header.get('PEAKNPX') or 0)
    dn = _detector_dn(data, header)
    if dn is None:
        return math.nan, None
    with np.errstate(invalid='ignore'):
        values = dn[valid & np.isfinite(dn)]
    if not values.size:
        return math.nan, None
    peak = float(values.max())
    if peak <= 0:
        return peak, 0
    return peak, int(np.count_nonzero(values >= peak * (1.0 - PLATEAU_TOLERANCE)))


def _round_down(value, digits=3):
    """``value`` rounded down to ``digits`` significant digits."""
    if not np.isfinite(value) or value <= 0:
        return None
    step = 10.0 ** (math.floor(math.log10(value)) - digits + 1)
    result = math.floor(value / step) * step
    return int(result) if result >= 1 else float(f'{result:.{digits}g}')


def saturation_estimate(records):
    """Saturation of one channel from the peaks of its frames.

    Frames with a plateau of at least PLATEAU_MIN_PIXELS pixels within
    CEILING_TOLERANCE of the channel's highest peak are saturated; with at least
    CEILING_MIN_FRAMES of them the saturation is the median of their peaks and
    the recommended ``saturation_dn`` is RECOMMENDED_FRACTION of it. Without
    such a ceiling no saturation was observed and nothing is recommended.
    """
    peaks = np.array([_float_or_nan(r.get('peak_dn')) for r in records], dtype=np.float64)
    plateaus = np.array([r.get('plateau_pixels') or 0 for r in records], dtype=np.float64)
    finite = np.isfinite(peaks)
    estimate = {'ceiling_dn': None, 'saturation_dn': None, 'saturated_frames': 0,
                'recommended_dn': None, 'frames_with_peak': int(finite.sum())}
    if not finite.any():
        return estimate
    ceiling = float(peaks[finite].max())
    estimate['ceiling_dn'] = round(ceiling, 3)
    saturated = (finite & (peaks >= ceiling * (1.0 - CEILING_TOLERANCE))
                 & (plateaus >= PLATEAU_MIN_PIXELS))
    estimate['saturated_frames'] = int(saturated.sum())
    if saturated.sum() >= CEILING_MIN_FRAMES:
        saturation = float(np.median(peaks[saturated]))
        estimate['saturation_dn'] = round(saturation, 3)
        estimate['recommended_dn'] = _round_down(RECOMMENDED_FRACTION * saturation)
    return estimate


def dataset_saturation(records, group_names):
    """Per-channel saturation estimates and one recommended level for the dataset.

    One ``saturation_dn`` serves all channels of an instrument, so the
    recommendation is the lowest of the channels that saturated.
    """
    channels = {}
    for group_index, name in enumerate(group_names):
        members = [r for r in records if r['group'] == group_index]
        channels[name] = saturation_estimate(members)
    recommended = [c['recommended_dn'] for c in channels.values() if c['recommended_dn']]
    return {
        'fraction': RECOMMENDED_FRACTION,
        'recommended_dn': min(recommended) if recommended else None,
        'channels': channels,
    }


def _source_information(source_path, saturation_dn):
    """QUALITY, NSATPIX, saturation count and peak detector DN of the source file."""
    info = {}
    with fits.open(source_path, memmap=False) as hdul:
        image, mask = _image_hdu(hdul)
        header = image.header
        data = np.asarray(image.data, dtype=np.float32)
    for key in HEADER_COLUMNS:
        if header.get(key) is not None:
            info[key] = header.get(key)
    info['native_pixels'] = int(data.shape[0] * data.shape[1])
    valid = np.isfinite(data) if mask is None else (np.isfinite(data) & mask)
    if header.get('NSATPIX') is None:
        saturated = _saturated_mask(data, header, valid, saturation_dn)
        if saturated is not None:
            info['NSATPIX'] = int(saturated.sum())
    info['peak_dn'], info['plateau_pixels'] = _peak_statistics(data, header, valid)
    return info


def scan_file(path, source_dir=None, saturation_dn=None):
    """Statistics and a log-intensity thumbnail of one frame."""
    record = {'path': os.path.abspath(path), 'name': os.path.basename(path), 'error': ''}
    try:
        with fits.open(path, memmap=False) as hdul:
            image, file_mask = _image_hdu(hdul)
            header = image.header.copy()
            data = np.asarray(image.data, dtype=np.float32)
        time = _parse_time(header)
        record.update(
            time=time.isot, unix=float(time.unix), instrument=_instrument(header),
            channel=_channel(header),
            exposure=_float_or_nan(_header_value(header, EXPOSURE_KEYS)),
            bunit=str(header.get('BUNIT') or ''),
        )
        for key in HEADER_COLUMNS:
            record[key.lower()] = header.get(key)

        with np.errstate(invalid='ignore'):
            valid = np.isfinite(data) & (data > 0)
        if file_mask is not None:
            valid &= file_mask
        n_pixels = data.size
        record['valid_fraction'] = float(valid.sum() / n_pixels)
        values = data[valid]
        if values.size:
            p01, p50, p99, p999 = np.log10(np.percentile(values, [1.0, 50.0, 99.0, 99.9]))
            record.update(level=float(p50), low=float(p01), high=float(p99), peak=float(p999),
                          maximum=float(np.log10(values.max())))
        else:
            record.update(level=math.nan, low=math.nan, high=math.nan, peak=math.nan,
                          maximum=math.nan)

        record['peak_dn'], record['plateau_pixels'] = _peak_statistics(
            data, header, np.isfinite(data) if file_mask is None else (np.isfinite(data) & file_mask))
        saturated = _saturated_mask(data, header, valid, saturation_dn)
        native_pixels = n_pixels
        # The header count (AIA NSATPIX) is authoritative; the computed mask
        # marks the saturated pixels in the thumbnail.
        n_saturated = None if saturated is None else int(saturated.sum())
        if header.get('NSATPIX') is not None:
            n_saturated = int(header['NSATPIX'])
        source_name = header.get('SRCFILE')
        if source_dir is not None and source_name:
            source_path = os.path.join(source_dir, str(source_name))
            if os.path.isfile(source_path):
                source = _source_information(source_path, saturation_dn)
                native_pixels = source.pop('native_pixels', native_pixels)
                if 'NSATPIX' in source:
                    n_saturated = int(source['NSATPIX'])
                # The preparation input defines the saturation level.
                record['peak_dn'] = source.pop('peak_dn')
                record['plateau_pixels'] = source.pop('plateau_pixels')
                for key, value in source.items():
                    if key != 'NSATPIX' and record.get(key.lower()) is None:
                        record[key.lower()] = value
            else:
                record['source_missing'] = True
        record['nsatpix'] = n_saturated
        record['saturated_fraction'] = (
            math.nan if n_saturated is None else n_saturated / max(native_pixels, 1)
        )

        reduced, factor = _block_reduce(data, valid, THUMBNAIL_SIZE)
        with np.errstate(invalid='ignore', divide='ignore'):
            thumb = np.where(reduced > 0, np.log10(reduced), np.nan)
        record['thumb'] = thumb.astype(np.float16)
        record['thumb_invalid'] = _block_any(~valid, factor)
        record['thumb_saturated'] = None if saturated is None else _block_any(saturated, factor)
        record['cmap'] = _colormap_name(header, data)
    except Exception as error:  # a defective file is a finding, not a failure
        record.update(error=f'unreadable: {type(error).__name__}: {error}', time='', unix=math.nan,
                      instrument='unreadable', channel='', thumb=None, thumb_saturated=None)
    return record


# SunPy colormap name patterns by instrument keyword (FITS header text).
COLORMAP_PATTERNS = (
    ('AIA', 'sdoaia{channel}'),
    ('EUVI', 'euvi{channel}'),
    ('FSI', 'solar orbiterfsi{channel}'),
    ('HRI', 'solar orbiterhri_euv{channel}'),
    ('SUVI', 'goes-rsuvi{channel}'),
    ('EIT', 'sohoeit{channel}'),
)


def _colormap_name(header, data):
    """SunPy colormap of the instrument channel, ``gray`` when unknown."""
    import logging
    import warnings

    from sunpy.visualization import colormaps as sunpy_colormaps

    text = ' '.join(
        str(header.get(key) or '') for key in ('INSTRUME', 'TELESCOP', 'DETECTOR')
    ).upper()
    channel = _channel(header)
    for keyword, pattern in COLORMAP_PATTERNS:
        name = pattern.format(channel=channel)
        if keyword in text and name in sunpy_colormaps.cmlist:
            return name
    try:  # other instruments: the colormap SunPy assigns to the map
        from sunpy.map import Map

        logging.getLogger('sunpy').setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Map(data, header).cmap.name
    except Exception:
        return 'gray'


def _read_cme_image(path):
    with fits.open(path, memmap=False) as hdul:
        image, valid_mask = _image_hdu(hdul)
        data = np.array(image.data, dtype=np.float32, copy=True)
        if valid_mask is not None:
            data[~valid_mask] = np.nan
        return data, image.header


# Match the September CME overview and input-data plots.
CME_COLORMAPS = {
    'cor2': 'soholasco2',
    'cor2_clear': 'soholasco2',
    'lasco_c2': 'soholasco2',
    'lasco_c3': 'soholasco3',
    'ccor': 'Blues_r',
    'punch': 'punch',
    'punch_pam': 'punch',
}


def _set_cme_thumbnail(record, brightness, valid):
    reduced, factor = _block_reduce(brightness, valid, THUMBNAIL_SIZE)
    record['thumb'] = reduced.astype(np.float16)
    record['thumb_invalid'] = _block_any(~valid, factor)


def _preview_rgb(scaled, cmap, invalid):
    rgb = cmap(np.clip(np.nan_to_num(scaled, nan=0.0), 0, 1), bytes=True)[..., :3]
    rgb[invalid] = INVALID_RGB
    return rgb


def _write_cme_preview(path, brightness, valid, low, peak, cmap_name):
    from PIL import Image

    scaled = (brightness - low) / max(peak - low, 1e-12)
    rgb = _preview_rgb(scaled, _colormap(cmap_name), ~valid)
    Image.fromarray(rgb[::-1]).save(path, quality=90, subsampling=0)


def _scan_cme_file(task):
    """Measure calibrated MSB for classification before rendering the previews."""
    from sunerf.data.coronagraph.detect_outliers import measure_image

    path, number = task
    source = Path(path)
    paired = source.parent.name in ('tB', 'pB')
    record = dict(path=path, name=source.name, error='', unix=math.nan, time='',
                  instrument=source.parent.parent.name if paired else source.parent.name,
                  channel=source.parent.name if paired else 'tB', thumb=None,
                  thumb_saturated=None, cmap='gray')
    try:
        data, header = _read_cme_image(path)
        record['cmap'] = CME_COLORMAPS.get(record['instrument'].lower()) or _colormap_name(header, data)
        time = _parse_time(header)
        record.update(time=time.isot, unix=float(time.unix), quality=header.get('QUALITY'))
        measure_image(record, data, header)
        valid = np.isfinite(data)
        # A fixed asinh scale retains negative brightness and faint outer-corona detail.
        if record.get('std') == 0:
            record['error'] = 'constant brightness across all finite pixels'
        brightness = np.arcsinh(data.astype(np.float64) / 1e-12)
        if valid.any():
            values = brightness[valid]
            low, level, high, peak = np.percentile(values, [1, 50, 99, 99.9])
            record.update(low=low, level=level, high=high, peak=peak, maximum=values.max())
        record.update(height=data.shape[0], width=data.shape[1])
        _set_cme_thumbnail(record, brightness, valid)
        name = f'{number:06d}.jpg'
    except Exception as error:
        record.update(error=f'unreadable: {type(error).__name__}: {error}', thumb=None)
        return record
    record['preview'] = 'previews/' + name
    return record


def _render_cme_previews(members, review_dir, workers, cmap_name):
    from sunerf.data.coronagraph.compute_correction import minimum_daily_mean

    # Members are time-sorted and belong to one instrument/channel. Grid
    # sizes can differ between observing modes, even in the same directory.
    grids = defaultdict(list)
    for frame in members:
        if frame['channel'] == 'tB' and frame.get('preview'):
            grids[(frame['height'], frame['width'])].append(frame)

    def reference_images(frames):
        for frame in frames:
            if frame['status'] == 'reject':
                continue
            data, _ = _read_cme_image(frame['path'])
            mask = review_dir / 'masks' / frame['mask_file']
            if mask.is_file():
                data[read_mask(mask, data.shape)] = np.nan
            yield frame['time'][:10], data

    backgrounds = {}

    def brightness_for(frame):
        data, _ = _read_cme_image(frame['path'])
        background = backgrounds.get(data.shape)
        residual = data if background is None else data - background
        return np.arcsinh(residual.astype(np.float64) / 1e-12)

    def update_thumbnail(frame):
        brightness = brightness_for(frame)
        _set_cme_thumbnail(frame, brightness, np.isfinite(brightness))

    def render(frame):
        brightness = brightness_for(frame)
        _write_cme_preview(review_dir / frame['preview'], brightness, np.isfinite(brightness), low, peak, cmap_name)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for shape, frames in grids.items():
            background = minimum_daily_mean(reference_images(frames), shape)
            if np.isfinite(background).any():
                backgrounds[shape] = background
                list(pool.map(update_thumbnail, frames))
            else:
                print(f"No valid background for {frames[0]['group_name']} {shape}; using raw tB previews.")
            for frame in frames:
                frame['tb_background_subtracted'] = int(shape in backgrounds)
        low, peak = _display_range(members, limit_decades=False)
        list(pool.map(render, members))
    return low, peak


def _classify_cme(members):
    from sunerf.data.coronagraph.detect_outliers import detect_outliers

    classify(members, reject_z=7.0, warn_z=4.5, valid_reject=0.85, valid_warn=0.95)
    dated = [r for r in members if np.isfinite(r['unix'])]
    metrics = [dict(r, time=datetime.fromtimestamp(r['unix'], timezone.utc)) for r in dated]
    if metrics:
        detect_outliers(metrics, window=6, threshold=6)
    for frame, metric in zip(dated, metrics):
        if metric['reasons']:
            frame['reasons'].insert(0, metric['reasons'])
            frame['n_reject'] += 1
            frame['status'] = 'reject'


def _scan_task(arguments):
    return scan_file(*arguments)


# ----------------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------------

def _robust_z(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    z = np.full(values.shape, np.nan)
    if finite.sum() < MIN_GROUP_SIZE:
        return z
    median = np.median(values[finite])
    scale = 1.4826 * np.median(np.abs(values[finite] - median))
    scale = max(scale, 1.0e-3 * max(1.0, abs(median)), 1.0e-6)
    z[finite] = (values[finite] - median) / scale
    return z


def _neighbour_indices(index, count):
    lower, upper = max(0, index - NEIGHBOURS), min(count, index + NEIGHBOURS + 1)
    return [j for j in range(lower, upper) if j != index]


def temporal_scores(records):
    """Level, peak and structure deviations of every frame from its neighbours.

    ``records`` is one instrument channel sorted by time. The level and peak
    residuals are the frame's median and 99.9th-percentile log intensity minus
    the median over its neighbours; the structure score is the median absolute
    log difference of its thumbnail to the pixelwise median of the neighbours'
    thumbnails.
    """
    count = len(records)
    level = np.array([r.get('level', math.nan) for r in records], dtype=np.float64)
    peak = np.array([r.get('peak', math.nan) for r in records], dtype=np.float64)
    level_residual = np.full(count, np.nan)
    peak_residual = np.full(count, np.nan)
    structure = np.full(count, np.nan)
    for index, record in enumerate(records):
        neighbours = _neighbour_indices(index, count)
        if not neighbours:
            continue
        with np.errstate(all='ignore'):
            if np.isfinite(level[neighbours]).any():
                level_residual[index] = level[index] - np.nanmedian(level[neighbours])
            if np.isfinite(peak[neighbours]).any():
                peak_residual[index] = peak[index] - np.nanmedian(peak[neighbours])
        thumb = record.get('thumb')
        stack = [records[j]['thumb'] for j in neighbours
                 if records[j].get('thumb') is not None and thumb is not None
                 and records[j]['thumb'].shape == thumb.shape]
        if stack:
            with np.errstate(all='ignore'), _suppress_nan_warnings():
                reference = np.nanmedian(np.stack(stack).astype(np.float32), axis=0)
                difference = np.abs(thumb.astype(np.float32) - reference)
                if np.isfinite(difference).any():
                    structure[index] = float(np.nanmedian(difference))
    level_z = _robust_z(level_residual)
    peak_z = _robust_z(peak_residual)
    structure_z = _robust_z(structure)
    for index, record in enumerate(records):
        record.update(level_residual=level_residual[index], peak_residual=peak_residual[index],
                      structure=structure[index], level_z=level_z[index],
                      peak_z=peak_z[index], structure_z=structure_z[index])


class _suppress_nan_warnings:
    def __enter__(self):
        import warnings
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *exc):
        return self._context.__exit__(*exc)


def classify(records, *, reject_z=REJECT_Z, warn_z=WARN_Z,
             valid_reject=VALID_FRACTION_REJECT, valid_warn=VALID_FRACTION_WARN):
    """Set ``status`` (reject/warn/ok) and ``reasons`` of one channel's records."""
    valid = np.array([r.get('valid_fraction', math.nan) for r in records], dtype=np.float64)
    exposure = np.array([r.get('exposure', math.nan) for r in records], dtype=np.float64)
    median_valid = np.nanmedian(valid) if np.isfinite(valid).any() else math.nan
    median_exposure = np.nanmedian(exposure) if np.isfinite(exposure).any() else math.nan
    times = [r.get('unix') for r in records]
    for index, record in enumerate(records):
        reject, warn = [], []
        if record['error'].startswith('unreadable'):
            reject.append(record['error'])
        elif record['error']:
            warn.append(record['error'])
        quality = record.get('quality')
        if quality not in (None, '') and _float_or_nan(quality) != 0:
            reject.append(f'header QUALITY={quality}')
        fraction = record.get('valid_fraction', math.nan)
        if np.isfinite(median_valid) and np.isfinite(fraction):
            if fraction < valid_reject * median_valid:
                reject.append(f'valid fraction {fraction:.3f} (channel median {median_valid:.3f})')
            elif fraction < valid_warn * median_valid:
                warn.append(f'valid fraction {fraction:.3f} (channel median {median_valid:.3f})')
        saturated = record.get('saturated_fraction', math.nan)
        # Saturation never rejects a frame: run_plasma masks the saturated pixels
        # (instruments[].saturation_dn) and keeps the rest of the frame.
        if np.isfinite(saturated) and saturated > SATURATED_FRACTION_WARN:
            warn.append(f'{record.get("nsatpix")} saturated pixels ({saturated:.1e} of detector)')
        for key, label in (('level_z', 'level jump'), ('structure_z', 'structure change')):
            z = record.get(key, math.nan)
            if np.isfinite(z) and abs(z) > reject_z:
                reject.append(f'{label} z={z:+.1f}')
            elif np.isfinite(z) and abs(z) > warn_z:
                warn.append(f'{label} z={z:+.1f}')
        z = record.get('peak_z', math.nan)
        if np.isfinite(z) and abs(z) > warn_z:
            warn.append(f'peak jump z={z:+.1f}')
        value = record.get('exposure', math.nan)
        if np.isfinite(median_exposure) and np.isfinite(value) and value < SHORT_EXPOSURE_WARN * median_exposure:
            warn.append(f'short exposure {value:.2f} s (channel median {median_exposure:.2f} s)')
        if index > 0 and times[index] == times[index - 1]:
            warn.append('duplicate observation time')
        record['status'] = 'reject' if reject else ('warn' if warn else 'ok')
        record['reasons'] = reject + warn
        record['n_reject'] = len(reject)


def assign_rows(records):
    """Observation rows of the review: one frame per channel and time.

    The channel with the most frames is the reference; each of its frames
    anchors one row. Every other frame joins the row of its nearest anchor
    when it lies within half the reference cadence (and ``ROW_TOLERANCE_SECONDS``)
    and that channel's slot is still free, closest frames first; a channel
    without a frame near an anchor leaves a gap in that row. Frames that join no
    anchor are grouped among themselves by time. Rows are numbered in time order.
    """
    timed = [r for r in records if np.isfinite(r.get('unix', math.nan))]
    rows = []  # (time, records)
    if timed:
        counts = {}
        for record in timed:
            counts[record['group']] = counts.get(record['group'], 0) + 1
        reference = max(sorted(counts), key=lambda group: counts[group])
        anchors = sorted((r for r in timed if r['group'] == reference), key=lambda r: r['unix'])
        anchor_times = np.array([r['unix'] for r in anchors])
        spacing = float(np.median(np.diff(anchor_times))) if len(anchors) > 1 else math.inf
        tolerance = min(ROW_TOLERANCE_SECONDS, 0.5 * spacing)
        slots = [{reference: record} for record in anchors]
        candidates, leftover = [], []
        for record in timed:
            if record['group'] == reference:
                continue
            index = int(np.searchsorted(anchor_times, record['unix']))
            nearest = min(
                (k for k in (index - 1, index) if 0 <= k < len(anchors)),
                key=lambda k: abs(anchor_times[k] - record['unix']),
            )
            distance = abs(anchor_times[nearest] - record['unix'])
            (candidates if distance <= tolerance else leftover).append((distance, nearest, record))
        for _, nearest, record in sorted(candidates, key=lambda item: item[0]):
            if record['group'] in slots[nearest]:
                leftover.append((None, None, record))
            else:
                slots[nearest][record['group']] = record
        rows.extend((anchor['unix'], list(slot.values())) for anchor, slot in zip(anchors, slots))
        start, members = None, {}
        for record in sorted((item[2] for item in leftover), key=lambda r: r['unix']):
            if (start is None or record['unix'] - start > ROW_TOLERANCE_SECONDS
                    or record['group'] in members):
                start, members = record['unix'], {}
                rows.append((start, []))
            members[record['group']] = record
            rows[-1][1].append(record)
    rows.sort(key=lambda row: row[0])
    for number, (_, members) in enumerate(rows):
        for record in members:
            record['row'] = number
    count = len(rows)
    timed_ids = {id(record) for record in timed}
    for record in records:
        if id(record) not in timed_ids:  # no observation time: a row of its own
            record['row'] = count
            count += 1
    return count


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def _slug(text):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('_') or 'group'


# Percentiles of every thumbnail whose minimum and maximum over the channel's
# accepted frames set its common display range (log10 intensity); the
# percentiles keep single hot or dead pixels from setting the range.
DISPLAY_PERCENTILES = (1.0, 99.9)
# Largest displayed dynamic range: near-zero off-limb noise of raw detector
# frames would otherwise pull the minimum of one frame down by many decades
# and wash out every other frame.
MAX_DISPLAY_DECADES = 4.0
INVALID_RGB = (0, 255, 0)
SATURATED_RGB = (255, 40, 40)


def _display_range(members, *, limit_decades=True):
    """Common log10 display range of one channel: min/max over its frames.

    At most ``MAX_DISPLAY_DECADES`` below the maximum. The range spans the
    accepted frames (automatic status other than
    ``reject``), so one corrupt frame does not compress the rest; rejected
    frames are shown on the same scale and clip.
    """
    accepted = [r for r in members if r['status'] != 'reject'] or members
    with np.errstate(all='ignore'), _suppress_nan_warnings():
        bounds = np.array([
            np.nanpercentile(r['thumb'].astype(np.float32), DISPLAY_PERCENTILES)
            for r in accepted
        ])
        vmin, vmax = float(np.nanmin(bounds[:, 0])), float(np.nanmax(bounds[:, 1]))
    if limit_decades:
        vmin = max(vmin, vmax - MAX_DISPLAY_DECADES)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def _colormap(name):
    import matplotlib
    from sunpy.visualization import colormaps as sunpy_colormaps

    if name in sunpy_colormaps.cmlist:
        return sunpy_colormaps.cmlist[name]
    try:
        return matplotlib.colormaps[name]
    except KeyError:
        return matplotlib.colormaps['gray']


def _render_thumbnails(records, output_dir, group_names, *, cme=False, workers=1):
    """JPEG previews on one colormap and one value range per channel."""
    from collections import Counter

    from PIL import Image

    thumbs_dir = output_dir / 'thumbs'
    channels = {}
    for group_index, name in enumerate(group_names):
        members = [r for r in records if r['group'] == group_index and r.get('thumb') is not None]
        if not members:
            continue
        cmap_name = Counter(r.get('cmap') or 'gray' for r in members).most_common(1)[0][0]
        if cme:
            vmin, vmax = _render_cme_previews(members, output_dir, workers, cmap_name)
        else:
            vmin, vmax = _display_range(members)
        cmap = _colormap(cmap_name)
        scale = 'asinh' if cme else 'log10'
        channels[name] = {'cmap': cmap_name, f'{scale}_min': round(vmin, 4), f'{scale}_max': round(vmax, 4)}
        group_dir = thumbs_dir / _slug(name)
        group_dir.mkdir(parents=True, exist_ok=True)
        for record in members:
            scaled = (record['thumb'].astype(np.float32) - vmin) / (vmax - vmin)
            rgb = _preview_rgb(scaled, cmap, record['thumb_invalid'] | ~np.isfinite(scaled))
            if record.get('thumb_saturated') is not None:
                rgb[record['thumb_saturated']] = SATURATED_RGB
            file_name = f'{record["index"]:06d}.jpg'
            Image.fromarray(np.ascontiguousarray(rgb[::-1])).save(group_dir / file_name, quality=82, subsampling=0)
            record['thumb_url'] = f'thumbs/{_slug(name)}/{file_name}'
    return channels


def _json_value(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


FRAMES_CSV = 'frames.csv'
DECISIONS_CSV = 'decisions.csv'
SCAN_JSON = 'scan.json'
SATURATION_CSV = 'saturation.csv'
# One row per channel; selected_dn is the level chosen in the review (the same
# for every channel of the dataset) or "none".
SATURATION_COLUMNS = ('channel', 'ceiling_dn', 'saturation_dn', 'saturated_frames',
                      'recommended_dn', 'selected_dn')
PAGE = 'index.html'
REASON_SEPARATOR = '; '

# Columns of frames.csv: identity and review layout, then the statistics.
FRAME_COLUMNS = (
    'index', 'group', 'group_name', 'row', 'time', 'unix', 'name', 'path', 'thumb',
    'status', 'n_reject', 'reasons', 'width', 'height', 'preview', 'mask_file', 'tb_background_subtracted',
)
STAT_COLUMNS = (
    'exposure', 'quality', 'nsatpix', 'aectype', 'img_type', 'valid_fraction',
    'saturated_fraction', 'peak_dn', 'plateau_pixels', 'level', 'low', 'high', 'peak',
    'maximum', 'level_residual',
    'peak_residual', 'structure', 'level_z', 'peak_z', 'structure_z', 'error',
)
INTEGER_COLUMNS = ('index', 'group', 'row', 'n_reject', 'plateau_pixels', 'width', 'height', 'tb_background_subtracted')
TEXT_COLUMNS = ('group_name', 'time', 'name', 'path', 'thumb', 'status', 'reasons',
                'img_type', 'aectype', 'error', 'preview', 'mask_file')
DECISION_COLUMNS = ('path', 'decision', 'auto_status', 'group_name', 'time', 'reasons')
PAGE_STATS = tuple(key for key, _ in METRICS) + ('plateau_pixels', 'nsatpix', 'quality', 'aectype', 'low', 'high',
                                                  'maximum')


def _write_frames_csv(records, path):
    with open(path, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(FRAME_COLUMNS + STAT_COLUMNS)
        for record in records:
            row = []
            for column in FRAME_COLUMNS + STAT_COLUMNS:
                value = record.get(column)
                if column == 'reasons':
                    value = REASON_SEPARATOR.join(
                        text.replace(';', ',') for text in (value or []))
                elif column == 'thumb':
                    value = record.get('thumb_url')
                value = _json_value(value)
                row.append('' if value is None else value)
            writer.writerow(row)


def load_frames(review_dir):
    """Rows of ``frames.csv`` with integer, float and text columns typed."""
    frames = []
    with open(Path(review_dir) / FRAMES_CSV, newline='') as stream:
        for row in csv.DictReader(stream):
            frame = {}
            for key, value in row.items():
                if key in TEXT_COLUMNS:
                    frame[key] = value
                elif value == '':
                    frame[key] = None
                elif key in INTEGER_COLUMNS:
                    frame[key] = int(value)
                else:
                    try:
                        frame[key] = float(value)
                    except ValueError:
                        frame[key] = value
            frame['reasons'] = [text for text in frame['reasons'].split(REASON_SEPARATOR) if text]
            frames.append(frame)
    return frames


def read_decisions(path):
    """Paths to reject from a decisions or frames CSV, or a plain list.

    ``decisions.csv`` (column ``decision``) and any CSV edited by hand with a
    ``decision`` column reject the rows marked ``reject``; ``frames.csv``
    without one rejects its automatic rejections (column ``status``). Other
    files list one path per line, ``#`` starting a comment.
    """
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with open(path, newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            if 'path' not in columns:
                raise SystemExit(f'{path} has no "path" column.')
            key = 'decision' if 'decision' in columns else 'status' if 'status' in columns else None
            if key is None:
                raise SystemExit(f'{path} has neither a "decision" nor a "status" column.')
            return [row['path'] for row in reader if row[key].strip().lower() == 'reject']
    entries = []
    with open(path) as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith('#'):
                entries.append(line)
    return entries


def _read_decision_rows(review_dir):
    path = Path(review_dir) / DECISIONS_CSV
    if not path.is_file():
        return {}
    with open(path, newline='') as stream:
        return {row['path']: row['decision'].strip().lower() for row in csv.DictReader(stream)}


def _mask_partners(frames):
    """Pair editable tB/pB images from the same observation and native pixel grid."""
    observations = defaultdict(list)
    for frame in frames:
        channel = Path(frame['path']).parent
        if channel.name in ('tB', 'pB') and frame.get('preview') and frame.get('mask_file'):
            observations[channel.parent, frame['row']].append(frame)
    partners = {}
    for members in observations.values():
        if len(members) != 2:
            continue
        a, b = members
        if (Path(a['path']).parent != Path(b['path']).parent
                and (a['width'], a['height']) == (b['width'], b['height'])
                and a['unix'] is not None and b['unix'] is not None and abs(a['unix'] - b['unix']) <= 1.0):
            partners[a['index']] = b['index']
            partners[b['index']] = a['index']
    return partners


def build_page(review_dir):
    """Write ``index.html`` from ``frames.csv``, ``scan.json`` and ``decisions.csv``.

    Needs only the review directory, so it runs wherever the directory was
    copied to.
    """
    from sunerf.data.frame_review_page import PAGE_TEMPLATE

    review_dir = Path(review_dir)
    meta = json.loads((review_dir / SCAN_JSON).read_text())
    frames = load_frames(review_dir)
    decisions = _read_decision_rows(review_dir)
    groups = [None] * (max((f['group'] for f in frames), default=-1) + 1)
    for frame in frames:
        groups[frame['group']] = frame['group_name']
    cme = meta.get('cme', False)
    partners = _mask_partners(frames) if cme else {}
    mask_dir = review_dir / 'masks'
    available_masks = {p.relative_to(mask_dir).as_posix() for p in mask_dir.rglob('*.npy')} if cme else set()
    masks = {f['path']: 'mask-preview/' + quote(f['mask_file']) for f in frames
             if f.get('mask_file') in available_masks}
    metrics = METRICS
    stats = PAGE_STATS
    if cme:
        metrics = [(key, label.replace('log10', 'asinh').replace('|log difference|', '|asinh difference|'))
                   for key, label in METRICS if key not in ('saturated_fraction', 'peak_dn')]
        stats = [key for key, _ in metrics] + ['quality', 'low', 'high', 'maximum']
    page_frames = []
    for frame in frames:
        page_frames.append({
            'i': frame['index'], 'g': frame['group'], 'row': frame['row'],
            'partner': partners.get(frame['index']),
            't': None if frame.get('unix') is None else frame['unix'] * 1000.0,
            'time': frame['time'], 'name': frame['name'], 'path': frame['path'],
            'maskFile': frame.get('mask_file'), 'preview': frame.get('preview'), 'width': frame.get('width'), 'height': frame.get('height'),
            'thumb': frame['thumb'] or None, 'auto': frame['status'],
            'tbBackgroundSubtracted': bool(frame.get('tb_background_subtracted')),
            'reasons': frame['reasons'], 'nr': frame['n_reject'],
            'stats': {key: _json_value(frame.get(key)) for key in stats},
        })
    payload = {
        'cme': cme, 'masks': masks,
        'id': meta['id'], 'title': meta['title'], 'created': meta['created'],
        'groups': groups, 'rows': max((f['row'] for f in frames), default=-1) + 1,
        'metrics': [list(item) for item in metrics], 'frames': page_frames,
        'decisions': decisions, 'decisionColumns': list(DECISION_COLUMNS),
        'channels': meta.get('channels', {}),
        'saturation': meta.get('saturation'),
        'saturationSelected': _read_saturation_selection(review_dir),
        'saturationColumns': list(SATURATION_COLUMNS),
    }
    text = json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')
    page = PAGE_TEMPLATE.replace('__TITLE__', _html_escape(meta['title'])).replace('__DATA__', text)
    (review_dir / PAGE).write_text(page)
    return review_dir / PAGE


def _print_saturation(saturation):
    print('\nSaturation (peak detector DN, the units of instruments[].saturation_dn):')
    for name, estimate in saturation['channels'].items():
        if estimate['frames_with_peak'] == 0:
            print(f'  {name:26s} no peak: prepared before PEAKDN (prepare again) and no --source-dir')
        elif estimate['saturation_dn'] is None:
            print(f'  {name:26s} no saturation plateau; highest peak {estimate["ceiling_dn"]:.6g}')
        else:
            print(f'  {name:26s} saturates at {estimate["saturation_dn"]:.6g} in '
                  f'{estimate["saturated_frames"]} frames -> {estimate["recommended_dn"]}')
    level = saturation['recommended_dn']
    print(f'  recommended: saturation_dn: {level if level else "(none observed; keep the current level)"}')


def _read_saturation_selection(review_dir):
    """Level saved by the review (``saturation.csv``), '' when none was saved."""
    path = Path(review_dir) / SATURATION_CSV
    if not path.is_file():
        return ''
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    return rows[0]['selected_dn'].strip() if rows else ''


def _write_saturation_text(review_dir, text):
    """Validate a saturation CSV posted by the page and replace saturation.csv."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or set(SATURATION_COLUMNS) - set(rows[0]):
        raise ValueError(f'saturation must be a CSV with columns {", ".join(SATURATION_COLUMNS)}')
    selected = {row['selected_dn'].strip().lower() for row in rows}
    if len(selected) != 1:
        raise ValueError('one selected_dn serves every channel of the dataset')
    value = selected.pop()
    if value != 'none':
        try:
            level = float(value)
        except ValueError as error:
            raise ValueError(f'selected_dn must be a level in DN or "none", got {value!r}') from error
        if not np.isfinite(level) or level <= 0:
            raise ValueError(f'selected_dn must be positive, got {value!r}')
    target = Path(review_dir) / SATURATION_CSV
    temporary = target.with_suffix('.csv.tmp')
    temporary.write_text(text)
    os.replace(temporary, target)
    return len(rows), value


def _html_escape(text):
    return (str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


def scan(patterns, output_dir, *, source_dir=None, workers=None, saturation_dn=None, title=None, cme=False):
    """Scan the files of ``patterns`` and write the review directory ``output_dir``.

    The directory holds only compact products: ``frames.csv`` (statistics and
    automatic flags, one row per file), ``scan.json`` (scan metadata),
    ``thumbs/`` (JPEG previews) and ``index.html``. It is meant to be copied to
    the machine where the frames are reviewed.
    """
    files = set()
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_dir():
            images = [p for p in path.iterdir()
                      if p.name.lower().endswith(('.fits', '.fts', '.fits.gz', '.fts.gz')) and p.is_file()]
            if not images:
                raise SystemExit(f'No FITS files in {path}.')
            files.update(str(p.absolute()) for p in images)
        else:
            if not glob.has_magic(str(path)) and not path.is_file():
                raise SystemExit(f'Missing input: {path}')
            files.update(os.path.abspath(p) for p in glob.iglob(str(path)) if os.path.isfile(p))
    files = sorted(files)
    if not files:
        raise SystemExit(f'No files match {patterns!r}.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or min(8, os.cpu_count() or 1)
    print(f'Scanning {len(files)} files with {workers} workers')
    tasks = [(path, source_dir, saturation_dn) for path in files]
    worker = _scan_task
    if cme:
        preview_dir = output_dir / 'previews'
        preview_dir.mkdir(exist_ok=True)
        (output_dir / 'masks').mkdir(exist_ok=True)
        worker = _scan_cme_file
        tasks = [(path, i) for i, path in enumerate(files)]
    records = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, record in enumerate(pool.map(worker, tasks, chunksize=4), 1):
                records.append(record)
                if count % 200 == 0:
                    print(f'  {count}/{len(files)}')
    else:
        records = [worker(task) for task in tasks]

    if cme:
        root = Path(os.path.commonpath([os.path.dirname(path) for path in files]))
        if root.name in ('tB', 'pB'):
            root = root.parent
        # A fresh review starts with already installed masks. A repeat scan
        # preserves pending local edits, including a cleared mask.
        previous = output_dir / FRAMES_CSV
        known = {r['path']: r.get('mask_file') for r in load_frames(output_dir)} if previous.exists() else {}
        for record in records:
            source = Path(record['path'])
            record['mask_file'] = known.get(str(source)) or str(source.relative_to(root)) + '.npy'
            if not known.get(str(source)) and mask_path(source).is_file():
                target = output_dir / 'masks' / record['mask_file']
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mask_path(source), target)

    group_names = sorted({f'{r["instrument"]} {r["channel"]}'.strip() for r in records})
    for record in records:
        record['group_name'] = f'{record["instrument"]} {record["channel"]}'.strip()
        record['group'] = group_names.index(record['group_name'])
    records.sort(key=lambda r: (
        r['group'], r['unix'] if np.isfinite(r.get('unix', math.nan)) else math.inf, r['name']))
    for index, record in enumerate(records):
        record['index'] = index
    for group_index in range(len(group_names)):
        members = [r for r in records if r['group'] == group_index]
        temporal_scores(members)
        if cme:
            _classify_cme(members)
        else:
            classify(members)
    assign_rows(records)

    channels = _render_thumbnails(records, output_dir, group_names, cme=cme, workers=workers)
    saturation = None if cme else dataset_saturation(records, group_names)
    _write_frames_csv(records, output_dir / FRAMES_CSV)
    created = datetime.now(timezone.utc).isoformat(timespec='seconds')
    meta = {
        'id': f'{os.path.abspath(output_dir)}|{created}',
        'title': title or f'Frame review: {", ".join(patterns)}',
        'created': created,
        'patterns': list(patterns), 'cme': cme,
        'review_dir': os.path.abspath(output_dir),
        'host': socket.gethostname(),
        'frames': len(records),
        # Colormap and common log10 display range of every channel's previews.
        'channels': channels,
        # Peak-based saturation estimate per channel and the recommended level.
        'saturation': saturation,
    }
    (output_dir / SCAN_JSON).write_text(json.dumps(meta, indent=2) + '\n')
    build_page(output_dir)

    print(f'{"channel":28s} {"frames":>7s} {"reject":>7s} {"warn":>7s}')
    for group_index, name in enumerate(group_names):
        members = [r for r in records if r['group'] == group_index]
        print(f'{name:28s} {len(members):7d} '
              f'{sum(r["status"] == "reject" for r in members):7d} '
              f'{sum(r["status"] == "warn" for r in members):7d}')
    if saturation:
        _print_saturation(saturation)
    missing = sum(bool(r.get('source_missing')) for r in records)
    if missing:
        print(f'{missing} source files (header SRCFILE) not found in {source_dir}; '
              'their saturation comes from their own header only.')
    size = sum(p.stat().st_size for p in output_dir.rglob('*') if p.is_file()) / 1e6
    remote = os.path.abspath(output_dir)
    mask_sync = f'  rsync -acv --delete {output_dir.name}/masks/ <server>:{remote}/masks/\n' if cme else ''
    print(f'\nReview directory ({size:.1f} MB): {remote}\n'
          'Copy it to your machine and review the frames there:\n'
          f'  rsync -av <server>:{remote} .\n'
          f'  python -m sunerf.data.frame_review review {output_dir.name}\n'
          'Copy the decisions back and apply them here:\n'
          f'  rsync -av {output_dir.name}/{DECISIONS_CSV} <server>:{remote}/\n'
          f'{mask_sync}  python -m sunerf.data.frame_review apply {remote}\n'
          f'(without {DECISIONS_CSV}, apply uses the automatic flags of {FRAMES_CSV}; '
          'apply also takes the parent folder of several review directories)')
    return records


# ----------------------------------------------------------------------------
# Local review
# ----------------------------------------------------------------------------

MAX_DECISIONS_BYTES = 64 * 1024 * 1024


def _write_decisions_text(review_dir, text):
    """Validate a decisions CSV posted by the page and replace decisions.csv."""
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or set(DECISION_COLUMNS) - set(rows[0]):
        raise ValueError(f'decisions must be a CSV with columns {", ".join(DECISION_COLUMNS)}')
    if any(row['decision'] not in ('reject', 'keep') for row in rows):
        raise ValueError('decision must be "reject" or "keep"')
    target = Path(review_dir) / DECISIONS_CSV
    with NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as temporary:
        temporary.write(text)
    os.replace(temporary.name, target)
    return len(rows), sum(row['decision'] == 'reject' for row in rows)


def _merge_review_masks(editable, sources, targets):
    """Add the painted pixels to each destination and report completed writes."""
    if not sources or not all(key in editable for key in [*sources, *targets]):
        raise ValueError('Select a painted mask and frames from this review.')
    shape = editable[sources[0]][1]
    if any(editable[key][1] != shape for key in [*sources, *targets]):
        raise ValueError('All masks must use the same pixel grid.')
    brush = np.zeros(shape, dtype=bool)
    for key in sources:
        brush |= read_mask(*editable[key])
    if not brush.any():
        raise ValueError('The source mask is empty.')

    applied = []
    for key in targets:
        path, _ = editable[key]
        try:
            pixels = read_mask(path, shape) | brush if path.is_file() else brush
            write_mask(path, pixels)
        except (OSError, ValueError) as error:
            return {'applied': applied, 'error': f'{key}: {error}'}
        applied.append(key)
    return {'applied': applied}


def make_review_server(review_dir, *, port=8765):
    """HTTP server on localhost for a review directory, and the page URL.

    The dashboard saves decisions, saturation selections, and per-frame CME masks
    in the review directory. Mask edits are limited to files listed in the scan.
    """
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Lock

    review_dir = Path(review_dir).resolve()
    if not (review_dir / FRAMES_CSV).is_file():
        raise SystemExit(f'{review_dir} has no {FRAMES_CSV}; run "scan" first.')
    build_page(review_dir)

    mask_dir = review_dir / 'masks'
    mask_lock = Lock()
    editable = {f['mask_file']: (mask_dir / f['mask_file'], (f['height'], f['width']))
                for f in load_frames(review_dir) if f.get('preview') and f.get('mask_file')}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            name = self.path.split('?')[0].lstrip('/')
            if name.startswith('mask-preview/'):
                key = unquote(name.removeprefix('mask-preview/'))
                if key not in editable or not editable[key][0].is_file():
                    self.send_error(404)
                    return
                image = preview_png(read_mask(*editable[key]))
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(image)))
                self.end_headers()
                self.wfile.write(image)
                return
            # A reload must see masks added or cleared since the server started.
            if editable and name in ('', PAGE):
                build_page(review_dir)
            super().do_GET()

        def do_POST(self):
            name = self.path.split('?')[0].lstrip('/')
            if name.startswith('pixel-mask/'):
                key = unquote(name.removeprefix('pixel-mask/'))
                if key not in editable:
                    self.send_error(404)
                    return
                size = int(self.headers.get('Content-Length') or 0)
                if size > MAX_DECISIONS_BYTES or size < 0:
                    self.send_error(413)
                    return
                path, shape = editable[key]
                if size:
                    try:
                        excluded = decode_brush(self.rfile.read(size), shape)
                    except (OSError, ValueError) as error:
                        self.send_error(400, str(error))
                        return
                    with mask_lock:
                        write_mask(path, excluded)
                else:
                    with mask_lock:
                        path.unlink(missing_ok=True)
                self.send_response(204)
                self.end_headers()
                return
            if name not in (DECISIONS_CSV, SATURATION_CSV, 'merge-masks'):
                self.send_error(404)
                return
            length = int(self.headers.get('Content-Length') or 0)
            if length <= 0 or length > MAX_DECISIONS_BYTES:
                self.send_error(413)
                return
            try:
                text = self.rfile.read(length).decode('utf-8')
                if name == 'merge-masks':
                    request = json.loads(text)
                    with mask_lock:
                        result = _merge_review_masks(editable, request['sources'], request['targets'])
                elif name == DECISIONS_CSV:
                    total, rejected = _write_decisions_text(review_dir, text)
                    result = {'frames': total, 'rejected': rejected}
                else:
                    total, selected = _write_saturation_text(review_dir, text)
                    result = {'channels': total, 'selected_dn': selected}
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send_error(400, str(error))
                return
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def end_headers(self):
            self.send_header('Cache-Control', 'no-store')
            super().end_headers()

        def log_message(self, format, *args):
            if self.command == 'POST':
                super().log_message(format, *args)

    handler = functools.partial(Handler, directory=str(review_dir))
    server = None
    for candidate in range(port, port + 20):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', candidate), handler)
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f'No free port in {port}..{port + 19}.')
    return server, f'http://127.0.0.1:{server.server_address[1]}/{PAGE}'


def serve_review(review_dir, *, port=8765, open_browser=True):
    """Serve the review page on localhost; the page saves to decisions.csv."""
    import webbrowser

    review_dir = Path(review_dir).resolve()
    server, url = make_review_server(review_dir, port=port)
    print(f'Reviewing {review_dir}\n  {url}\n'
          f'Decisions are saved to {review_dir / DECISIONS_CSV}. Stop with Ctrl-C.')
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ----------------------------------------------------------------------------
# Apply / restore
# ----------------------------------------------------------------------------

def _apply_review_masks(directory, dry_run):
    """Mirror only this scan's masks beside their prepared FITS, including explicit clears."""
    mask_dir = directory / 'masks'
    if not mask_dir.is_dir():
        return
    for frame in load_frames(directory):
        if not frame.get('preview'):
            continue
        source = Path(frame['path'])
        target = mask_path(source)
        path = mask_dir / frame['mask_file']
        if not path.is_file():
            if target.is_file():
                print(f'{"would clear" if dry_run else "clearing"} {target}')
                if not dry_run:
                    target.unlink(missing_ok=True)
            continue
        actual = source if source.is_file() else source.parent / REJECTED_DIRECTORY / source.name
        with fits.open(actual, memmap=False) as hdul:
            image, _ = _image_hdu(hdul)
            excluded = read_mask(path, image.shape)
        print(f'{"would mask" if dry_run else "masking"} {excluded.sum()} pixels of {source}')
        if not dry_run:
            write_mask(target, excluded)


def apply_rejections(list_path, *, dry_run=False):
    """Move every rejected file into a ``rejected/`` directory beside it."""
    moved, missing = 0, 0
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    for entry in read_decisions(list_path):
        source = Path(entry)
        if not source.is_file():
            already = source.parent / REJECTED_DIRECTORY / source.name
            print(f'{"already rejected" if already.is_file() else "missing"}: {source}')
            missing += 1
            continue
        target_dir = source.parent / REJECTED_DIRECTORY
        target = target_dir / source.name
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite rejected file {target}')
        if dry_run:
            print(f'would move {source} -> {target}')
        else:
            target_dir.mkdir(exist_ok=True)
            shutil.move(str(source), str(target))
            with open(target_dir / REJECTED_LOG, 'a') as log:
                log.write(f'{stamp}\t{source.name}\t{os.path.abspath(list_path)}\n')
        moved += 1
    verb = 'Would move' if dry_run else 'Moved'
    print(f'{verb} {moved} files into {REJECTED_DIRECTORY}/ ({missing} not found).')
    return moved


def find_review_dirs(root):
    """Review directories (holding ``frames.csv``) at or below ``root``."""
    root = Path(root)
    return sorted({path.parent for path in root.rglob(FRAMES_CSV)})


def apply_review(target, *, dry_run=False):
    """Apply every review directory at or below ``target``.

    Each directory applies its ``decisions.csv`` when the review saved one, and
    otherwise the automatic flags of its ``frames.csv``. A file ``target`` (a
    decisions or frames CSV, or a list of paths) is applied directly.
    """
    target = Path(target)
    if target.is_file():
        return apply_rejections(target, dry_run=dry_run)
    if not target.is_dir():
        raise SystemExit(f'{target} does not exist.')
    directories = find_review_dirs(target)
    if not directories:
        raise SystemExit(f'No review directory ({FRAMES_CSV}) at or below {target}.')
    total = 0
    for directory in directories:
        decisions, frames = directory / DECISIONS_CSV, directory / FRAMES_CSV
        if decisions.is_file():
            source, label = decisions, 'reviewed decisions'
            if frames.stat().st_mtime > decisions.stat().st_mtime:
                label += f'; WARNING: {FRAMES_CSV} is newer (rescanned after the review?)'
        else:
            source, label = frames, 'automatic flags, not reviewed'
        print(f'== {directory}: {source.name} ({label})')
        _apply_review_masks(directory, dry_run)
        total += apply_rejections(source, dry_run=dry_run)
    verb = 'Would move' if dry_run else 'Moved'
    print(f'{verb} {total} files from {len(directories)} review directories.')
    return total


def restore(directory, *, dry_run=False):
    """Move every file of ``directory/rejected`` back into ``directory``."""
    rejected_dir = Path(directory) / REJECTED_DIRECTORY
    if not rejected_dir.is_dir():
        print(f'No {REJECTED_DIRECTORY}/ directory in {directory}.')
        return 0
    restored = 0
    for path in sorted(rejected_dir.iterdir()):
        if path.name == REJECTED_LOG or not path.is_file():
            continue
        target = Path(directory) / path.name
        if target.exists():
            print(f'skipped, exists: {target}')
            continue
        if dry_run:
            print(f'would restore {path} -> {target}')
        else:
            shutil.move(str(path), str(target))
        restored += 1
    print(f'{"Would restore" if dry_run else "Restored"} {restored} files to {directory}.')
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)

    scan_parser = commands.add_parser(
        'scan', help='on the server: statistics, flags and JPEG previews of a FITS sequence')
    scan_parser.add_argument('patterns', nargs='+', help='prepared image directories, FITS files, or quoted FITS globs')
    scan_parser.add_argument('--output', required=True, help='review directory to write')
    scan_parser.add_argument('--source-dir', default=None,
                             help='directory of the files the inputs were prepared from (header SRCFILE)')
    scan_parser.add_argument('--saturation-dn', type=float, default=None,
                             help='detector saturation in DN when not AIA or SECCHI')
    scan_parser.add_argument('--cme', action='store_true',
                             help='review prepared coronagraph brightness, with native-grid mask editing')
    scan_parser.add_argument('--workers', type=int, default=None)
    scan_parser.add_argument('--title', default=None)

    review_parser = commands.add_parser(
        'review', help='locally: open the review page of a copied review directory')
    review_parser.add_argument('review_dir')
    review_parser.add_argument('--port', type=int, default=8765)
    review_parser.add_argument('--no-browser', action='store_true')

    apply_parser = commands.add_parser(
        'apply', help='on the server: move the rejected files of every review directory aside')
    apply_parser.add_argument(
        'target', help='a review directory or a folder of them: each applies its '
                       'decisions.csv, or frames.csv (automatic flags) when it has none; '
                       'a single CSV or path list is applied directly')
    apply_parser.add_argument('--dry-run', action='store_true')

    restore_parser = commands.add_parser('restore', help='move rejected files back')
    restore_parser.add_argument('directory', help='directory that holds the rejected/ subdirectory')
    restore_parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args(argv)
    if args.command == 'scan':
        scan(args.patterns, args.output, source_dir=args.source_dir, workers=args.workers,
             saturation_dn=args.saturation_dn, title=args.title, cme=args.cme)
    elif args.command == 'review':
        serve_review(args.review_dir, port=args.port, open_browser=not args.no_browser)
    elif args.command == 'apply':
        apply_review(args.target, dry_run=args.dry_run)
    else:
        restore(args.directory, dry_run=args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
