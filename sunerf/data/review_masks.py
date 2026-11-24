"""Per-observation boolean NumPy masks. True excludes a pixel in FITS array order."""
import io
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image


def mask_path(source):
    source = Path(source)
    return source.parent / '.review-masks' / (source.name + '.npy')


def decode_brush(data, shape):
    """Translate the browser canvas alpha plane into native FITS coordinates."""
    with Image.open(io.BytesIO(data)) as image:
        if image.format != 'PNG' or image.size != tuple(reversed(shape)) or image.mode != 'RGBA':
            raise ValueError(f'expected an RGBA brush image for frame shape {shape}')
        alpha = np.asarray(image.getchannel('A'))
        if not np.all((alpha == 0) | (alpha == 255)) or not np.any(alpha):
            raise ValueError('brush must contain painted pixels and binary alpha values')
        return np.flipud(alpha != 0)


def preview_png(excluded):
    overlay = np.zeros((*excluded.shape, 4), dtype=np.uint8)
    overlay[np.flipud(excluded)] = (255, 73, 108, 255)
    output = io.BytesIO()
    Image.fromarray(overlay).save(output, format='PNG')
    return output.getvalue()


def write_mask(target, excluded):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=target.parent, delete=False)
    try:
        with temporary:
            np.save(temporary, excluded, allow_pickle=False)
        Path(temporary.name).replace(target)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def read_mask(path, shape):
    excluded = np.load(path, allow_pickle=False)
    if excluded.dtype != np.bool_ or excluded.shape != tuple(shape):
        raise ValueError(f'Expected a boolean mask {path} with image shape {shape}')
    return excluded


def apply_pixel_mask(source, data):
    path = mask_path(source)
    if not path.is_file():
        return data
    excluded = read_mask(path, data.shape)
    result = np.array(data, dtype=np.result_type(data.dtype, np.float32), copy=True)
    result[excluded] = np.nan
    return result


def load_reviewed_map(path):
    from sunpy.map import Map

    original = Map(path)
    data = apply_pixel_mask(path, original.data)
    return original if data is original.data else Map(data, original.meta)


def load_reviewed_fits(path):
    from astropy.io import fits

    return apply_pixel_mask(path, fits.getdata(path))
