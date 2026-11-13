"""Overview of the coronagraph observations in a time period, one image per frame.

Every instrument writes to its own folder
``<out_path>/<instrument>/<instrument>_<product>_<time>.png`` (tB of PUNCH and CCOR, pB of
STEREO-A COR2).  PUNCH and CCOR are shown as raw data products, not the prepared training data:
PUNCH L3 PAM (B plane) and CCOR L2, without quality filter, clipping of negative values or
cropping; only pixels the product marks as without data are left out (CCOR mask extension,
PUNCH zero uncertainty).  COR2 is shown as the prepared pB of the training.  Each image is
resampled to at most ``RESOLUTION`` pixels and rotated to solar north up, on the image plane in
projected solar radii, with white axes and labels on a transparent background, in the colormap
of the instrument: PUNCH (punchbowl, black-orange-white in LAB), STEREO COR2 (LASCO C2), CCOR
(Blues_r, NOAA CCOR ATBD).

The color scale is fixed per instrument for all frames: asinh with the linear width of the
training scaling of the instrument, from zero (negative pixels in the lowest color) to the 99.9th
percentile of the pixels of ``N_NORM_FRAMES`` frames sampled from the period.  Reading, sampling and plotting run in
``--num_workers`` processes.
"""
import argparse
import glob
import multiprocessing
import os
from functools import partial

import matplotlib
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.ticker import MaxNLocator
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.coronagraph.prep_ccor import _load_ccor_header, _rotate_ccor_map
from sunerf.evaluation.cme.plot_ref_series import normalize_datetime, read_observation_date

# display name, product, colormap, linear width of the asinh scale in MSB (scaling of the training: a * 1e-9)
INSTRUMENTS = {'punch': ('PUNCH', 'tB', cm.punch, 1e-14),
               'ccor': ('CCOR', 'tB', plt.get_cmap('Blues_r'), 1e-12),
               'cor2': ('STEREO-A COR2', 'pB', cm.soholasco2, 1e-12)}
N_NORM_FRAMES = 10
# pixels along the long image axis after resampling
RESOLUTION = 1024
STYLE = {'text.color': 'white', 'axes.edgecolor': 'white', 'axes.labelcolor': 'white',
         'xtick.color': 'white', 'ytick.color': 'white',
         'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16}


def list_files(pattern, time_range, label, pool):
    """Files of ``pattern`` observed in ``time_range`` and their observation times, sorted by time."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f'No files found for {pattern}')
    start, end = time_range
    times = tqdm(pool.imap(read_observation_date, paths, chunksize=16), total=len(paths),
                 desc=f'{label}: reading observation times')
    files = sorted((time, path) for time, path in zip(times, paths) if start <= time <= end)
    if not files:
        raise ValueError(f'No files of {pattern} in {start} - {end}')
    return files


def load_raw(name, path):
    """Raw data product of one file as a map; only pixels the product marks as without data are NaN."""
    if name == 'punch':
        # L3 PAM planes B, pB, pB'; pixels without data have zero uncertainty in B and pB
        data = np.array(fits.getdata(path, 1)[0], dtype=np.float64)
        uncertainty = fits.getdata(path, 2)
        data[(uncertainty[0] == 0) & (uncertainty[1] == 0)] = np.nan
        return Map(data, fits.getheader(path, 1))
    if name == 'ccor':
        # L2 image with the mask of the product, any recorded 90-degree rotation undone
        data = np.array(fits.getdata(path, 1), dtype=np.float64)
        data[fits.getdata(path, 2) != 0] = np.nan
        return _rotate_ccor_map(Map(data, _load_ccor_header(path)), fits.getheader(path, 0))
    return Map(path)


def projected_image(name, path):
    """Raw image of one file, solar north up, and its extent on the image plane in projected solar radii;
    None for a file whose header gives no valid rotation (singular PC/CD matrix)."""
    s_map = load_raw(name, path)
    rotation = np.asarray(s_map.rotation_matrix, dtype=float)
    if not np.all(np.isfinite(rotation)) or abs(np.linalg.det(rotation)) < 1e-6:
        print(f'Skipping {path}: singular rotation matrix {rotation.tolist()}', flush=True)
        return None
    # resampled before the rotation, which is costly at the full resolution
    factor = min(RESOLUTION / max(s_map.data.shape), 1.0)
    s_map = s_map.resample(np.round(np.array(s_map.data.shape[::-1]) * factor) * u.pixel)
    s_map = s_map.rotate(order=1, missing=np.nan)
    # pixel offsets from Sun center times the observer distance
    rsun_per_pixel = [float(scale.to_value(u.rad / u.pix)) * float(s_map.dsun.to_value(u.R_sun))
                      for scale in s_map.scale]
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=s_map.coordinate_frame)
    center = [float(value) for value in s_map.wcs.world_to_pixel(sun_center)]
    ny, nx = s_map.data.shape
    extent = np.array([(-0.5 - center[0]) * rsun_per_pixel[0], (nx - 0.5 - center[0]) * rsun_per_pixel[0],
                       (-0.5 - center[1]) * rsun_per_pixel[1], (ny - 0.5 - center[1]) * rsun_per_pixel[1]])
    return np.array(s_map.data, dtype=np.float32), extent


def observed_half_size(image, extent):
    """Half width of the square window around the Sun that contains all finite pixels."""
    ny, nx = image.shape
    x = np.linspace(extent[0], extent[1], nx + 1)
    x = 0.5 * (x[1:] + x[:-1])
    y = np.linspace(extent[2], extent[3], ny + 1)
    y = 0.5 * (y[1:] + y[:-1])
    rows, columns = np.nonzero(np.isfinite(image))
    return max(np.abs(x[columns]).max(), np.abs(y[rows]).max())


def sample_frame(path, name):
    """Finite pixel values and window half width of one frame; None for a skipped file."""
    projected = projected_image(name, path)
    if projected is None:
        return None
    image, extent = projected
    return image[np.isfinite(image)], observed_half_size(image, extent)


def display_settings(files, name, linear_width, label, pool):
    """Color scale and window half width shared by all frames, from frames sampled over the period."""
    indices = np.unique(np.linspace(0, len(files) - 1, min(N_NORM_FRAMES, len(files))).astype(int))
    paths = [files[index][1] for index in indices]
    samples = list(tqdm(pool.imap(partial(sample_frame, name=name), paths), total=len(paths), desc=f'{label}: color scale'))
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        raise ValueError(f'No valid {label} frame among the sampled files')
    values, half_sizes = zip(*samples)
    # negative pixels are kept and shown in the lowest color
    norm = AsinhNorm(linear_width=linear_width, vmin=0.0, vmax=np.percentile(np.concatenate(values), 99.9))
    return norm, max(half_sizes)


def plot_frame(image, extent, cmap, norm, half_size, out_file):
    """One frame; pixels without data stay transparent."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.imshow(image, origin='lower', cmap=cmap, norm=norm, extent=extent)
        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel(r'x [R$_\odot$]')
        ax.set_ylabel(r'y [R$_\odot$]')
        fig.savefig(out_file, dpi=300, bbox_inches='tight', transparent=True)
        plt.close(fig)


def plot_file(file, name, cmap, norm, half_size, instrument_path):
    """Plot one ``(time, path)`` frame of an instrument; False for a skipped file."""
    time, path = file
    product = INSTRUMENTS[name][1]
    projected = projected_image(name, path)
    if projected is None:
        return False
    out_file = os.path.join(instrument_path, f'{name}_{product}_{time:%Y%m%dT%H%M%S}.png')
    plot_frame(*projected, cmap, norm, half_size, out_file)
    return True


def plot_instrument(name, pattern, time_range, out_path, pool):
    """All frames of one instrument in the period."""
    label, _, cmap, linear_width = INSTRUMENTS[name]
    instrument_path = os.path.join(out_path, name)
    os.makedirs(instrument_path, exist_ok=True)
    files = list_files(pattern, time_range, label, pool)
    norm, half_size = display_settings(files, name, linear_width, label, pool)
    plot = partial(plot_file, name=name, cmap=cmap, norm=norm, half_size=half_size, instrument_path=instrument_path)
    written = sum(tqdm(pool.imap_unordered(plot, files), total=len(files), desc=f'{label}: plotting frames'))
    print(f'{written} {label} frames written to {instrument_path}, {len(files) - written} skipped', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_path', type=str, required=True, help='Output directory, one folder per instrument')
    parser.add_argument('--time_range', type=normalize_datetime, nargs=2, required=True, metavar=('START', 'END'),
                        help='Period of the plotted frames')
    parser.add_argument('--punch_tB', type=str, default=None, help='Glob of the PUNCH L3 PAM files')
    parser.add_argument('--ccor_tB', type=str, default=None, help='Glob of the CCOR L2 files')
    parser.add_argument('--cor2_pB', type=str, default=None, help='Glob of the prepared COR2 pB files')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()

    inputs = {'punch': args.punch_tB, 'ccor': args.ccor_tB, 'cor2': args.cor2_pB}
    if all(pattern is None for pattern in inputs.values()):
        raise ValueError('Provide the files of at least one instrument')
    with multiprocessing.Pool(args.num_workers) as pool:
        for name, pattern in inputs.items():
            if pattern is not None:
                plot_instrument(name, pattern, args.time_range, args.out_path, pool)


if __name__ == '__main__':
    main()
