"""Alignment check: PUNCH, LASCO C3 and LASCO C2 reprojected into the LASCO C3 frame and stacked.

For each instrument, the prepared file closest to ``--time`` is used, LASCO optionally minus its correction
mask (in its native frame, as in training). All maps are reprojected into the helioprojective frame of the
C3 map (observer and roll from its header, Sun center at the canvas center), on a canvas that covers the full
PUNCH field of view at the PUNCH pixel scale; off-disk pixels on a spherical screen centered at the C3
observer. Each layer is shown in the same colormap on a log scale between its own 1st and 99.5th percentile,
stacked opaque from bottom to top: PUNCH, C3, C2 (pixels without data show the layer below).
"""
import argparse
import glob
import multiprocessing
import os

import matplotlib
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sunpy.coordinates import SphericalScreen
from sunpy.map import Map, make_fitswcs_header

from sunerf.data.review_masks import load_reviewed_map
from sunerf.evaluation.cme.plot_ref_series import normalize_datetime, read_observation_date

# bottom to top
LAYERS = {'punch': 'PUNCH', 'lasco_c3': 'LASCO C3', 'lasco_c2': 'LASCO C2'}
CMAP = 'gray'
# figure width and height (inches); the dpi is set for about one output pixel per canvas pixel
FIGURE_SIZE = 12.0


def closest_file(pattern, time, pool):
    paths = sorted(path for path in glob.glob(pattern) if os.path.isfile(path))
    if not paths:
        raise ValueError(f'No files found for {pattern}')
    times = pool.map(read_observation_date, paths, chunksize=16)
    index = int(np.argmin([abs((t - time).total_seconds()) for t in times]))
    return times[index], paths[index]


def log_normalized(image):
    """log10 of the positive pixels scaled to [0, 1] between the 1st and 99.5th percentile; NaN elsewhere."""
    image = np.where(image > 0, image, np.nan)
    log_image = np.log10(image)
    vmin, vmax = np.nanpercentile(log_image, [1, 99.5])
    return np.clip((log_image - vmin) / (vmax - vmin), 0, 1)


def canvas_map(c3_map, punch_map):
    """Empty map in the C3 frame centered on the Sun, covering the finite PUNCH pixels at the PUNCH scale."""
    scale = float(np.min(np.abs(u.Quantity(punch_map.scale).to_value(u.arcsec / u.pix))))
    sun_center = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=punch_map.coordinate_frame)
    cx, cy = (float(value) for value in punch_map.wcs.world_to_pixel(sun_center))
    rows, columns = np.nonzero(np.isfinite(punch_map.data))
    half_size = np.hypot(columns - cx, rows - cy).max() * scale
    n = int(np.ceil(2 * half_size / scale))
    reference = SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=c3_map.coordinate_frame)
    header = make_fitswcs_header((n, n), reference, scale=[scale, scale] * u.arcsec / u.pix,
                                 rotation_matrix=c3_map.rotation_matrix)
    return Map(np.zeros((n, n), dtype=np.float32), header)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_file', type=str, required=True, help='Output figure')
    parser.add_argument('--time', type=normalize_datetime, required=True, help='Compared observation time')
    parser.add_argument('--punch', type=str, required=True, help='Glob of the prepared PUNCH files')
    parser.add_argument('--lasco_c2', type=str, required=True, help='Glob of the prepared LASCO C2 files')
    parser.add_argument('--lasco_c3', type=str, required=True, help='Glob of the prepared LASCO C3 files')
    parser.add_argument('--lasco_c2_correction', type=str, default=None,
                        help='Correction mask (.npy) subtracted from the LASCO C2 tB')
    parser.add_argument('--lasco_c3_correction', type=str, default=None,
                        help='Correction mask (.npy) subtracted from the LASCO C3 tB')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()

    patterns = {'punch': args.punch, 'lasco_c3': args.lasco_c3, 'lasco_c2': args.lasco_c2}
    with multiprocessing.Pool(args.num_workers) as pool:
        files = {name: closest_file(pattern, args.time, pool) for name, pattern in patterns.items()}
    corrections = {'lasco_c2': args.lasco_c2_correction, 'lasco_c3': args.lasco_c3_correction}
    maps = {}
    for name, (time, path) in files.items():
        print(f'{LAYERS[name]}: {time:%Y-%m-%d %H:%M:%S} {path}', flush=True)
        s_map = load_reviewed_map(path)
        if corrections.get(name) is not None:
            # subtracted in the native frame, as in training
            s_map = Map(np.array(s_map.data, dtype=np.float64) - np.load(corrections[name]), s_map.meta)
        maps[name] = s_map

    target = canvas_map(maps['lasco_c3'], maps['punch'])
    print(f'Canvas: {target.data.shape[1]} x {target.data.shape[0]} pixels, {target.scale[0]:.1f}', flush=True)
    layers = {}
    for name, s_map in maps.items():
        with SphericalScreen(maps['lasco_c3'].observer_coordinate, only_off_disk=True):
            s_map = s_map.reproject_to(target.wcs)
        layers[name] = log_normalized(np.array(s_map.data, dtype=np.float64))

    fig = plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
    ax = fig.add_subplot(projection=target)
    ax.set_facecolor('black')
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad(alpha=0.0)
    for image in layers.values():
        ax.imshow(image, origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    target.draw_limb(axes=ax, color='yellow', lw=0.8)
    ax.coords.grid(color='white', alpha=0.4, ls=':', lw=0.5)
    ax.coords[0].set_axislabel('Helioprojective longitude')
    ax.coords[1].set_axislabel('Helioprojective latitude')
    ax.set_xlim(-0.5, target.data.shape[1] - 0.5)
    ax.set_ylim(-0.5, target.data.shape[0] - 0.5)
    label = '\n'.join(f'{LAYERS[name]} {time:%Y-%m-%d %H:%M}' for name, (time, _) in files.items())
    ax.text(0.01, 0.99, label, transform=ax.transAxes, color='white', fontsize=9, va='top', ha='left')
    ax.set_title('Reprojected into the LASCO C3 frame (bottom to top: PUNCH, C3, C2)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    # about one output pixel per canvas pixel
    dpi = max(100, int(np.ceil(target.data.shape[0] / (0.8 * FIGURE_SIZE))))
    fig.savefig(args.out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {args.out_file} ({dpi} dpi)', flush=True)


if __name__ == '__main__':
    main()
