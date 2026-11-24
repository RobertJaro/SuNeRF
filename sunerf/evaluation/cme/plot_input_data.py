"""Overview figure of the prepared training data: one row per observing sequence, ``N_FRAMES`` columns.

The columns are ``N_FRAMES`` times evenly spaced over ``--time_range``; every row shows the frame of
its sequence closest to each time (observation time in the panel title).  Rows: PUNCH pB, STEREO-A
COR2 clear tB, CCOR tB, LASCO C2 tB and LASCO C3 tB, in the colormaps of ``plot_data_overview``;
COR2 clear and LASCO optionally minus their correction masks, as in training.  Images are shown in units of the training scaling
(1e-9 MSB), solar north up on the image plane in projected solar radii, with one colorbar per
row: asinh with the linear width ``a`` of the training scaling of the instrument (1e-5 for PUNCH,
1e-3 otherwise), from zero (negative pixels in the lowest color) to the 99.9th percentile of the
pixels of the row.
"""
import argparse
import glob
import multiprocessing
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.review_masks import load_reviewed_map
from sunerf.evaluation.cme.plot_data_overview import observed_half_size, project_map
from sunerf.evaluation.cme.plot_ref_series import normalize_datetime, read_observation_date

N_FRAMES = 5
# image scaling of the coronagraph datasets (MSB)
SCALING = 1.0e-9
# key: display name, product, colormap, linear width of the asinh scale (training scaling ``a``)
ROWS = {'punch': ('PUNCH', 'pB', cm.punch, 1e-5),
        'cor2_clear': ('STEREO-A COR2 clear', 'tB', cm.soholasco2, 1e-3),
        'ccor': ('CCOR', 'tB', plt.get_cmap('Blues_r'), 1e-3),
        'lasco_c2': ('LASCO C2', 'tB', cm.soholasco2, 1e-3),
        'lasco_c3': ('LASCO C3', 'tB', cm.soholasco3, 1e-3)}


def select_frames(pattern, time_range, label, pool):
    """``(time, path)`` of the file closest to each of the ``N_FRAMES`` evenly spaced times."""
    # files only: a rejected/ directory of moved-aside frames matches "tB/*"
    paths = sorted(path for path in glob.glob(pattern) if os.path.isfile(path))
    if not paths:
        raise ValueError(f'No files found for {pattern}')
    times = list(tqdm(pool.imap(read_observation_date, paths, chunksize=16), total=len(paths),
                      desc=f'{label}: reading observation times'))
    start, end = time_range
    files = sorted((time, path) for time, path in zip(times, paths) if start <= time <= end)
    if not files:
        raise ValueError(f'No files of {pattern} in {start} - {end}')
    file_times = np.array([(time - start).total_seconds() for time, _ in files])
    targets = np.linspace(0, (end - start).total_seconds(), N_FRAMES)
    return [files[int(np.argmin(np.abs(file_times - target)))] for target in targets]


def load_frame(task):
    """Image of one prepared file in units of the training scaling, minus the correction mask if given."""
    path, correction_path = task
    s_map = load_reviewed_map(path)
    data = np.array(s_map.data, dtype=np.float64)
    if correction_path is not None:
        data = data - np.load(correction_path)
    return project_map(Map(data / SCALING, s_map.meta), path)


def plot_row(axes, frames, images, name, fig):
    """One observing sequence with a shared color scale and window, colorbar on the right."""
    label, product, cmap, linear_width = ROWS[name]
    valid = [(frame, image) for frame, image in zip(frames, images) if image is not None]
    if not valid:
        raise ValueError(f'No valid {label} frame')
    values = np.concatenate([image[np.isfinite(image)] for _, (image, _) in valid])
    norm = AsinhNorm(linear_width=linear_width, vmin=0.0, vmax=np.percentile(values, 99.9))
    half_size = max(observed_half_size(image, extent) for _, (image, extent) in valid)
    cmap = cmap.copy()
    cmap.set_bad('black')
    for index, (ax, (time, _), projected) in enumerate(zip(axes, frames, images)):
        ax.set_facecolor('black')
        # solar disk to scale (1 R_sun on the image plane)
        ax.add_patch(Circle((0, 0), 1.0, color='yellow', zorder=3))
        ax.text(0.04, 0.96, f'{time:%Y-%m-%d %H:%M}', transform=ax.transAxes, color='white',
                fontsize=7, ha='left', va='top', zorder=4)
        ax.set_xlim(-half_size, half_size)
        ax.set_ylim(-half_size, half_size)
        ax.set_aspect('equal')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(labelsize=7)
        if index == 0:
            ax.set_ylabel(f'{label}\n' + r'y [R$_\odot$]', fontsize=8)
        else:
            ax.tick_params(labelleft=False)
        if projected is None:
            continue
        im = ax.imshow(projected[0], origin='lower', cmap=cmap, norm=norm, extent=projected[1])
    cbar = fig.colorbar(im, ax=list(axes), location='right', pad=0.01, shrink=0.95)
    cbar.set_label(f'{product} [10$^{{-9}}$ MSB]', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # ticks inside the linear range of the asinh scale overlap the zero tick
    cbar.set_ticks([tick for tick in cbar.get_ticks() if tick == 0 or tick >= 10 * linear_width])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_file', type=str, required=True, help='Output figure')
    parser.add_argument('--time_range', type=normalize_datetime, nargs=2, required=True, metavar=('START', 'END'),
                        help='Period of the plotted frames')
    parser.add_argument('--punch_pB', type=str, default=None, help='Glob of the prepared PUNCH pB files')
    parser.add_argument('--cor2_clear_tB', type=str, default=None, help='Glob of the prepared COR2 clear tB files')
    parser.add_argument('--cor2_clear_correction', type=str, default=None,
                        help='Correction mask (.npy) subtracted from the COR2 clear tB')
    parser.add_argument('--ccor_tB', type=str, default=None, help='Glob of the prepared CCOR tB files')
    parser.add_argument('--lasco_c2_tB', type=str, default=None, help='Glob of the prepared LASCO C2 tB files')
    parser.add_argument('--lasco_c2_correction', type=str, default=None,
                        help='Correction mask (.npy) subtracted from the LASCO C2 tB')
    parser.add_argument('--lasco_c3_tB', type=str, default=None, help='Glob of the prepared LASCO C3 tB files')
    parser.add_argument('--lasco_c3_correction', type=str, default=None,
                        help='Correction mask (.npy) subtracted from the LASCO C3 tB')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()

    inputs = {'punch': (args.punch_pB, None), 'cor2_clear': (args.cor2_clear_tB, args.cor2_clear_correction),
              'ccor': (args.ccor_tB, None), 'lasco_c2': (args.lasco_c2_tB, args.lasco_c2_correction),
              'lasco_c3': (args.lasco_c3_tB, args.lasco_c3_correction)}
    inputs = {name: value for name, value in inputs.items() if value[0] is not None}
    if not inputs:
        raise ValueError('Provide the files of at least one observing sequence')

    with multiprocessing.Pool(args.num_workers) as pool:
        rows = {}
        for name, (pattern, correction) in inputs.items():
            frames = select_frames(pattern, args.time_range, ROWS[name][0], pool)
            images = pool.map(load_frame, [(path, correction) for _, path in frames])
            rows[name] = (frames, images)

    fig, axes = plt.subplots(len(rows), N_FRAMES, figsize=(1.6 * N_FRAMES + 1.2, 1.6 * len(rows)),
                             layout='constrained', squeeze=False)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
    for row_axes, (name, (frames, images)) in zip(axes, rows.items()):
        plot_row(row_axes, frames, images, name, fig)
    for ax in axes[-1]:
        ax.set_xlabel(r'x [R$_\odot$]', fontsize=8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    fig.savefig(args.out_file, dpi=300)
    plt.close(fig)
    print(f'Saved {args.out_file}', flush=True)


if __name__ == '__main__':
    main()
