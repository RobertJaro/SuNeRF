"""Video of radially integrated emission measure maps of an EUV plasma reconstruction.

A fixed equal-area Carrington grid (longitude by sin(latitude)) is built
first; at every frame time its cells are sampled along the radius, and the
emission measure n_e^2 dr is integrated per temperature bin (local temperature
inside the bin) and the electron density n_e dr over the whole shell. Every
frame shows one map per temperature bin and the radial column density in
longitude and sin(latitude), between a start and end time at a fixed cadence.
"""
import argparse
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from sunerf.evaluation.loader import PlasmaSuNeRFLoader
from sunerf.evaluation.plasma.radial import integrate_columns, radial_columns, rendered_shell
from sunerf.evaluation.plasma.video import parse_time

EM_CMAP = 'magma'
DENSITY_CMAP = 'viridis'
# percentiles of the first frame that fix the color scales of the video
EM_PERCENTILES = (1.0, 99.9)
DENSITY_PERCENTILES = (1.0, 99.9)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Video of radially integrated emission measure per temperature bin on a fixed '
                    'latitude-longitude grid.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to the plasma save state')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output directory of the frames; default: <run directory>/video_em_maps')
    parser.add_argument('--time_start', type=parse_time, default=None,
                        help='First frame time (ISO); default: start of the training data')
    parser.add_argument('--time_end', type=parse_time, default=None,
                        help='Last frame time (ISO); default: end of the training data')
    parser.add_argument('--cadence_minutes', type=float, default=60.0, help='Time between frames')
    parser.add_argument('--radius_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Radial integration range in solar radii; default: the rendered shell '
                             '(sampling.min_distance to max_distance)')
    parser.add_argument('--n_radius', type=int, default=256, help='Radial samples of the integration')
    parser.add_argument('--n_lat', type=int, default=180, help='Grid rows, uniform in sin(latitude)')
    parser.add_argument('--n_lon', type=int, default=360, help='Grid columns, uniform in Carrington longitude')
    parser.add_argument('--em_bins', type=int, default=3, help='Number of temperature bins')
    parser.add_argument('--em_log_T_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='log10 K range split into --em_bins; default: the diagnostic temperature grid')
    parser.add_argument('--em_edges', type=float, nargs='+', default=None,
                        help='Explicit bin edges in log10 K (overrides --em_bins and --em_log_T_range)')
    parser.add_argument('--em_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the emission measure color scales [cm^-5]; default: percentiles '
                             'of the first frame, per bin')
    parser.add_argument('--density_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the column density color scale [cm^-2]; default: percentiles of '
                             'the first frame')
    parser.add_argument('--batch_size', type=int, default=65536, help='Points per model batch')
    parser.add_argument('--dpi', type=int, default=150, help='Saved frame DPI')
    parser.add_argument('--fps', type=float, default=10.0, help='Frame rate of the encoded video')
    parser.add_argument('--frames_only', action='store_true', help='Write the frames without encoding a video')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames even if output files exist')
    return parser


def temperature_edges(loader, args):
    if args.em_edges is not None:
        edges = np.asarray(args.em_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError('--em_edges must be at least two increasing values')
        return edges
    if args.em_log_T_range is None:
        log_T = np.asarray(loader.log_T_range, dtype=float)
        args.em_log_T_range = (float(log_T.min()), float(log_T.max()))
    return np.linspace(args.em_log_T_range[0], args.em_log_T_range[1], args.em_bins + 1)


def frame_range(data, percentiles):
    values = data[np.isfinite(data) & (data > 0)]
    if values.size == 0:
        return 1.0, 10.0
    low, high = np.percentile(values, percentiles)
    return (low, high) if high > low else (values.min(), values.max() * 10)


def build_grid(n_lat, n_lon, radii):
    """Sample points of an equal-area map: cell centers in sin(latitude) and Carrington longitude, each with a
    radial column. Returns the sin(latitude) and longitude axes [deg] and the Cartesian points
    (lat, lon, radius, 3) in solar radii."""
    sin_lat = np.linspace(-1.0, 1.0, n_lat + 1)
    sin_lat = 0.5 * (sin_lat[:-1] + sin_lat[1:])
    lon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(np.arcsin(sin_lat), np.deg2rad(lon), indexing='ij')
    return sin_lat, lon, radial_columns(lat_grid, lon_grid, radii)


def main():
    args = build_parser().parse_args()
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'video_em_maps')
    os.makedirs(args.out_path, exist_ok=True)

    loader = PlasmaSuNeRFLoader(args.sunerf_path)
    if args.radius_range is None:
        args.radius_range = rendered_shell(loader)
    edges = temperature_edges(loader, args)
    times = np.concatenate([np.asarray(loader.times(k)) for k in loader.ds_keys])
    start = pd.Timestamp(times.min()).to_pydatetime() if args.time_start is None else args.time_start
    end = pd.Timestamp(times.max()).to_pydatetime() if args.time_end is None else args.time_end
    frame_times = pd.date_range(start, end, freq=pd.Timedelta(minutes=args.cadence_minutes)).to_pydatetime()

    radii = np.linspace(args.radius_range[0], args.radius_range[1], args.n_radius)
    sin_lat, lon, points = build_grid(args.n_lat, args.n_lon, radii)
    extent = [0.0, 360.0, -1.0, 1.0]
    lat_ticks = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])

    em_norms = [LogNorm(*args.em_range) if args.em_range else None for _ in range(len(edges) - 1)]
    density_norm = LogNorm(*args.density_range) if args.density_range else None
    n_cols = len(edges)

    frame_paths = []
    for i, time in tqdm(enumerate(frame_times), total=len(frame_times)):
        frame_path = os.path.join(args.out_path, f'em_maps_{i:03d}.jpg')
        frame_paths.append(frame_path)
        if os.path.exists(frame_path) and not args.overwrite:
            continue
        em_maps, column_density = integrate_columns(loader, points, radii, time, list(zip(edges[:-1], edges[1:])),
                                                    args.batch_size)
        for j, em_map in enumerate(em_maps):
            if em_norms[j] is None:
                em_norms[j] = LogNorm(*frame_range(em_map, EM_PERCENTILES))
        if density_norm is None:
            density_norm = LogNorm(*frame_range(column_density, DENSITY_PERCENTILES))

        fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.2), constrained_layout=True)
        panels = [(em_map, em_norms[j], EM_CMAP,
                   rf'EM, log$_{{10}}$T in [{edges[j]:.2f}, {edges[j + 1]:.2f}) [cm$^{{-5}}$]')
                  for j, em_map in enumerate(em_maps)]
        panels.append((column_density, density_norm, DENSITY_CMAP, r'Column $N_e$ [cm$^{-2}$]'))
        for ax, (data, norm, cmap, title) in zip(axes, panels):
            colormap = plt.get_cmap(cmap).copy()
            colormap.set_bad((0.0, 0.0, 0.0, 0.0))
            im = ax.imshow(data, origin='lower', extent=extent, cmap=colormap, norm=norm, aspect='auto')
            colorbar = fig.colorbar(im, ax=ax, location='top', aspect=30, pad=0.02)
            colorbar.set_label(title)
            ax.set_xlabel('Carrington longitude [deg]')
            ax.set_yticks(np.sin(np.deg2rad(lat_ticks)))
            ax.set_yticklabels([f'{v:.0f}' for v in lat_ticks])
        axes[0].set_ylabel('Latitude [deg], linear in sin(latitude)')
        for ax in axes[1:]:
            ax.tick_params(labelleft=False)
        fig.suptitle(
            f'{time.isoformat(" ", timespec="minutes")}, '
            f'r = {args.radius_range[0]:.3f} .. {args.radius_range[1]:.3f} R$_\\odot$',
            fontsize=13,
        )
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close(fig)

    if not args.frames_only:
        from sunerf.evaluation.image_sequence_to_video import write_video

        video_path = args.out_path.rstrip('/') + '.mp4'
        write_video(frame_paths, video_path, args.fps)
        print(f'Wrote {len(frame_paths)} frames to {args.out_path} and the video to {video_path}')


if __name__ == '__main__':
    main()
