import argparse
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

# Importing the SunPy colormap module registers instrument colormap names with
# Matplotlib. Channel selection itself comes exclusively from the artifact.
from sunpy.visualization import colormaps as _sunpy_colormaps  # noqa: F401

from sunerf.configuration import canonical_channel_id
from sunerf.evaluation.loader import PlasmaSuNeRFLoader


def _colorbar(fig, ax, image):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(image, cax=cax)


def _channel_cmap(channel):
    cmap = channel.get('cmap', 'gray')
    try:
        return plt.get_cmap(cmap)
    except ValueError:
        return plt.get_cmap('gray')


def _positive_lognorm(values, *, floor=None):
    finite_positive = np.asarray(values)[
        np.isfinite(values) & (np.asarray(values) > 0)
    ]
    if finite_positive.size == 0:
        return None
    vmin = float(finite_positive.min()) if floor is None else float(floor)
    vmax = float(finite_positive.max())
    if vmax <= vmin:
        return None
    return LogNorm(vmin=vmin, vmax=vmax)


def _trajectory(avg_time, n_points=20):
    return (
        list(zip(np.zeros(n_points), np.linspace(0, 360, n_points),
                 [avg_time] * n_points, np.ones(n_points)))
        + list(zip(np.linspace(0, 360, n_points), np.zeros(n_points),
                   [avg_time] * n_points, np.ones(n_points)))
        + list(zip(np.linspace(0, 45, n_points), np.linspace(0, 90, n_points),
                   [avg_time] * n_points, np.linspace(1, 0.7, n_points)))
        + list(zip(np.full(n_points, 45), np.linspace(90, 360, n_points),
                   [avg_time] * n_points, np.linspace(0.7, 1.0, n_points)))
    )


def _select_channel_metadata(channel_metadata, requested_channels=None):
    """Resolve user-facing channel aliases against immutable artifact IDs."""
    channel_metadata = tuple(dict(channel) for channel in channel_metadata)
    if requested_channels is None:
        return channel_metadata

    selected = []
    selected_ids = set()
    for requested in requested_channels:
        exact = [
            channel for channel in channel_metadata
            if str(channel['id']) == str(requested)
        ]
        matches = exact or [
            channel for channel in channel_metadata
            if canonical_channel_id(channel['id']) == canonical_channel_id(requested)
        ]
        if not matches:
            available = ', '.join(str(channel['id']) for channel in channel_metadata)
            raise ValueError(
                f"Unknown channel {requested!r}; artifact channels are: {available}."
            )
        if len(matches) > 1:
            candidates = ', '.join(str(channel['id']) for channel in matches)
            raise ValueError(
                f"Channel alias {requested!r} is ambiguous; matches: {candidates}."
            )
        channel = matches[0]
        channel_id = str(channel['id'])
        if channel_id in selected_ids:
            raise ValueError(f"Channel {channel_id!r} was selected more than once.")
        selected.append(channel)
        selected_ids.add(channel_id)
    if not selected:
        raise ValueError('At least one channel must be selected.')
    return tuple(selected)


def _write_video(frame_paths, video_file, fps):
    try:
        from sunerf.evaluation.image_sequence_to_video import write_video
    except ImportError as error:
        raise RuntimeError(
            "Video encoding requires the 'evaluation' optional dependencies: "
            "install SuNeRF with 'sunerf[evaluation]'."
        ) from error
    write_video(frame_paths, video_file, fps)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Create an artifact-driven plasma reconstruction video.'
    )
    parser.add_argument('--chk-path', '--chk_path', dest='chk_path', required=True)
    parser.add_argument('--video-path', '--video_path', dest='video_path', required=True)
    parser.add_argument('--instrument-key', '--instrument_key', dest='instrument_key')
    parser.add_argument(
        '--channels', nargs='+',
        help='Artifact channel IDs or unambiguous wavelength aliases to render.',
    )
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', '--batch_size', dest='batch_size', type=int, default=512)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument(
        '--video-file', type=Path,
        help='Output movie path (default: <video_path>/plasma.mp4).',
    )
    parser.add_argument(
        '--frames-only', action='store_true',
        help='Write JPEG frames without encoding a movie.',
    )
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error('--fps must be greater than zero')

    frame_directory = Path(args.video_path)
    frame_directory.mkdir(parents=True, exist_ok=True)
    loader = PlasmaSuNeRFLoader(args.chk_path)
    instrument_key = args.instrument_key or loader.instrument_keys[0]
    if instrument_key not in loader.instrument_keys:
        parser.error(
            f"unknown instrument {instrument_key!r}; choose one of {loader.instrument_keys}"
        )
    try:
        channel_metadata = _select_channel_metadata(
            loader.channel_metadata(instrument_key), args.channels
        )
    except ValueError as error:
        parser.error(str(error))
    ds_key = loader.dataset_key(instrument_key)
    avg_time = loader.start_time(ds_key) + (
        loader.end_time(ds_key) - loader.start_time(ds_key)
    ) / 2

    points = _trajectory(avg_time)
    resolution = (args.resolution, args.resolution) * u.pix
    image_norms = [ImageNormalize(stretch=AsinhStretch(0.001)) for _ in channel_metadata]
    n_columns = max(len(channel_metadata), 3)

    frame_paths = []
    for index, (lat, lon, time, distance_au) in tqdm(
        enumerate(points), total=len(points)
    ):
        outputs = loader.load_image(
            lat * u.deg,
            lon * u.deg,
            time,
            distance=distance_au * u.AU,
            batch_size=args.batch_size,
            resolution=resolution,
            instrument_key=instrument_key,
        )
        fig, axes = plt.subplots(2, n_columns, figsize=(n_columns * 3, 5), squeeze=False)

        for channel_index, channel in enumerate(channel_metadata):
            channel_id = channel['id']
            channel_map = outputs['maps'][channel_id]
            image = axes[0, channel_index].imshow(
                channel_map.data,
                cmap=_channel_cmap(channel),
                norm=image_norms[channel_index],
                origin='lower',
            )
            axes[0, channel_index].set_title(str(channel_id))
            _colorbar(fig, axes[0, channel_index], image)

        column_density = outputs.get('column_electron_density_cm2')
        if column_density is None:
            column_density = outputs.get('total_ne')
        if column_density is None:
            raise ValueError(
                'Renderer produced neither column_electron_density_cm2 nor total_ne.'
            )
        absorption = outputs['mean_absorption']
        diagnostic_images = [
            ('Mean log$_{10}$(T / K)', outputs['mean_T'], 'plasma', None),
            ('Column $N_e$ [cm$^{-2}$]', column_density, 'viridis',
             _positive_lognorm(column_density, floor=1)),
            ('Integrated absorption', absorption, 'cool',
             _positive_lognorm(absorption)),
        ]
        for diagnostic_index, (title, values, cmap, norm) in enumerate(diagnostic_images):
            image = axes[1, diagnostic_index].imshow(
                np.squeeze(values), cmap=cmap, norm=norm, origin='lower'
            )
            axes[1, diagnostic_index].set_title(title)
            _colorbar(fig, axes[1, diagnostic_index], image)

        for ax in axes.ravel():
            ax.axis('off')
        fig.tight_layout()
        frame_path = frame_directory / f'{index:03d}.jpg'
        fig.savefig(frame_path, dpi=300)
        plt.close(fig)
        frame_paths.append(frame_path)

    if not args.frames_only:
        video_file = args.video_file or frame_directory / 'plasma.mp4'
        video_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_video(frame_paths, video_file, args.fps)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f'Wrote {len(frame_paths)} frames and video to {video_file}')
    else:
        print(f'Wrote {len(frame_paths)} frames to {frame_directory}')


if __name__ == '__main__':
    main()
