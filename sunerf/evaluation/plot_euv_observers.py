"""Overview of the EUV observers behind a prepared data set.

The script reads every FITS header in a folder and produces a single figure:

* a 3D view with the Sun at the center and one arrow per observation, drawn at
  the observer position and pointing towards Sun center, colored by time;
* a 2D coverage histogram with one row per instrument and the number of
  observations per time bin.

Only FITS headers are read, so the overview is cheap even for large archives.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'matplotlib'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from sunpy.coordinates import frames
from tqdm import tqdm

from sunerf.data.euv.observation import _header_time

warnings.filterwarnings('ignore', category=VerifyWarning)

_BASE_SUFFIXES = ('.fits', '.fit', '.fts')
FITS_SUFFIXES = _BASE_SUFFIXES + tuple(f'{suffix}.gz' for suffix in _BASE_SUFFIXES)
ARROW_COLOR_MAP = 'plasma'
COUNT_COLOR_MAP = 'viridis'
# Bin widths offered to the coverage histogram, from minutes to one week.
BIN_WIDTH_HOURS = (1 / 12, 0.5, 1, 3, 6, 12, 24, 48, 168)
TARGET_BINS = 150
MAX_ARROWS_PER_INSTRUMENT = 80


@dataclass(frozen=True)
class Observation:
    """One prepared EUV observation reduced to its observer geometry."""

    path: Path
    time: datetime
    instrument: str
    # Heliocentric inertial cartesian position of the observer, in AU.
    position: np.ndarray


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path, help='Folder with prepared EUV FITS files.')
    parser.add_argument('--output', type=Path, default=Path('euv_observers.jpg'),
                        help='Output image path.')
    return parser.parse_args()


def find_fits_files(path: Path) -> list[Path]:
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = [file for file in sorted(path.rglob('*'))
             if file.name.lower().endswith(FITS_SUFFIXES) and file.is_file()]
    if not files:
        raise FileNotFoundError(f'No FITS files found below {path}.')
    return files


def instrument_label(header: fits.Header) -> str:
    """Build a stable ``observatory/detector channel`` label from the header."""
    observatory = str(header.get('OBSRVTRY') or header.get('TELESCOP') or 'UNKNOWN').strip()
    # SDO writes TELESCOP='SDO/AIA'; keep the platform part only.
    observatory = observatory.split('/')[0].strip()
    detector = str(header.get('DETECTOR') or header.get('INSTRUME') or '').strip()
    channel = header.get('WAVELNTH')
    label = f'{observatory}/{detector}' if detector else observatory
    if channel is not None:
        label = f'{label} {round(float(channel))}'
    return label


def observer_position(header: fits.Header, time: datetime) -> np.ndarray:
    """Return the observer position in heliocentric inertial cartesian AU."""
    distance = header.get('DSUN_OBS')
    if distance is None:
        raise ValueError('FITS header is missing DSUN_OBS.')
    if header.get('HGLN_OBS') is not None and header.get('HGLT_OBS') is not None:
        frame = frames.HeliographicStonyhurst(obstime=time)
        longitude, latitude = header['HGLN_OBS'], header['HGLT_OBS']
    elif header.get('CRLN_OBS') is not None and header.get('CRLT_OBS') is not None:
        frame = frames.HeliographicCarrington(observer='self', obstime=time)
        longitude, latitude = header['CRLN_OBS'], header['CRLT_OBS']
    else:
        raise ValueError('FITS header has neither Stonyhurst nor Carrington observer angles.')
    observer = SkyCoord(float(longitude) * u.deg, float(latitude) * u.deg,
                        float(distance) * u.m, frame=frame)
    inertial = observer.transform_to(frames.HeliocentricInertial(obstime=time)).cartesian
    return np.array([inertial.x.to_value(u.AU), inertial.y.to_value(u.AU),
                     inertial.z.to_value(u.AU)])


def load_observation(path: Path) -> Observation | str:
    """Return the observer geometry, or the reason the file was unusable."""
    try:
        header = fits.getheader(path)
        time = _header_time(header)
        return Observation(path=path, time=time, instrument=instrument_label(header),
                           position=observer_position(header, time))
    except Exception as error:
        return f'{path.name}: {error}'


def load_observations(files: list[Path]) -> list[Observation]:
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(load_observation, files),
                            total=len(files), desc='Reading FITS headers'))
    observations = [result for result in results if isinstance(result, Observation)]
    skipped = [result for result in results if not isinstance(result, Observation)]
    if skipped:
        print(f'Skipped {len(skipped)} unreadable file(s), for example: {skipped[0]}')
    if not observations:
        raise ValueError('No FITS file provided a usable observation time and observer.')
    return sorted(observations, key=lambda observation: observation.time)


def group_by_instrument(observations: list[Observation]) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.instrument, []).append(observation)
    return {label: grouped[label] for label in sorted(grouped)}


def group_by_platform(grouped: dict[str, list[Observation]]) -> dict[str, list[Observation]]:
    """Merge the channels of one telescope, which share the same trajectory."""
    platforms: dict[str, list[Observation]] = {}
    for label, observations in grouped.items():
        platforms.setdefault(label.rsplit(' ', 1)[0], []).extend(observations)
    return {label: sorted(observations, key=lambda observation: observation.time)
            for label, observations in platforms.items()}


def subsample(observations: list[Observation], count: int) -> list[Observation]:
    if len(observations) <= count:
        return observations
    index = np.linspace(0, len(observations) - 1, count).round().astype(int)
    return [observations[position] for position in np.unique(index)]


def time_bin_edges(times: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick a readable bin width and return the edges in date numbers and the width."""
    span_hours = max((times.max() - times.min()) * 24, 1 / 60)
    width = min(BIN_WIDTH_HOURS, key=lambda value: abs(span_hours / value - TARGET_BINS))
    width_days = width / 24
    start = np.floor(times.min() / width_days) * width_days
    n_bins = int(np.ceil((times.max() - start) / width_days)) + 1
    return start + np.arange(n_bins + 1) * width_days, width


def format_bin_width(width_hours: float) -> str:
    if width_hours < 1:
        return f'{width_hours * 60:.0f} min'
    if width_hours < 24:
        return f'{width_hours:.0f} h'
    return f'{width_hours / 24:.0f} d'


def plot_observers(ax, grouped: dict[str, list[Observation]], norm: Normalize):
    """Draw the Sun, the observer trajectories and one arrow per observation."""
    positions = np.array([observation.position
                          for observations in grouped.values()
                          for observation in observations])
    extent = float(np.abs(positions).max()) * 1.1
    arrow_length = 0.15 * extent

    ax.scatter([0], [0], [0], color='gold', s=160, edgecolors='darkorange',
               depthshade=False, zorder=5, label='Sun')
    # Channels of the same telescope share a trajectory; draw and label it once.
    for label, observations in group_by_platform(grouped).items():
        track = np.array([observation.position for observation in observations])
        ax.plot(track[:, 0], track[:, 1], track[:, 2], color='0.6', lw=0.8, alpha=0.7)
        anchor = track[-1] * (1 + 2 * arrow_length / np.linalg.norm(track[-1]))
        ax.text(*anchor, label, color='0.3', fontsize=7, ha='center', va='bottom')

    for observations in grouped.values():
        arrows = subsample(observations, MAX_ARROWS_PER_INSTRUMENT)
        origin = np.array([observation.position for observation in arrows])
        # Every instrument points at Sun center; the arrow shows the view direction.
        direction = -origin / np.linalg.norm(origin, axis=1, keepdims=True) * arrow_length
        colors = plt.get_cmap(ARROW_COLOR_MAP)(
            norm(mdates.date2num([observation.time for observation in arrows])))
        ax.quiver(origin[:, 0], origin[:, 1], origin[:, 2],
                  direction[:, 0], direction[:, 1], direction[:, 2],
                  colors=colors, arrow_length_ratio=0.35, lw=1.0)

    # Spacecraft stay close to the ecliptic; flatten the box instead of wasting
    # half the panel on empty space, keeping the same scale on all three axes.
    z_extent = max(float(np.abs(positions[:, 2]).max()) * 1.4, 0.25 * extent)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-z_extent, z_extent)
    ax.set_box_aspect((1, 1, z_extent / extent), zoom=1.3)
    ax.set_xlabel('X [AU]')
    ax.set_ylabel('Y [AU]')
    ax.set_zlabel('Z [AU]')
    ax.set_title('Observer positions and viewing directions (HCI)')
    ax.view_init(elev=25, azim=-60)


def plot_coverage(ax, cax, grouped: dict[str, list[Observation]]):
    """Draw the per-instrument observation counts as a 2D histogram."""
    times = {label: mdates.date2num([observation.time for observation in observations])
             for label, observations in grouped.items()}
    edges, width_hours = time_bin_edges(np.concatenate(list(times.values())))
    counts = np.array([np.histogram(times[label], bins=edges)[0] for label in grouped])

    maximum = max(1, int(counts.max()))
    image = ax.pcolormesh(edges, np.arange(len(grouped) + 1) - 0.5,
                          np.ma.masked_equal(counts, 0), cmap=COUNT_COLOR_MAP,
                          vmin=0.5, vmax=maximum + 0.5)
    ax.set_yticks(np.arange(len(grouped)))
    ax.set_yticklabels([f'{label}  ({len(grouped[label])})' for label in grouped])
    ax.set_ylim(-0.5, len(grouped) - 0.5)
    ax.set_xlim(edges[0], edges[-1])
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel('Time')
    ax.set_title(f'Observation coverage ({format_bin_width(width_hours)} bins)')
    ax.grid(axis='x', alpha=0.25)
    ax.set_axisbelow(True)
    ticks = np.arange(1, maximum + 1) if maximum <= 10 else None
    plt.colorbar(image, cax=cax, ticks=ticks,
                 label=f'Observations per {format_bin_width(width_hours)}')


def plot_overview(grouped: dict[str, list[Observation]], output: Path):
    times = mdates.date2num([observation.time
                             for observations in grouped.values()
                             for observation in observations])
    # A single observation time would collapse the color scale.
    norm = Normalize(vmin=times.min(), vmax=max(times.max(), times.min() + 1e-6))

    height = 6 + 0.25 * len(grouped)
    fig = plt.figure(figsize=(9, height))
    grid = GridSpec(3, 2, figure=fig, height_ratios=[4, 0.14, 2 + 0.2 * len(grouped)],
                    width_ratios=[40, 1], hspace=0.55, wspace=0.04)
    observer_ax = fig.add_subplot(grid[0, :], projection='3d')
    time_cax = fig.add_subplot(grid[1, 0])
    coverage_ax = fig.add_subplot(grid[2, 0])
    count_cax = fig.add_subplot(grid[2, 1])

    plot_observers(observer_ax, grouped, norm)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=ARROW_COLOR_MAP),
                            cax=time_cax, orientation='horizontal', label='Observation time')
    locator = mdates.AutoDateLocator()
    colorbar.ax.xaxis.set_major_locator(locator)
    colorbar.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    plot_coverage(coverage_ax, count_cax, grouped)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)


def print_summary(grouped: dict[str, list[Observation]]):
    for label, observations in grouped.items():
        print(f'{label:>24} | {len(observations):5d} files | '
              f'{observations[0].time.isoformat()} -> {observations[-1].time.isoformat()} | '
              f'{np.linalg.norm(observations[0].position):.2f} AU')


def main():
    args = parse_args()
    observations = load_observations(find_fits_files(args.path))
    grouped = group_by_instrument(observations)
    print_summary(grouped)
    plot_overview(grouped, args.output)
    print(f'Saved observer overview to {args.output}')


if __name__ == '__main__':
    main()
