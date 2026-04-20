import argparse
import glob
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from dateutil.parser import parse
from matplotlib.colors import LogNorm
from tqdm import tqdm

warnings.filterwarnings("ignore", category=VerifyWarning)


TIME_KEY_CANDIDATES = {
    "auto": ["DATE-AVG", "DATE_AVG", "DATE-OBS", "DATE_OBS", "DATE-BEG", "DATE_BEG", "DATE-END", "DATE_END"],
    "avg": ["DATE-AVG", "DATE_AVG"],
    "obs": ["DATE-OBS", "DATE_OBS"],
    "beg": ["DATE-BEG", "DATE_BEG"],
    "end": ["DATE-END", "DATE_END"],
}


@dataclass
class Series:
    observable: str
    label: str
    pattern: str
    files: list[str]
    times: list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot observation coverage for tB and pB FITS files."
    )
    parser.add_argument("--tb-path", action="append", default=[], help="Glob pattern for one tB series.")
    parser.add_argument("--tb-label", action="append", default=[], help="Label for one tB series.")
    parser.add_argument("--pb-path", action="append", default=[], help="Glob pattern for one pB series.")
    parser.add_argument("--pb-label", action="append", default=[], help="Label for one pB series.")
    parser.add_argument("--out-path", required=True, help="Output image path, e.g. results/coverage.png")
    parser.add_argument(
        "--time-key",
        choices=sorted(TIME_KEY_CANDIDATES.keys()),
        default="auto",
        help="Header time keyword preference.",
    )
    parser.add_argument("--title", default="Observation coverage")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--size", type=float, nargs=2, default=(12.0, 4.0), metavar=("WIDTH", "HEIGHT"))
    return parser.parse_args()


def validate_series_args(paths, labels, observable):
    if not paths and not labels:
        return
    if len(paths) != len(labels):
        raise ValueError(
            f"{observable} paths/labels mismatch: got {len(paths)} path(s) and {len(labels)} label(s)."
        )


def extract_time_from_header(file_path, time_key_mode):
    for ext in range(4):
        try:
            header = fits.getheader(file_path, ext=ext)
        except Exception:
            continue
        for key in TIME_KEY_CANDIDATES[time_key_mode]:
            value = header.get(key)
            if value:
                return parse(str(value)).replace(tzinfo=None)
    raise ValueError(f"No supported time keyword found in {file_path}")


def load_series(paths, labels, observable, time_key_mode):
    series_list = []
    for pattern, label in zip(paths, labels):
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {observable} pattern: {pattern}")
        desc = f"Loading {observable} times: {label}"
        times = [extract_time_from_header(file_path, time_key_mode) for file_path in tqdm(files, desc=desc)]
        series_list.append(Series(observable=observable, label=label, pattern=pattern, files=files, times=times))
    return series_list


def build_time_edges(series_list):
    all_times = [time for series in series_list for time in series.times]
    x_min = min(all_times)
    x_max = max(all_times)
    start = np.datetime64(x_min.replace(minute=0, second=0, microsecond=0))
    start_hour = (x_min.hour // 6) * 6
    start = np.datetime64(x_min.replace(hour=start_hour, minute=0, second=0, microsecond=0))
    end = np.datetime64(x_max.replace(minute=0, second=0, microsecond=0)) + np.timedelta64(6, "h")
    return mdates.date2num(
        np.arange(start, end + np.timedelta64(6, "h"), np.timedelta64(6, "h"))
        .astype("datetime64[ms]")
        .astype(object)
    )


def plot_heatmap(ax, subset, time_edges, observable, n_rows):
    if not subset:
        ax.set_visible(False)
        return None

    counts = np.zeros((n_rows, len(time_edges) - 1), dtype=np.int32)
    for row, series in enumerate(subset):
        time_values = mdates.date2num(series.times)
        hist, _ = np.histogram(time_values, bins=time_edges)
        counts[row] = hist

    masked_counts = np.ma.masked_equal(counts, 0)
    image = ax.imshow(
        masked_counts,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(1, counts.max())),
        extent=[time_edges[0], time_edges[-1], -0.5, n_rows - 0.5],
        origin="lower",
    )
    ax.set_yticks(np.arange(n_rows))
    labels = [series.label for series in subset] + [""] * (n_rows - len(subset))
    ax.set_yticklabels(labels)
    ax.set_ylabel(observable)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    return image


def plot_coverage(series_list, out_path, title, dpi, size):
    grouped = {
        observable: [series for series in series_list if series.observable == observable]
        for observable in ("tB", "pB")
    }
    visible_observables = [observable for observable in ("tB", "pB") if grouped[observable]]
    fig, axes = plt.subplots(
        len(visible_observables),
        1,
        figsize=size,
        constrained_layout=True,
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    time_edges = build_time_edges(series_list)
    locator = mdates.AutoDateLocator()
    n_rows = max(len(grouped[observable]) for observable in visible_observables)
    images = []
    for ax, observable in zip(axes, visible_observables):
        image = plot_heatmap(ax, grouped[observable], time_edges, observable, n_rows)
        ax.set_title(observable, loc="left", fontweight="bold")
        ax.xaxis_date()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        images.append(image)

    axes[0].set_title(title)
    axes[-1].set_xlabel("Time")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    for ax, image in zip(axes, images):
        if image is not None:
            fig.colorbar(image, ax=ax, pad=0.01, label="Observations / bin")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_summary(series_list):
    for series in series_list:
        start = min(series.times)
        end = max(series.times)
        print(
            f"{series.observable:>2} | {series.label} | {len(series.files)} file(s) | "
            f"{start.isoformat()} -> {end.isoformat()}"
        )


def main():
    args = parse_args()
    validate_series_args(args.tb_path, args.tb_label, "tB")
    validate_series_args(args.pb_path, args.pb_label, "pB")
    if not args.tb_path and not args.pb_path:
        raise SystemExit("Provide at least one --tb-path or --pb-path series.")

    series_list = []
    series_list.extend(load_series(args.tb_path, args.tb_label, "tB", args.time_key))
    series_list.extend(load_series(args.pb_path, args.pb_label, "pB", args.time_key))

    plot_coverage(
        series_list=series_list,
        out_path=args.out_path,
        title=args.title,
        dpi=args.dpi,
        size=tuple(args.size),
    )
    print_summary(series_list)
    print(f"Saved coverage plot to {args.out_path}")


if __name__ == "__main__":
    main()
