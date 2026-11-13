#!/usr/bin/env python3
"""Plot 9: kinematics of the CME front and cavity in the meridional cut.

The front is the maximum of the outermost enhancement of ``r^2 n_e`` along a ray
(the bright dense front) and the cavity the minimum behind it.  Both are tracked
along fixed rays of the meridional cut through the CME direction over all dumps,
outward only, so that a ray is no longer followed after the structure has left.

Top: radial speed of the front and of the cavity as a function of latitude and
time for GT and every run (quadratic fit in a sliding 45 min window along every
ray, median over three neighbouring rays).  White dotted lines mark the cone.

Bottom: height, speed and acceleration over time of the median over the rays of
the cone; the bands give the 16th to 84th percentile of the height over these
rays.  The curves end when the fastest front ray comes within 3 solar radii of the outer limit.  Speed and
acceleration come from quadratic fits in sliding windows of 45 min and 90 min.

Reads the CME series of GT and the runs
(``sunerf.evaluation.cme.export_cme_series``).  Writes ``psi_cme_kinematics`` and,
to ``psi_cme_kinematics.txt``, the speed at 20 solar radii and the acceleration of
a constant-acceleration fit to the median tracks.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.cme.mass_transport import fit_constant_acceleration, local_kinematics
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, RUN_COLORS, TRUTH_COLOR, TRUTH_TITLE, add_colorbar, flat_label, format_time_axis, load_series,
    save_figure, title_size, track_cut,
)

STRUCTURES = ("Front", "Cavity")
SPEED_WINDOW = 45 * 60.0  # s
ACCELERATION_WINDOW = 90 * 60.0  # s
REFERENCE_RADIUS = 20.0  # R_sun
MIN_RAYS = 3
TOP_MARGIN = 3.0  # R_sun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psi-series", type=Path, required=True, help="CME series of the GT cubes (npz).")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def seconds(times):
    return np.array([(time - times[0]).total_seconds() for time in times])


def speed_map(times_s, tracks):
    """Radial speed [km/s] of ``tracks[time, ray]``, smoothed over three neighbouring rays."""
    speed = np.stack([local_kinematics(times_s, track, SPEED_WINDOW)[0] for track in tracks.T], axis=1)
    padded = np.pad(speed, ((0, 0), (1, 1)), constant_values=np.nan)
    neighbours = np.stack([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]])
    with np.errstate(all="ignore"):
        smoothed = np.nanmedian(neighbours, axis=0)
    # rays without an own speed stay empty
    return np.where(np.isfinite(speed), smoothed, np.nan)


def end_of_tracking(front, radius):
    """Index of the first time a front ray comes within ``TOP_MARGIN`` of the outer limit of the grid.

    Beyond that time the fastest rays leave the grid one after the other and
    statistics over the remaining rays would follow the slower flanks only.
    """
    with np.errstate(all="ignore"):
        highest = np.nanmax(np.where(np.isfinite(front), front, -np.inf), axis=1)
    reached = np.flatnonzero(highest >= radius[-1] - TOP_MARGIN)
    return reached[0] if reached.size else len(front)


def cone_statistics(tracks, end):
    """Median and 16th / 84th percentile over the rays ``tracks[time, ray]`` for every time before ``end``."""
    enough = np.isfinite(tracks).sum(axis=1) >= MIN_RAYS
    enough[end:] = False
    with np.errstate(all="ignore"):
        low, median, high = np.nanpercentile(np.where(enough[:, None], tracks, np.nan), (16, 50, 84), axis=1)
    return median, low, high


def write_fits(path: Path, labels, fits) -> None:
    """Speed at the reference radius and acceleration of the constant-acceleration fit of every median track."""
    lines = [
        "Constant-acceleration fit r(t) = r0 + v0 t + a t^2 / 2 to the median height-time track of the cone rays.",
        f"speed: km/s at {REFERENCE_RADIUS:g} R_sun (- if the track does not cross it); acceleration: m/s^2.",
        "",
        f"{'structure':<12}{'quantity':<14}" + "".join(f"  {flat_label(label):>28}" for label in labels),
    ]
    for structure in STRUCTURES:
        for index, quantity in enumerate(("speed", "acceleration")):
            values = [fits[structure, label][index] for label in labels]
            cells = [f"{value:>28.1f}" if np.isfinite(value) else f"{'-':>28}" for value in values]
            lines.append(f"{structure:<12}{quantity:<14}" + "".join(f"  {cell}" for cell in cells))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    series = [load_series(args.psi_series)] + [load_series(export_dir) for _, export_dir in args.runs]
    labels = [TRUTH_TITLE] + [label for label, _ in args.runs]
    psi = series[0]
    target_latitude, half_angle = float(psi["target_latitude"]), float(psi["cone_half_angle"])
    period = psi["times"][0], psi["times"][-1]
    tracks = [dict(zip(STRUCTURES, track_cut(entry))) for entry in series]
    speeds = [{structure: speed_map(seconds(entry["times"]), track[structure]) for structure in STRUCTURES}
              for entry, track in zip(series, tracks)]

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2 * FIGURE_WIDTH / (len(series) + 0.4) + 4.2), constrained_layout=True)
    blocks = fig.add_gridspec(2, 1, height_ratios=(2 * FIGURE_WIDTH / (len(series) + 0.4), 3.9))

    maps = blocks[0].subgridspec(len(STRUCTURES), len(series) + 1, width_ratios=(*[1] * len(series), 0.05))
    finite = speeds[0]["Front"][np.isfinite(speeds[0]["Front"])]
    norm = Normalize(0.0, np.percentile(finite, 99) if finite.size else 1.0)
    for row, structure in enumerate(STRUCTURES):
        for column, (entry, speed, label) in enumerate(zip(series, speeds, labels)):
            axis = fig.add_subplot(maps[row, column])
            latitude_deg = np.rad2deg(entry["cut_latitude"])
            artist = axis.pcolormesh(entry["times"], latitude_deg, speed[structure].T, norm=norm, cmap="viridis",
                                     shading="nearest", rasterized=True)
            axis.set_facecolor("0.85")
            for side in (-1, 1):
                axis.axhline(np.rad2deg(target_latitude + side * half_angle), color="white", linewidth=0.5,
                             linestyle=":")
            axis.set_xlim(*period)
            if row == 0:
                axis.set_title(label, fontsize=title_size(len(series)))
            if column == 0:
                axis.set_ylabel(f"{structure}\nLatitude [deg]", fontsize=7)
            else:
                axis.set_yticklabels([])
            format_time_axis(axis, max_ticks=3)
            if row == len(STRUCTURES) - 1:
                axis.set_xlabel("Time [UT]", fontsize=7)
            else:
                axis.set_xticklabels([])
        add_colorbar(fig, artist, maps[row, -1], "Radial speed [km/s]")

    rows = (r"Radius [R$_\odot$]", "Speed [km/s]", r"Accel. [m/s$^2$]")
    curves = blocks[1].subgridspec(len(rows), len(STRUCTURES), height_ratios=(1.2, 1, 1))
    fits = {}
    for column, structure in enumerate(STRUCTURES):
        axes = [fig.add_subplot(curves[row, column]) for row in range(len(rows))]
        for entry, track, label, color in zip(series, tracks, labels, [TRUTH_COLOR, *RUN_COLORS]):
            times = entry["times"]
            in_cone = np.abs(entry["cut_latitude"] - target_latitude) <= half_angle
            end = end_of_tracking(track["Front"][:, in_cone], entry["radius"])
            median, low, high = cone_statistics(track[structure][:, in_cone], end)
            fits[structure, label] = fit_constant_acceleration(seconds(times), median, REFERENCE_RADIUS)
            axes[0].plot(times, median, "-", linewidth=0.8, color=color, label=flat_label(label))
            axes[0].fill_between(times, low, high, color=color, alpha=0.15, linewidth=0)
            axes[1].plot(times, local_kinematics(seconds(times), median, SPEED_WINDOW)[0], "-", linewidth=0.8,
                         color=color)
            axes[2].plot(times, local_kinematics(seconds(times), median, ACCELERATION_WINDOW)[1], "-",
                         linewidth=0.8, color=color)
        axes[0].set_title(structure, fontsize=8)
        axes[2].axhline(0.0, color="0.6", linewidth=0.5)
        axes[-1].set_xlabel("Time [UT]", fontsize=7)
        if column == 0:
            axes[0].legend(fontsize=5.5, loc="lower right")
        for row, axis in enumerate(axes):
            axis.set_xlim(*period)
            if column == 0:
                axis.set_ylabel(rows[row], fontsize=7)
            format_time_axis(axis)
            if axis is not axes[-1]:
                axis.set_xticklabels([])
    write_fits(args.output_dir / "psi_cme_kinematics.txt", labels, fits)
    print(save_figure(fig, args.output_dir, "psi_cme_kinematics"))
    print(args.output_dir / "psi_cme_kinematics.txt")


if __name__ == "__main__":
    main()
