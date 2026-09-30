#!/usr/bin/env python3
"""Plot 6: propagation of the mass inside the CME cone; no background is subtracted from the mass.

The top row shows the mass per unit radius as a function of radius and time for
GT and every run.  The ambient wind is nearly constant in this quantity, so the
CME is the ridge.  The lines mark the leading edge (outermost radius where the
mass per radius exceeds its value in the first frame by 10 %), the outer edge of
the dense body and its peak; white lines belong to the shown density, black
lines repeat GT.  The bottom row shows the mass inside fixed shells over time
with the correlation coefficient of every run with GT.  A fully ionized
hydrogen plasma converts electrons to mass.

Reads the CME series of GT and the runs
(``sunerf.evaluation.cme.export_cme_series``).  Writes ``psi_mass_propagation``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.cme.mass_transport import PROTON_MASS_G
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, RUN_COLORS, TRUTH_COLOR, TRUTH_TITLE, add_colorbar, format_time_axis, load_series, save_figure,
    series_leading_edge,
)

SHELLS = ((3.0, 15.0), (15.0, 30.0))
# track of the series, label, line style
TRACKS = ((None, "leading edge", "-"), ("body_edge_apex", "body edge", "--"), ("body_peak_median", "body peak", ":"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psi-series", type=Path, required=True, help="CME series of the GT cubes (npz).")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def shell_mass(entry, shell):
    """Mass in grams between two radii for every time."""
    inside = (entry["radius"] >= shell[0]) & (entry["radius"] <= shell[1])
    return np.trapezoid(entry["mass_per_radius"][:, inside], entry["radius"][inside], axis=1) * PROTON_MASS_G


def plot_tracks(axis, entry, color, labelled):
    for key, label, style in TRACKS:
        track = series_leading_edge(entry, "mass_per_radius") if key is None else entry[key]
        axis.plot(entry["times"], track, style, color=color, linewidth=0.7, label=label if labelled else None)


def correlation(reference, values):
    """Pearson correlation coefficient of two curves over their common finite samples."""
    valid = np.isfinite(reference) & np.isfinite(values)
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(reference[valid], values[valid])[0, 1])


def main() -> None:
    args = parse_args()
    series = [load_series(args.psi_series)] + [load_series(export_dir) for _, export_dir in args.runs]
    labels = [TRUTH_TITLE] + [label for label, _ in args.runs]
    colors = [TRUTH_COLOR, *RUN_COLORS]
    psi = series[0]

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / (len(series) + 0.4) + 2.6), constrained_layout=True)
    grid = GridSpec(2, len(series) + 1, figure=fig, width_ratios=(*[1] * len(series), 0.05),
                    height_ratios=(1, 0.9))
    norm = LogNorm(*np.percentile(psi["mass_per_radius"] * PROTON_MASS_G, (1.0, 99.5)))
    for column, (entry, label) in enumerate(zip(series, labels)):
        axis = fig.add_subplot(grid[0, column])
        artist = axis.pcolormesh(entry["times"], entry["radius"], entry["mass_per_radius"].T * PROTON_MASS_G,
                                 norm=norm, cmap="inferno", shading="nearest", rasterized=True)
        if column > 0:
            plot_tracks(axis, psi, "black", labelled=False)
        plot_tracks(axis, entry, "white", labelled=column == 0)
        axis.set_title(label, fontsize=8)
        axis.set_xlabel("Time [UT]", fontsize=7)
        if column == 0:
            axis.set_ylabel(r"Radius [R$_\odot$]", fontsize=7)
            axis.legend(fontsize=5.5, loc="upper left", labelcolor="white", frameon=False)
        else:
            axis.set_yticklabels([])
        if column == 1:
            axis.text(0.04, 0.96, "black: GT", transform=axis.transAxes, va="top", ha="left", fontsize=5.5,
                      color="white")
        format_time_axis(axis, max_ticks=3)
    add_colorbar(fig, artist, grid[0, -1], r"$dM/dr$ [g R$_\odot^{-1}$]")

    shells = grid[1, :].subgridspec(1, len(SHELLS))
    for column, shell in enumerate(SHELLS):
        axis = fig.add_subplot(shells[0, column])
        reference = shell_mass(psi, shell)
        for entry, label, color in zip(series, labels, colors):
            mass = shell_mass(entry, shell)
            r = correlation(reference, mass) if entry is not psi and len(mass) == len(reference) else np.nan
            axis.plot(entry["times"], mass, "-", linewidth=0.9, color=color,
                      label=f"{label} (r = {r:.3f})" if np.isfinite(r) else label)
        axis.set_title(f"Mass in the cone, {shell[0]:.0f}-{shell[1]:.0f} R$_\\odot$", fontsize=8)
        axis.set_xlabel("Time [UT]", fontsize=7)
        axis.set_ylabel("Mass [g]", fontsize=7)
        axis.yaxis.get_offset_text().set_fontsize(6.5)
        axis.legend(fontsize=5.5)
        format_time_axis(axis)
    print(save_figure(fig, args.output_dir, "psi_mass_propagation"))


if __name__ == "__main__":
    main()
