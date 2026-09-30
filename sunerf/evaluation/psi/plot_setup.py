#!/usr/bin/env python3
"""Plot 4: setup overview with the observer configuration and the degraded observations.

The sketch of the viewpoints sits between the degraded tB (top) and pB (bottom)
images of the training views; the yellow disk marks the size of the Sun.  Views
listed in ``--tb-only`` are trained on total brightness alone and show no pB.  Reads
the degraded GT images only; writes ``psi_setup``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from sunerf.evaluation.cme.export_layout import view_frame
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, PRODUCT_LABELS, VIEWPOINTS, add_colorbar, add_sun, image_norm, label_plane_axes, load_maps,
    map_extent, plot_image, save_figure,
)

ROLES = {"Earth": ("C1", "L1: C2-like\n(3 views)"), "L4": ("C0", "L4: COR2-like"),
         "L5": ("C3", "L5: CCOR-like\n+ PUNCH-like")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degraded-dir", type=Path, required=True, help="Degraded GT images <view>/{tb,pb}.")
    parser.add_argument("--frames", type=view_frame, nargs="+", required=True, metavar="VIEW=FRAME",
                        help="Views (columns; the sketch is in the center) and their frame, e.g. L5=153 L4=153.")
    parser.add_argument("--tb-only", nargs="*", default=[],
                        help="Views observed in total brightness only; their pB panel is left out.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_geometry(axis) -> None:
    """Schematic ecliptic view of the training viewpoints and the held-out polar view."""
    axis.set_aspect("equal")
    axis.set_xlim(-1.5, 1.5)
    axis.set_ylim(-1.5, 1.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.7", linestyle=":"))
    axis.plot(0, 0, marker="o", color="orange", markersize=9)
    for name, longitude in VIEWPOINTS.items():
        color, text = ROLES[name]
        x, y = np.sin(np.deg2rad(longitude)), -np.cos(np.deg2rad(longitude))
        axis.plot([0, x], [0, y], color=color, linewidth=0.8)
        axis.plot(x, y, marker="s", color=color, markersize=5)
        axis.annotate(text, (x, y), xytext=(0, -9), textcoords="offset points",
                      ha="center", va="top", fontsize=6.5, color=color)
    axis.plot(0, 0.35, marker="^", color="C2", markersize=6)
    axis.annotate("P1: 80° lat.\n(held out)", (0, 0.35), xytext=(0, 6), textcoords="offset points",
                  ha="center", fontsize=6.5, color="C2")


def main() -> None:
    args = parse_args()
    frames = dict(args.frames)
    views = list(frames)
    maps = load_maps(args.degraded_dir, views, frames)

    n_left = (len(views) + 1) // 2
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2 * FIGURE_WIDTH / (len(views) + 2.3) + 0.4), constrained_layout=True)
    # The sketch covers two rows and two image widths.
    grid = GridSpec(2, len(views) + 2, figure=fig,
                    width_ratios=(*[1] * n_left, 2, *[1] * (len(views) - n_left), 0.06))
    plot_geometry(fig.add_subplot(grid[:, n_left]))
    columns = [*range(n_left), *range(n_left + 1, len(views) + 1)]
    for row, product in enumerate(PRODUCT_LABELS):
        # One range per observable over all views that observe it.
        observed = [view for view in views if product == "tb" or view not in args.tb_only]
        norm = image_norm([np.asarray(maps[view][product].data) for view in observed])
        for column, view in zip(columns, views):
            axis = fig.add_subplot(grid[row, column])
            if view not in observed:
                axis.text(0.5, 0.5, "tB only", transform=axis.transAxes, ha="center", va="center", fontsize=7)
                axis.set_xticks([])
                axis.set_yticks([])
                continue
            artist = plot_image(axis, np.asarray(maps[view][product].data), norm, map_extent(maps[view][product]))
            add_sun(axis)
            if row == 0:
                axis.set_title(view, fontsize=8)
            label_plane_axes(axis, r"x [R$_\odot$]", r"y [R$_\odot$]", show_x=row == 1 or view in args.tb_only,
                             show_y=column == 0)
        add_colorbar(fig, artist, grid[row, -1], f"{PRODUCT_LABELS[product]} [MSB]")
    print(save_figure(fig, args.output_dir, "psi_setup"))


if __name__ == "__main__":
    main()
