#!/usr/bin/env python3
"""Plot 2: pB images of GT, the degraded observation and every run.

Rows are the views, columns the clean GT render, the degraded observation and
the clean image rendered from every run (``sunerf.evaluation.cme.export_images``).
The held-out view has no degraded observation.  Dashed lines separate the
column groups and the yellow disk marks the size of the Sun.

PSNR and SSIM of every image against the GT render are computed on the
displayed scale: log brightness mapped to [0, 1] with the color range of the GT
image of the view, over the pixels valid in both images.  Writes ``psi_images``
and the metrics to ``psi_images.txt``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity

from sunerf.evaluation.cme.export_layout import image_file, labelled_path, view_frame
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, TRUTH_TITLE, add_colorbar, add_panel_metric, add_sun, image_norm, label_plane_axes, load_maps,
    map_extent, plot_image, save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True, help="Clean GT images <view>/{tb,pb}.")
    parser.add_argument("--degraded-dir", type=Path, required=True, help="Degraded GT images <view>/{tb,pb}.")
    parser.add_argument("--frames", type=view_frame, nargs="+", required=True, metavar="VIEW=FRAME",
                        help="Views (rows) and their frame of the image series, e.g. L4=153 P1=153.")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--heldout-view", default="P1", help="View without degraded observation.")
    parser.add_argument("--separators", type=int, nargs="*", default=[],
                        help="Columns (counted from 1) followed by a dashed separator, e.g. 2 4.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def image_correlation(reference, estimate) -> float:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    return float(np.corrcoef(reference[valid], estimate[valid])[0, 1])


def image_metrics(reference, estimate, norm) -> dict:
    """PSNR [dB] and SSIM of the images on the displayed scale ``norm`` (values in [0, 1])."""
    valid = np.isfinite(reference) & np.isfinite(estimate) & (reference > 0)
    # norm clips to its range; pixels outside the common field of view do not enter the averages
    scaled_reference = np.where(valid, np.ma.filled(norm(np.where(valid, reference, norm.vmin)), 0.0), 0.0)
    scaled_estimate = np.where(valid, np.ma.filled(norm(np.where(valid, np.clip(estimate, norm.vmin, None),
                                                                 norm.vmin)), 0.0), 0.0)
    mse = np.mean((scaled_reference[valid] - scaled_estimate[valid]) ** 2)
    _, ssim_map = structural_similarity(scaled_reference, scaled_estimate, data_range=1.0, full=True)
    return {"psnr": float(-10.0 * np.log10(mse)), "ssim": float(ssim_map[valid].mean()),
            "correlation": image_correlation(reference, estimate)}


def write_metrics(path: Path, titles, metrics) -> None:
    lines = [
        "pB images against the GT render.",
        "PSNR [dB] and SSIM on the displayed scale: log brightness mapped to [0, 1] with the color range",
        "of the GT image of the view, over the pixels valid in both images; correlation: Pearson",
        "coefficient of the linear brightness.",
        "",
        f"{'view':<10}{'metric':<13}" + "".join(f"  {title:>18}" for title in titles),
    ]
    for view, row in metrics.items():
        for key in ("psnr", "ssim", "correlation"):
            cells = [f"{entry[key]:>18.4f}" if entry is not None else f"{'-':>18}" for entry in row]
            lines.append(f"{view:<10}{key:<13}" + "".join(f"  {cell}" for cell in cells))
    path.write_text("\n".join(lines) + "\n")


def add_separators(fig, axes, columns) -> None:
    """Dashed vertical lines over all rows after the given columns (counted from 1)."""
    fig.canvas.draw()
    top = max(axis.get_position().y1 for axis in axes[0] if axis is not None)
    bottom = min(axis.get_position().y0 for axis in axes[-1] if axis is not None)
    for column in columns:
        x = 0.5 * (max(row[column - 1].get_position().x1 for row in axes)
                   + min(row[column].get_tightbbox().transformed(fig.transFigure.inverted()).x0 for row in axes))
        line = Line2D([x, x], [bottom, top], transform=fig.transFigure, color="0.3", linewidth=0.8,
                      linestyle="--")
        line.set_in_layout(False)
        fig.add_artist(line)


def main() -> None:
    args = parse_args()
    frames = dict(args.frames)
    clean = load_maps(args.clean_dir, frames, frames)
    degraded = load_maps(args.degraded_dir, [view for view in frames if view != args.heldout_view], frames)
    titles = [TRUTH_TITLE, "Degraded", *(label for label, _ in args.runs)]

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / (len(titles) + 0.6) * len(frames) + 0.3),
                     constrained_layout=True)
    grid = GridSpec(len(frames), len(titles) + 1, figure=fig, width_ratios=(*[1] * len(titles), 0.05))
    axes, metrics = [], {}
    for row, (view, frame) in enumerate(frames.items()):
        reference = np.asarray(clean[view]["pb"].data)
        images = [reference, np.asarray(degraded[view]["pb"].data) if view in degraded else None]
        images += [np.load(image_file(export_dir, view, "pb", frame)) for _, export_dir in args.runs]
        norm, extent = image_norm([reference]), map_extent(clean[view]["pb"])
        label = f"{view} (held out)" if view == args.heldout_view else view
        metrics[view] = [image_metrics(reference, image, norm) if image is not None else None
                         for image in images[1:]]
        axes.append([])
        for column, image in enumerate(images):
            axis = fig.add_subplot(grid[row, column])
            axes[-1].append(axis)
            if row == 0:
                axis.set_title(titles[column], fontsize=7)
            if image is None:
                axis.text(0.5, 0.5, "held out", transform=axis.transAxes, ha="center", va="center", fontsize=7)
                axis.set_xticks([])
                axis.set_yticks([])
                continue
            artist = plot_image(axis, image, norm, extent)
            add_sun(axis)
            label_plane_axes(axis, r"x [R$_\odot$]", f"{label}\n" + r"y [R$_\odot$]",
                             show_x=row == len(frames) - 1, show_y=column == 0)
            if column > 0:
                entry = metrics[view][column - 1]
                add_panel_metric(axis, f"PSNR {entry['psnr']:.1f} dB\nSSIM {entry['ssim']:.3f}")
        add_colorbar(fig, artist, grid[row, -1], "pB [MSB]")
    add_separators(fig, axes, args.separators)
    print(save_figure(fig, args.output_dir, "psi_images"))
    write_metrics(args.output_dir / "psi_images.txt", titles[1:], metrics)
    print(args.output_dir / "psi_images.txt")


if __name__ == "__main__":
    main()
