#!/usr/bin/env python3
"""Plot a few clean and degraded PSI tB/pB frames without loading a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from sunerf.data.psi.prepare_psi_clear import paired_product_files


DISPLAY_NAME = {"tb": "tB", "pb": "pB"}
PLOT_PRODUCTS = ("tb", "pb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--degraded-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument(
        "--truth-dir",
        type=Path,
        default=None,
        help="degrade_psi truth directory; checks the recovered additive term against the injected one.",
    )
    return parser.parse_args()


def _finite_values(*arrays: np.ndarray) -> np.ndarray:
    values = [array[np.isfinite(array)] for array in arrays]
    values = [value for value in values if value.size]
    if not values:
        raise ValueError("No finite pixels are available to plot.")
    return np.concatenate(values)


def _frame_token(path: Path, product: str) -> str:
    marker = f"_{product}"
    if marker not in path.stem:
        raise ValueError(f"Cannot identify frame number in {path.name}.")
    return path.stem.rsplit(marker, 1)[1]


def _matched_clean(clean_map: Map, degraded_map: Map) -> np.ndarray:
    if clean_map.data.shape != degraded_map.data.shape:
        raise ValueError("Clean and degraded image shapes differ.")
    data = np.asarray(clean_map.data, dtype=np.float32).copy()
    data[~np.isfinite(degraded_map.data)] = np.nan
    return data


def plot_sample(
    clean_paths: dict[str, Path],
    degraded_paths: dict[str, Path],
    view: str,
    output_dir: Path,
    truth_dir: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    titles = ("True (clean, matched FOV)", "Degraded", "Difference (degraded - clean)")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row, product in enumerate(PLOT_PRODUCTS):
        clean_map = Map(clean_paths[product])
        degraded_map = Map(degraded_paths[product])
        clean = _matched_clean(clean_map, degraded_map)
        degraded = np.asarray(degraded_map.data, dtype=np.float32)
        difference = degraded - clean
        difference[~np.isfinite(degraded)] = np.nan

        if truth_dir is not None:
            # The injected additive term is constant in time and follows the
            # (clean + additive) * multiplier convention. A recovered term that
            # departs from it means that the clean and degraded files do not
            # belong together.
            multiplier = np.load(truth_dir / view / f"{product}_multiplier.npy")
            additive = degraded / multiplier - clean
            injected = np.load(truth_dir / view / f"{product}_additive.npy")
            # The residual is signed (background subtraction) and the recovered
            # one carries the frame noise: compare where the residual is a
            # sizable part of its amplitude, relative to that amplitude.
            amplitude = float(np.nanmax(np.abs(injected)))
            both = np.isfinite(injected) & np.isfinite(additive) & (np.abs(injected) > 0.1 * amplitude)
            mismatch = float(np.median(np.abs(additive[both] - injected[both])) / amplitude)
            if mismatch > 0.05:
                print(
                    f"WARNING {view} {product} {degraded_paths[product].name}: recovered additive "
                    f"departs from the injected one by {100 * mismatch:.1f}% (median); the clean and "
                    "degraded sequences are inconsistent, e.g. one of them was re-rendered."
                )

        # Shared stretch for the true and degraded images. The degraded image is
        # background-subtracted and may be negative, so the lower bound follows it.
        values = _finite_values(clean, degraded)
        vmax = max(float(np.nanpercentile(values, 99.5)), np.finfo(np.float32).tiny)
        vmin = min(0.0, float(np.nanpercentile(_finite_values(degraded), 0.5)))
        image_norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(1.0e-2))
        for column, image in enumerate((clean, degraded)):
            artist = axes[row, column].imshow(
                image, origin="lower", cmap=cm.soholasco2, norm=image_norm
            )
            fig.colorbar(artist, ax=axes[row, column], fraction=0.046, label="MSB")

        # The difference is signed and as steep as the corona: a symmetric log
        # norm centered on zero keeps both the faint outer and the bright inner
        # residuals visible.
        difference_vmax = max(
            float(np.nanpercentile(np.abs(_finite_values(difference)), 99.5)), np.finfo(np.float32).tiny
        )
        difference_artist = axes[row, 2].imshow(
            difference,
            origin="lower",
            cmap="RdBu_r",
            norm=SymLogNorm(
                linthresh=1.0e-2 * difference_vmax, vmin=-difference_vmax, vmax=difference_vmax
            ),
        )
        fig.colorbar(difference_artist, ax=axes[row, 2], fraction=0.046, label="difference [MSB]")
        axes[row, 0].set_ylabel(DISPLAY_NAME[product])
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    frame = _frame_token(clean_paths["tb"], "tb")
    fig.suptitle(f"PSI clean vs. degraded: {view}, frame {frame}", fontsize=15)
    output_path = output_dir / f"{view}_frame_{frame}.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be positive.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for view in args.views:
        clean_pairs = paired_product_files(args.clean_dir / view)
        degraded_pairs = paired_product_files(args.degraded_dir / view)
        degraded_by_name = {pair["tb"].name: pair for pair in degraded_pairs}
        common_pairs = [
            (pair, degraded_by_name[pair["tb"].name])
            for pair in clean_pairs
            if pair["tb"].name in degraded_by_name
        ]
        if not common_pairs:
            raise FileNotFoundError(
                f"No matching clean/degraded frames found for {view}."
            )
        indices = np.unique(
            np.linspace(
                0, len(common_pairs) - 1, min(args.sample_count, len(common_pairs))
            ).astype(int)
        )
        outputs.extend(
            plot_sample(*common_pairs[index], view, args.output_dir, args.truth_dir)
            for index in indices
        )

    print("Degradation previews:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
