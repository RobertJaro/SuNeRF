#!/usr/bin/env python3
"""Plot a few clean and degraded PSI tB/pB frames without loading a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib.colors import TwoSlopeNorm
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from sunerf.data.psi.prepare_psi_clear import paired_product_files
from sunerf.data.psi.degrade_psi import radial_gain_mask
from sunerf.data.psi.prepare_psi_clear import radial_rsun


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
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    titles = (
        "Clean (matched FOV)",
        "Actual degraded",
        "Injected additive mask",
        "Injected radial gain",
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row, product in enumerate(PLOT_PRODUCTS):
        clean_map = Map(clean_paths[product])
        degraded_map = Map(degraded_paths[product])
        clean = _matched_clean(clean_map, degraded_map)
        degraded = np.asarray(degraded_map.data, dtype=np.float32)
        required_gain_keys = ("occmin", "occmax", "gainin", "gainout", "gainpow")
        if not all(key in degraded_map.meta for key in required_gain_keys):
            raise ValueError(
                f"Degraded map lacks radial-gain metadata: {degraded_paths[product]}"
            )
        multiplier = radial_gain_mask(
            radial_rsun(degraded_map),
            degraded_map.meta["occmin"],
            degraded_map.meta["occmax"],
            degraded_map.meta["gainin"],
            degraded_map.meta["gainout"],
            degraded_map.meta["gainpow"],
        )
        additive = degraded / multiplier - clean
        additive[~np.isfinite(degraded)] = np.nan
        if truth_dir is not None:
            # The injected term is constant in time. A recovered term that departs
            # from it means that the clean and degraded files do not belong together.
            injected = np.load(truth_dir / view / f"{product}_additive.npy")
            both = np.isfinite(injected) & np.isfinite(additive) & (injected > 0)
            mismatch = float(np.median(np.abs(additive[both] - injected[both]) / injected[both]))
            if mismatch > 0.01:
                print(
                    f"WARNING {view} {product} {degraded_paths[product].name}: recovered additive "
                    f"departs from the injected one by {100 * mismatch:.1f}% (median); the clean and "
                    "degraded sequences are inconsistent, e.g. one of them was re-rendered."
                )
        multiplier[~np.isfinite(degraded)] = np.nan

        values = _finite_values(clean, degraded)
        vmax = max(float(np.nanpercentile(values, 99.5)), np.finfo(np.float32).tiny)
        image_norm = ImageNormalize(vmin=0.0, vmax=vmax, stretch=AsinhStretch(1.0e-2))
        for column, image in enumerate((clean, degraded)):
            artist = axes[row, column].imshow(
                image, origin="lower", cmap=cm.soholasco2, norm=image_norm
            )
            fig.colorbar(artist, ax=axes[row, column], fraction=0.046, label="MSB")

        # The additive term is positive and as steep as the corona, so it takes
        # the stretch of the images with its own upper bound; on their scale only
        # its innermost ring would exceed the linear part of the stretch.
        additive_vmax = max(
            float(np.nanpercentile(_finite_values(additive), 99.5)), np.finfo(np.float32).tiny
        )
        additive_artist = axes[row, 2].imshow(
            additive,
            origin="lower",
            cmap="viridis",
            norm=ImageNormalize(vmin=0.0, vmax=additive_vmax, stretch=AsinhStretch(1.0e-2)),
        )
        fig.colorbar(
            additive_artist,
            ax=axes[row, 2],
            fraction=0.046,
            label="additive [MSB]",
        )

        multiplier_values = _finite_values(multiplier)
        multiplier_limit = max(
            float(np.nanpercentile(np.abs(multiplier_values - 1.0), 99.5)),
            np.finfo(np.float32).eps,
        )
        multiplier_artist = axes[row, 3].imshow(
            multiplier,
            origin="lower",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(
                vmin=1.0 - multiplier_limit,
                vcenter=1.0,
                vmax=1.0 + multiplier_limit,
            ),
        )
        fig.colorbar(
            multiplier_artist,
            ax=axes[row, 3],
            fraction=0.046,
            label="radial gain",
        )
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
