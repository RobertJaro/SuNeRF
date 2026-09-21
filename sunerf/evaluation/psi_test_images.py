#!/usr/bin/env python3
"""Compare clean, degraded, and recovered tB/pB images at fixed frame 050."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from sunerf.data.psi.psi_test_paths import (
    FRAME_TOKEN,
    PRODUCTS,
    channel_file,
)

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


OUTPUT_MAP_KEYS = {"tb": "tB_map", "pb": "pB_map"}
DISPLAY_CHANNEL = {"tb": "tB", "pb": "pB"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sunerf-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--degraded-dir", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--degraded-views", nargs="*", default=[])
    return parser.parse_args()


def comparison_metrics(
    reference: np.ndarray, estimate: np.ndarray
) -> dict[str, float | int]:
    valid = np.isfinite(reference) & np.isfinite(estimate)
    x = np.asarray(reference[valid], dtype=np.float64)
    y = np.asarray(estimate[valid], dtype=np.float64)
    if x.size < 2:
        raise ValueError(
            "Fewer than two finite pixels are available for image metrics."
        )
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    scale = float(np.percentile(x, 99) - np.percentile(x, 1))
    return {
        "n_pixels": int(x.size),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "rmse_msb": rmse,
        "nrmse_p01_p99": rmse / scale if scale > 0 else float("nan"),
    }


def _finite_vmax(arrays: list[np.ndarray]) -> float:
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    if values.size == 0:
        raise ValueError("Cannot plot a row without finite brightness values.")
    vmax = float(np.percentile(values, 99.5))
    return max(vmax, np.finfo(np.float32).tiny)


def plot_channel(
    channel: str,
    results: dict[str, dict[str, np.ndarray | None]],
    output_dir: Path,
    views: list[str],
    degraded_views: set[str],
) -> Path:
    fig, axes = plt.subplots(
        len(views),
        3,
        figsize=(12, max(4.0, 3.5 * len(views))),
        constrained_layout=True,
        squeeze=False,
    )
    titles = ("Original clean", "Synthetic degraded input", "Recovered clean")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    for row, view in enumerate(views):
        clean = results[view]["clean"]
        degraded = results[view]["degraded"]
        recovered = results[view]["recovered"]
        plotted = (
            [clean, recovered] if degraded is None else [clean, degraded, recovered]
        )
        norm = ImageNormalize(
            vmin=0.0,
            vmax=_finite_vmax(plotted),
            stretch=AsinhStretch(1.0e-2),
            clip=False,
        )
        images = (clean, degraded, recovered)
        image_artist = None
        for column, image in enumerate(images):
            axis = axes[row, column]
            if image is None:
                axis.axis("off")
                axis.text(
                    0.5,
                    0.5,
                    "withheld\n(no degraded input)",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="0.4",
                )
                continue
            image_artist = axis.imshow(
                image,
                origin="lower",
                cmap=cm.soholasco2,
                norm=norm,
                interpolation="none",
            )
            axis.set_xticks([])
            axis.set_yticks([])
        role = "degraded input" if view in degraded_views else "held out"
        axes[row, 0].set_ylabel(f"{view} ({role})")
        fig.colorbar(
            image_artist,
            ax=[axis for axis in axes[row] if axis.axison],
            fraction=0.025,
            pad=0.01,
            label="MSB",
        )

    fig.suptitle(
        f"PSI frame {FRAME_TOKEN}: {DISPLAY_CHANNEL[channel]} recovery", fontsize=15
    )
    output_path = output_dir / f"image_comparison_{DISPLAY_CHANNEL[channel]}.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    if not args.checkpoint_path.is_file():
        raise FileNotFoundError(
            "Degraded-data training has not completed "
            f"(missing {args.checkpoint_path})."
        )
    if not args.sunerf_path.is_file():
        raise FileNotFoundError(
            f"Run degraded-data training before evaluation: {args.sunerf_path}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    degraded_views = set(args.degraded_views)
    unknown = degraded_views - set(args.views)
    if unknown:
        raise ValueError(f"Degraded views are absent from --views: {sorted(unknown)}")
    # This file is produced locally by save_thomson_sunerf and contains pickled
    # PyTorch modules, including the learned detector-degradation corrections.
    loader = ThomsonSuNeRFLoader(args.sunerf_path, trusted=True)

    results = {}
    metrics = {"frame": FRAME_TOKEN, "units": "mean solar brightness", "views": {}}
    for view in args.views:
        clean_maps = {
            product: Map(channel_file(args.clean_dir, view, product))
            for product in PRODUCTS
        }
        recovered = loader.load_map(
            clean_maps["tb"],
            instrument_key=view,
            batch_size=8192,
            progress=True,
        )
        degraded_maps = (
            {
                product: Map(channel_file(args.degraded_dir, view, product))
                for product in PRODUCTS
            }
            if view in degraded_views
            else None
        )
        results[view] = {}
        metrics["views"][view] = {}
        for product in PRODUCTS:
            recovered_data = np.asarray(recovered[OUTPUT_MAP_KEYS[product]].data)
            clean_data = np.asarray(clean_maps[product].data)
            degraded_data = (
                np.asarray(degraded_maps[product].data)
                if degraded_maps is not None
                else None
            )
            results[view][product] = {
                "clean": clean_data,
                "degraded": degraded_data,
                "recovered": recovered_data,
            }
            channel_metrics = {
                "recovered_vs_clean": comparison_metrics(clean_data, recovered_data),
            }
            if degraded_data is not None:
                channel_metrics["degraded_vs_clean"] = comparison_metrics(
                    clean_data, degraded_data
                )
            metrics["views"][view][product] = channel_metrics

        if view in degraded_views:
            learned = loader.load_correction_masks(
                clean_maps["tb"], instrument_key=view, apply_valid_mask=True
            )
            truth = {
                "tB_add": np.load(args.truth_dir / view / "tb_additive.npy"),
                "pB_add": np.load(args.truth_dir / view / "pb_additive.npy"),
                "calibration_gain": np.load(
                    args.truth_dir / view / "tb_multiplier.npy"
                ),
            }
            learned_arrays = {
                "tB_add": learned["tB_add"].data * loader.msb_norm,
                "pB_add": learned["pB_add"].data * loader.msb_norm,
                "calibration_gain": learned["calibration_gain"].data,
            }
            metrics["views"][view]["correction_terms"] = {
                term: comparison_metrics(truth[term], learned_arrays[term])
                for term in truth
            }

    output_paths = []
    for product in PRODUCTS:
        per_channel = {view: results[view][product] for view in args.views}
        output_paths.append(
            plot_channel(
                product, per_channel, args.output_dir, args.views, degraded_views
            )
        )

    metrics_path = args.output_dir / "image_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    print("Image evaluations:")
    for output_path in output_paths:
        print(f"  {output_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()
