import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def percentile_value(value):
    value = float(value)
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError("percentile must be between 0 and 100")
    return value


def compute_daily_medians(per_day_images, min_frames_per_day=6):
    daily_medians = []
    for day in sorted(per_day_images):
        images = per_day_images[day]
        if len(images) < min_frames_per_day:
            print(f"Skipping day {day} with only {len(images)} frames")
            continue
        day_stack = np.stack(images, axis=0)
        daily_medians.append(np.nanmedian(day_stack, axis=0))
    if not daily_medians:
        raise ValueError("No valid days available to compute daily correction mask.")
    return np.stack(daily_medians, axis=0)


def compute_correction_mask(stack, per_day_images, correction_type, percentile=5.0):
    if correction_type in ("daily-min", "daily-percentile"):
        daily_medians = compute_daily_medians(per_day_images)
    if correction_type == "daily-min":
        return np.nanmin(daily_medians, axis=0)
    if correction_type == "daily-percentile":
        return np.nanpercentile(daily_medians, percentile, axis=0)
    if correction_type == "full-min":
        return np.nanmin(stack, axis=0)
    raise ValueError(f"Unknown --type value: {correction_type}")


def main():
    import matplotlib.pyplot as plt
    import sunpy.map
    from astropy.visualization import ImageNormalize, AsinhStretch
    from sunpy.visualization.colormaps import cm

    parser = argparse.ArgumentParser(
        description="Compute a temporal correction mask from SunPy maps."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Glob pattern to FITS files (e.g. '/path/to/*.fits')",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .npy file",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional output path for a diagnostic plot (default: <output>.png)",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=0,
        help="Index of the frame used as reference for diagnostics (default: 0)",
    )
    parser.add_argument(
        "--type",
        choices=("daily-min", "daily-percentile", "full-min"),
        default="full-min",
        help=(
            "Correction type: minimum over daily medians (daily-min), "
            "percentile over daily medians (daily-percentile), or minimum "
            "over the full dataset (full-min)."
        ),
    )
    parser.add_argument(
        "--percentile",
        type=percentile_value,
        default=5.0,
        help="Percentile over daily medians for daily-percentile (default: 5).",
    )
    args = parser.parse_args()

    # create output directory if it doesn't exist
    output_dir = Path(args.output).parent
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(args.input))
    if not files:
        raise ValueError("No files matched --input pattern")

    stack = []
    per_day_images = defaultdict(list)
    for file_path in tqdm(files, desc="Loading maps"):
        m = sunpy.map.Map(file_path)
        image = np.asarray(m.data)
        stack.append(image)
        if args.type in ("daily-min", "daily-percentile"):
            day = m.date.isot[:10]
            per_day_images[day].append(image)
    stack = np.stack(stack, axis=0)
    if args.reference_index < 0 or args.reference_index >= stack.shape[0]:
        raise ValueError(f"--reference-index must be in [0, {stack.shape[0] - 1}]")

    mask = compute_correction_mask(
        stack,
        per_day_images,
        correction_type=args.type,
        percentile=args.percentile,
    )

    np.save(args.output, mask)

    reference = stack[args.reference_index]
    subtracted = reference - mask

    plot_output = args.plot_output if args.plot_output is not None else f"{args.output}.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    subtracted_plot = subtracted.copy()
    subtracted_plot[subtracted_plot < 0] = np.nan

    im0 = axes[0].imshow(reference, cmap=cm.soholasco2, norm=ImageNormalize(stretch=AsinhStretch(1e-4)), origin="lower")
    axes[0].set_title(f"Original (idx={args.reference_index})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mask, cmap=cm.soholasco2, norm=ImageNormalize(stretch=AsinhStretch(1e-4)), origin="lower")
    axes[1].set_title("Correction Mask")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(subtracted_plot, cmap=cm.soholasco2, norm=ImageNormalize(stretch=AsinhStretch(1e-4)), origin="lower")
    axes[2].set_title("Original - Mask")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    fig.savefig(plot_output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
