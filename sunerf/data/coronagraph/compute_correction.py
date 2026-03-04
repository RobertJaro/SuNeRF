import argparse
import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch
from sunpy.visualization.colormaps import cm
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Compute percentile correction mask from FITS stack."
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
        "--percentile",
        type=float,
        default=20.0,
        help="Percentile for correction mask (default: 20)",
    )
    parser.add_argument(
        "--use-min",
        action="store_true",
        help="Use minimum across stack instead of percentile",
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
    args = parser.parse_args()

    files = sorted(glob.glob(args.input))
    if not files:
        raise ValueError("No files matched --input pattern")

    stack = [fits.getdata(f) for f in tqdm(files, desc="Loading data")]
    stack = np.stack(stack, axis=0)
    if args.reference_index < 0 or args.reference_index >= stack.shape[0]:
        raise ValueError(f"--reference-index must be in [0, {stack.shape[0] - 1}]")

    if args.use_min:
        mask = np.min(stack, axis=0)
    else:
        mask = np.percentile(stack, args.percentile, axis=0)
    np.save(args.output, mask)

    reference = stack[args.reference_index]
    subtracted = reference - mask

    plot_output = args.plot_output if args.plot_output is not None else f"{args.output}.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    brightness_norm = ImageNormalize(stretch=AsinhStretch(1e-4), vmin=0, vmax=1e-9)
    subtracted_plot = subtracted.copy()
    subtracted_plot[subtracted_plot < 0] = np.nan

    im0 = axes[0].imshow(reference, cmap=cm.soholasco2, norm=brightness_norm, origin="lower")
    axes[0].set_title(f"Original (idx={args.reference_index})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mask, cmap=cm.soholasco2, norm=brightness_norm, origin="lower")
    axes[1].set_title("Correction Mask")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(subtracted_plot, cmap=cm.soholasco2, norm=brightness_norm, origin="lower")
    axes[2].set_title("Original - Mask")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    fig.savefig(plot_output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
