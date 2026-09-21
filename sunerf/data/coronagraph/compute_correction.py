import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, percentile_filter
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


def compute_correction_mask(
    stack,
    per_day_images,
    correction_type,
    percentile=5.0,
    min_frames_per_day=6,
):
    if correction_type in ("daily-min", "daily-percentile"):
        daily_medians = compute_daily_medians(
            per_day_images, min_frames_per_day=min_frames_per_day
        )
    if correction_type == "daily-min":
        return np.nanmin(daily_medians, axis=0)
    if correction_type == "daily-percentile":
        return np.nanpercentile(daily_medians, percentile, axis=0)
    if correction_type == "full-min":
        return np.nanmin(stack, axis=0)
    raise ValueError(f"Unknown --type value: {correction_type}")


def smooth_polar(mask: np.ndarray, center, angle_deg: float, radius_pixels: float,
                 percentile: float | None = 0.0) -> np.ndarray:
    """Smooth around the Sun in position angle and along the radius.

    Streamers are radial and tens of degrees wide, so an isotropic filter hardly
    touches them.  The mask is processed on a (radius, position angle) grid
    instead, periodically in angle.  A running ``percentile`` over the position
    angles within ``+-angle_deg`` first removes the streamers while following the
    lower envelope of the mask, so the result does not exceed the data between
    them; a Gaussian of half that width then removes the steps of the rank
    filter.  ``percentile=None`` applies a plain Gaussian of width ``angle_deg``,
    an angular mean that lies above the mask between streamers.  Missing pixels
    are ignored and stay missing.
    """
    rows, columns = np.indices(mask.shape, dtype=np.float64)
    dx, dy = columns - center[0], rows - center[1]
    radius, angle = np.hypot(dx, dy), np.arctan2(dy, dx)

    n_radius = int(np.ceil(radius.max())) + 1
    n_angle = int(np.clip(np.ceil(2.0 * np.pi * n_radius / 4.0), 360, 1440))
    grid_angle = -np.pi + (np.arange(n_angle) + 0.5) * 2.0 * np.pi / n_angle
    grid_radius, grid_angle = np.meshgrid(np.arange(n_radius, dtype=np.float64), grid_angle, indexing="ij")
    source = [center[1] + grid_radius * np.sin(grid_angle), center[0] + grid_radius * np.cos(grid_angle)]

    finite = np.isfinite(mask)
    value = map_coordinates(np.where(finite, mask, 0.0).astype(np.float32), source, order=1, mode="constant", cval=0.0)
    weight = map_coordinates(finite.astype(np.float32), source, order=1, mode="constant", cval=0.0)
    bins_per_degree = n_angle / 360.0
    sigma_angle = angle_deg * bins_per_degree

    if percentile is not None and angle_deg > 0:
        valid = weight > 0.5
        # Missing samples rank above every value and drop out of low percentiles.
        ranked = np.where(valid, value / np.where(valid, weight, 1.0), np.inf)
        window = min(2 * int(round(sigma_angle)) + 1, n_angle)
        ranked = percentile_filter(ranked, percentile, size=(1, window), mode="wrap")
        valid = np.isfinite(ranked)
        value, weight = np.where(valid, ranked, 0.0).astype(np.float32), valid.astype(np.float32)
        sigma_angle = 0.5 * sigma_angle

    sigma = (max(radius_pixels, 1e-3), max(sigma_angle, 1e-3))
    polar = {
        "value": gaussian_filter(value, sigma, mode=("nearest", "wrap")),
        "weight": gaussian_filter(weight, sigma, mode=("nearest", "wrap")),
    }

    # One wrapped column on either side closes the periodic seam for the interpolation.
    target = [radius, (angle + np.pi) / (2.0 * np.pi) * n_angle + 0.5]
    smoothed = {
        name: map_coordinates(np.pad(values, ((0, 0), (1, 1)), mode="wrap"), target, order=1, mode="nearest")
        for name, values in polar.items()
    }
    covered = smoothed["weight"] > 1e-6
    # Image corners lie beyond every sampled ring and keep their value.
    result = np.where(covered, smoothed["value"] / np.where(covered, smoothed["weight"], 1.0), mask)
    result[~finite] = np.nan
    return result.astype(np.float32, copy=False)


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
    parser.add_argument(
        "--min-frames-per-day",
        type=int,
        default=6,
        help=(
            "Minimum frames required for a daily median (default: 6). "
            "Only used by daily-min and daily-percentile."
        ),
    )
    parser.add_argument(
        "--smooth-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing width in position angle around the Sun, which "
            "suppresses radial streamer structure (default: 0, no smoothing)."
        ),
    )
    parser.add_argument(
        "--smooth-fraction",
        type=float,
        default=0.0,
        help=(
            "Gaussian smoothing width along the radius as a fraction of the "
            "image width (default: 0). Large values flatten the radial falloff."
        ),
    )
    parser.add_argument(
        "--smooth-percentile",
        type=percentile_value,
        default=0.0,
        help=(
            "Running percentile over the position angles within +-smooth-angle-deg "
            "(default: 0, the running minimum). Low values follow the lower envelope "
            "of the mask, so subtracting it does not turn the corona between "
            "streamers negative."
        ),
    )
    args = parser.parse_args()
    if args.min_frames_per_day < 1:
        parser.error("--min-frames-per-day must be positive")
    if args.smooth_angle_deg < 0 or not 0 <= args.smooth_fraction < 1:
        parser.error("require --smooth-angle-deg >= 0 and --smooth-fraction in [0, 1)")

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
        min_frames_per_day=args.min_frames_per_day,
    )

    unsmoothed = None
    if args.smooth_angle_deg > 0 or args.smooth_fraction > 0:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        reference_map = sunpy.map.Map(files[args.reference_index])
        sun_center = reference_map.world_to_pixel(
            SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=reference_map.coordinate_frame)
        )
        unsmoothed = mask
        mask = smooth_polar(
            mask,
            (sun_center.x.to_value(u.pixel), sun_center.y.to_value(u.pixel)),
            args.smooth_angle_deg,
            args.smooth_fraction * mask.shape[1],
            args.smooth_percentile,
        )

    np.save(args.output, mask)

    reference = stack[args.reference_index]
    subtracted = reference - mask

    plot_output = args.plot_output if args.plot_output is not None else f"{args.output}.png"
    n_panels = 3 if unsmoothed is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), layout="constrained")

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

    if unsmoothed is not None:
        im3 = axes[3].imshow(unsmoothed, cmap=cm.soholasco2, norm=ImageNormalize(stretch=AsinhStretch(1e-4)), origin="lower")
        axes[3].set_title("Mask before smoothing")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        im4 = axes[4].imshow(unsmoothed / mask, cmap="RdBu_r", vmin=0.5, vmax=1.5, origin="lower")
        axes[4].set_title(f"Before / after (angle {args.smooth_angle_deg:g} deg, radius {args.smooth_fraction:g})")
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    fig.savefig(plot_output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
