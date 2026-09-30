#!/usr/bin/env python3
"""Apply one detector degradation to one prepared PSI observer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy.ndimage import distance_transform_edt, map_coordinates
from sunpy.map import Map
from tqdm.auto import tqdm

from sunerf.data.psi.prepare_psi_clear import (
    PRODUCTS,
    paired_product_files,
    radial_rsun,
    save_prepared_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--noise-instrument",
        "--synthetic",
        dest="synthetic",
        required=True,
        help="Physical detector supplying the noise mask.",
    )
    parser.add_argument("--inner-rsun", type=float, default=None,
                        help="Inner field-of-view limit; default keeps the field of view of the data.")
    parser.add_argument("--outer-rsun", type=float, default=None,
                        help="Outer field-of-view limit; default keeps the field of view of the data.")
    parser.add_argument("--mask-fraction", type=float, default=0.1)
    parser.add_argument(
        "--noise-fraction", type=float, default=0.02,
        help="Photon-like noise: sigma = fraction * sqrt(I * I_ref) with I_ref the reference "
             "p99, i.e. the relative noise at the p99 level; 0 disables the noise.",
    )
    parser.add_argument(
        "--keep-background", action="store_true",
        help="Skip the per-pixel minimum background subtraction over the series.",
    )
    # The gain is optional; the defaults leave the images unscaled. It is the
    # product of a scalar calibration, an axisymmetric radial profile, and a
    # second-order azimuthal modulation.
    parser.add_argument("--calibration", type=float, default=1.0,
                        help="Scalar radiometric calibration factor of the view.")
    parser.add_argument("--azimuthal-gain", type=float, default=0.0,
                        help="Amplitude of the 1 + a * cos(2 * position_angle) gain modulation.")
    parser.add_argument("--inner-gain", type=float, default=1.0)
    parser.add_argument("--outer-gain", type=float, default=1.0)
    parser.add_argument("--gain-power", type=float, default=1.0)
    parser.add_argument("--pb-mask", type=Path, required=True)
    parser.add_argument("--tb-mask", type=Path, required=True)
    parser.add_argument("--truth-dir", type=Path, required=True)
    parser.add_argument("--truth-frame", default="050")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def degradation_config(
    *,
    synthetic: str,
    inner_rsun: float | None = None,
    outer_rsun: float | None = None,
    mask_fraction: float = 0.1,
    noise_fraction: float = 0.02,
    background_subtraction: bool = True,
    calibration: float = 1.0,
    azimuthal_gain: float = 0.0,
    inner_gain: float = 1.0,
    outer_gain: float = 1.0,
    gain_power: float = 1.0,
) -> dict:
    """Validate and normalize degradation keyword arguments."""
    config = {
        "synthetic": str(synthetic),
        "inner_rsun": None if inner_rsun is None else float(inner_rsun),
        "outer_rsun": None if outer_rsun is None else float(outer_rsun),
        "mask_fraction": float(mask_fraction),
        "noise_fraction": float(noise_fraction),
        "background_subtraction": bool(background_subtraction),
        "calibration": float(calibration),
        "azimuthal_gain": float(azimuthal_gain),
        "inner_gain": float(inner_gain),
        "outer_gain": float(outer_gain),
        "gain_power": float(gain_power),
    }
    if not config["synthetic"]:
        raise ValueError("synthetic must not be empty.")
    if (config["inner_rsun"] is None) != (config["outer_rsun"] is None):
        raise ValueError("inner_rsun and outer_rsun must be given together.")
    if config["inner_rsun"] is not None and not 0 <= config["inner_rsun"] < config["outer_rsun"]:
        raise ValueError("Require 0 <= inner_rsun < outer_rsun.")
    if not 0 < config["mask_fraction"] <= 1 or not np.isfinite(config["mask_fraction"]):
        raise ValueError("mask_fraction must be finite and in (0, 1].")
    if not 0 <= config["noise_fraction"] < 1 or not np.isfinite(config["noise_fraction"]):
        raise ValueError("noise_fraction must be finite and in [0, 1).")
    if any(
        not np.isfinite(config[key]) or config[key] <= 0
        for key in ("calibration", "inner_gain", "outer_gain", "gain_power")
    ):
        raise ValueError(
            "calibration, inner_gain, outer_gain, and gain_power must be positive and finite."
        )
    if not np.isfinite(config["azimuthal_gain"]) or not abs(config["azimuthal_gain"]) < 1:
        raise ValueError("azimuthal_gain must be finite with magnitude below one.")
    return config


def fill_nearest_finite(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    missing = ~np.isfinite(data)
    if not np.any(missing):
        return data.copy()
    if np.all(missing):
        raise ValueError("Detector noise mask contains no finite pixels.")
    nearest = distance_transform_edt(
        missing, return_distances=False, return_indices=True
    )
    return data[tuple(nearest)].astype(np.float32, copy=False)


def _sun_center(s_map: Map) -> tuple[float, float]:
    center = s_map.world_to_pixel(
        SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=s_map.coordinate_frame)
    )
    return center.x.to_value(u.pixel), center.y.to_value(u.pixel)


def valid_annulus(data: np.ndarray, center) -> tuple[float, float]:
    """Inner and outer pixel radius of the rings that are mostly finite."""
    rows, columns = np.indices(data.shape, dtype=np.float64)
    rings = np.rint(np.hypot(columns - center[0], rows - center[1])).astype(int)
    finite_fraction = np.bincount(rings.ravel(), weights=np.isfinite(data).ravel()) / np.bincount(rings.ravel())
    covered = np.flatnonzero(finite_fraction >= 0.5)
    if covered.size < 2:
        raise ValueError("Image has no valid annulus around the Sun.")
    return float(covered[0]), float(covered[-1])


def resample_detector_mask(mask_map: Map, target_map: Map) -> np.ndarray:
    """Map the valid annulus of the detector pattern onto that of the target image.

    The pattern is attached to the detector, not the Sun, so observer and time are
    ignored.  Radii between the occulter edge and the outer edge of the mask map
    linearly onto the same range of the target image at equal position angle, so
    the whole pattern covers the field of view whatever the two plate scales and
    occulter sizes are.  Gaps such as a pylon take the nearest valid value.
    """
    mask_center, target_center = _sun_center(mask_map), _sun_center(target_map)
    mask_inner, mask_outer = valid_annulus(mask_map.data, mask_center)
    target_inner, target_outer = valid_annulus(target_map.data, target_center)

    rows, columns = np.indices(target_map.data.shape, dtype=np.float64)
    dx, dy = columns - target_center[0], rows - target_center[1]
    radius = np.hypot(dx, dy)
    scale = (mask_outer - mask_inner) / (target_outer - target_inner)
    mask_radius = mask_inner + (radius - target_inner) * scale
    stretch = np.divide(mask_radius, radius, out=np.zeros_like(radius), where=radius > 0)
    sampled = map_coordinates(
        fill_nearest_finite(mask_map.data),
        [mask_center[1] + dy * stretch, mask_center[0] + dx * stretch],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return sampled.astype(np.float32, copy=False)


def additive_degradation_mask(
    mask: np.ndarray,
    image_level: float,
    fraction: float,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Create an additive background from an observational detector mask.

    The finite 1st and 99th mask percentiles map to zero and one. The normalized
    pattern defines a positive additive term capped at ``fraction * image_level``.
    """
    mask = np.asarray(mask, dtype=np.float32)
    selection = np.isfinite(mask)
    if valid is not None:
        if np.shape(valid) != mask.shape:
            raise ValueError("valid mask and detector mask must have matching shapes.")
        selection &= np.asarray(valid, dtype=bool)
    if not np.any(selection):
        raise ValueError(
            "No finite detector-mask pixels are available for normalization."
        )

    mask_low, mask_high = np.nanpercentile(mask[selection], (1.0, 99.0))
    if not np.isfinite(mask_low) or not np.isfinite(mask_high) or mask_high <= mask_low:
        raise ValueError(
            "Detector mask has no finite intensity range after normalization."
        )
    image_level = float(image_level)
    if not np.isfinite(image_level) or image_level <= 0:
        raise ValueError("image_level must be positive and finite.")
    normalized = np.clip((mask - mask_low) / (mask_high - mask_low), 0.0, 1.0)
    additive = float(fraction) * image_level * normalized
    additive[~selection] = np.nan
    return additive.astype(np.float32, copy=False)


def photon_noise(
    image: np.ndarray, image_level: float, fraction: float, seed: int
) -> np.ndarray:
    """Deterministic photon-like noise with ``sigma = fraction * sqrt(I * image_level)``.

    The noise is ``fraction`` of the signal at ``image_level`` (the reference
    p99) and grows relative to the signal toward fainter pixels as in shot noise.
    The same ``seed`` reproduces the same field, so a two-pass background
    estimate sees exactly the noise that is written.
    """
    if fraction <= 0:
        return np.zeros(np.shape(image), dtype=np.float32)
    sigma = float(fraction) * np.sqrt(np.clip(image, 0.0, None) * float(image_level))
    rng = np.random.default_rng(seed)
    return (sigma * rng.standard_normal(np.shape(image))).astype(np.float32)


def frame_seed(name: str) -> int:
    """Stable per-frame seed from the file name (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def radial_gain_mask(
    radius_rsun: np.ndarray,
    inner_rsun: float,
    outer_rsun: float,
    inner_gain: float,
    outer_gain: float,
    power: float,
) -> np.ndarray:
    """Create an axisymmetric gain varying as normalized radius to ``power``."""
    radius_rsun = np.asarray(radius_rsun, dtype=np.float32)
    normalized_radius = np.clip(
        (radius_rsun - float(inner_rsun)) / (float(outer_rsun) - float(inner_rsun)),
        0.0,
        1.0,
    )
    gain = float(inner_gain) + (float(outer_gain) - float(inner_gain)) * (
        normalized_radius ** float(power)
    )
    return gain.astype(np.float32, copy=False)


def azimuthal_gain_mask(
    shape: tuple[int, int], center: tuple[float, float], amplitude: float, order: int = 2
) -> np.ndarray:
    """Create a ``1 + amplitude * cos(order * position_angle)`` gain about ``center``.

    ``center`` is the Sun centre as ``(x, y)`` pixel coordinates. The default
    second order mimics the two-fold asymmetry of vignetting and polarizer
    non-uniformity; the mean over a full annulus is one.
    """
    rows, columns = np.indices(shape, dtype=np.float32)
    angle = np.arctan2(rows - float(center[1]), columns - float(center[0]))
    return (1.0 + float(amplitude) * np.cos(int(order) * angle)).astype(np.float32)


def load_detector_masks(pb_mask: Path, tb_mask: Path) -> dict[str, Map]:
    masks = {}
    for product, path in {"pb": pb_mask, "tb": tb_mask}.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {product} detector noise mask: {path}")
        mask = Map(path)
        if np.asarray(mask.data).ndim != 2:
            raise ValueError(
                f"{product} detector noise mask is not two-dimensional: {path}"
            )
        if not np.isfinite(mask.data).any():
            raise ValueError(
                f"{product} detector noise mask has no finite pixels: {path}"
            )
        masks[product] = mask
    return masks


def _write_array(path: Path, value: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def save_truth(
    truth_dir: Path,
    config: dict,
    additives: dict[str, np.ndarray],
    multipliers: dict[str, np.ndarray],
    valid: np.ndarray,
    overwrite: bool,
    masks: dict[str, np.ndarray] | None = None,
    backgrounds: dict[str, np.ndarray] | None = None,
) -> None:
    _write_array(
        truth_dir / "tb_additive.npy",
        np.where(valid, additives["tb"], np.nan),
        overwrite,
    )
    _write_array(
        truth_dir / "pb_additive.npy",
        np.where(valid, additives["pb"], np.nan),
        overwrite,
    )
    _write_array(
        truth_dir / "tb_multiplier.npy",
        np.where(valid, multipliers["tb"], np.nan),
        overwrite,
    )
    _write_array(
        truth_dir / "pb_multiplier.npy",
        np.where(valid, multipliers["pb"], np.nan),
        overwrite,
    )
    _write_array(truth_dir / "valid_mask.npy", valid, overwrite)
    for name, arrays in (("mask", masks), ("background", backgrounds)):
        if arrays is None:
            continue
        for product in PRODUCTS:
            _write_array(
                truth_dir / f"{product}_{name}.npy",
                np.where(valid, arrays[product], np.nan),
                overwrite,
            )
    metadata = {
        **config,
        "formula": "degraded[channel] = (clear[channel] + additive[channel]) * multiplier[channel]",
        "image_geometry_preserved": True,
        "additive_masks_separate": True,
        "multiplicative_masks_separate": False,
        "multiplicative_gain_shared_between_channels": True,
        "additive_and_multiplicative_patterns_correlated": False,
        "mask_geometry": "detector-centered linear WCS",
        "mask_normalization": "clip((mask - p01) / (p99 - p01), 0, 1)",
        "additive_peak_fraction_of_reference_p99": config["mask_fraction"],
        "noise_sigma": "noise_fraction * sqrt(I * p99(reference))",
        "noise_fraction": config["noise_fraction"],
        "background_subtraction": config["background_subtraction"],
        "background_estimate": "per-pixel minimum of the noisy degraded series",
        "additive_definition": (
            "mask - background / multiplier, so that the saved images are "
            "(clean + additive) * multiplier + noise"
        ),
        "multiplier_geometry": "scalar calibration * radial power law * second-order azimuthal modulation",
        "multiplier_formula": (
            "calibration * (inner_gain + (outer_gain - inner_gain) * "
            "clip((r - inner_rsun) / (outer_rsun - inner_rsun), 0, 1) ** gain_power) * "
            "(1 + azimuthal_gain * cos(2 * position_angle))"
        ),
        "calibration": config["calibration"],
        "azimuthal_gain": config["azimuthal_gain"],
        "multiplier_endpoints": {
            "inner": config["inner_gain"],
            "outer": config["outer_gain"],
        },
        "missing_mask_policy": "nearest finite detector value",
        "fov_unit": "R_sun",
        "shape": list(valid.shape),
    }
    metadata_path = truth_dir / "metadata.json"
    if overwrite or not metadata_path.exists():
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def degrade_sequence(
    input_dir: Path,
    output_dir: Path,
    truth_dir: Path,
    config: dict,
    mask_maps: dict[str, Map],
    truth_frame: str,
    overwrite: bool,
) -> None:
    input_pairs = paired_product_files(input_dir)
    reference_pairs = [
        paths for paths in input_pairs if paths["tb"].stem.endswith(truth_frame)
    ]
    if len(reference_pairs) != 1:
        raise FileNotFoundError(
            f"Expected one degradation reference frame *{truth_frame}.fts in {input_dir}; "
            f"found {len(reference_pairs)}."
        )

    def prepare_frame(input_paths):
        target_maps = {product: Map(path) for product, path in input_paths.items()}
        if target_maps["tb"].date != target_maps["pb"].date:
            raise ValueError(
                f"tB/pB timestamps differ: {input_paths['tb'].name}, "
                f"{input_paths['pb'].name}"
            )
        mask_patterns = {
            product: resample_detector_mask(mask_maps[product], target_maps[product])
            for product in PRODUCTS
        }
        source_valid = np.ones(target_maps["tb"].data.shape, dtype=bool)
        for product in PRODUCTS:
            source_valid &= np.isfinite(target_maps[product].data)
            source_valid &= np.isfinite(mask_patterns[product])
        return target_maps, mask_patterns, source_valid

    reference_maps, _, reference_valid = prepare_frame(reference_pairs[0])
    image_levels = {
        product: float(
            np.nanpercentile(
                np.abs(reference_maps[product].data[reference_valid]), 99.0
            )
        )
        for product in PRODUCTS
    }
    # Without explicit limits the data keep their field of view, whose radial
    # extent anchors the gain profile.
    clip_fov = config["inner_rsun"] is not None
    if clip_fov:
        inner_rsun, outer_rsun = config["inner_rsun"], config["outer_rsun"]
    else:
        reference_radius = radial_rsun(reference_maps["tb"])[reference_valid]
        inner_rsun, outer_rsun = float(reference_radius.min()), float(reference_radius.max())
    runtime_config = {
        **config,
        "inner_rsun": inner_rsun,
        "outer_rsun": outer_rsun,
        "mask_reference_frame": truth_frame,
        "mask_intensity_levels": image_levels,
    }

    def degrade_frame(input_paths):
        """Noisy degraded images of one frame, its injected mask and multiplier."""
        target_maps, mask_patterns, source_valid = prepare_frame(input_paths)
        radius = radial_rsun(target_maps["tb"])
        multiplier = (
            config["calibration"]
            * radial_gain_mask(
                radius,
                inner_rsun,
                outer_rsun,
                config["inner_gain"],
                config["outer_gain"],
                config["gain_power"],
            )
            * azimuthal_gain_mask(
                radius.shape, _sun_center(target_maps["tb"]), config["azimuthal_gain"]
            )
        )
        valid = source_valid
        if clip_fov:
            valid = valid & (radius >= inner_rsun) & (radius <= outer_rsun)
        masks, degraded = {}, {}
        for product, input_path in input_paths.items():
            clear = np.asarray(target_maps[product].data, dtype=np.float32)
            masks[product] = additive_degradation_mask(
                mask_patterns[product],
                image_levels[product],
                config["mask_fraction"],
                source_valid,
            )
            image = (clear + masks[product]) * multiplier
            image = image + photon_noise(
                image, image_levels[product], config["noise_fraction"], frame_seed(input_path.name)
            )
            image[~valid] = np.nan
            degraded[product] = image
        return target_maps, degraded, masks, multiplier, valid

    # Pass 1: the background estimate is the per-pixel minimum of the noisy
    # degraded series, as a daily-min correction of real data would find it.
    backgrounds = None
    if config["background_subtraction"]:
        backgrounds = {product: None for product in PRODUCTS}
        for input_paths in tqdm(input_pairs, desc=f"Background {input_dir.name}"):
            _, degraded, _, _, _ = degrade_frame(input_paths)
            for product in PRODUCTS:
                current = backgrounds[product]
                backgrounds[product] = (
                    degraded[product] if current is None else np.fmin(current, degraded[product])
                )

    # Pass 2: subtract the estimate and write the frames. Pixel values are
    # clipped to the FOV only; shape, WCS, and pixel scale stay as in the clean map.
    truth_written = False
    for input_paths in tqdm(input_pairs, desc=f"Degrading {input_dir.name}"):
        target_maps, degraded, masks, multiplier, valid = degrade_frame(input_paths)
        for product, input_path in input_paths.items():
            image = degraded[product]
            if backgrounds is not None:
                image = image - backgrounds[product]
            image[~valid] = np.nan
            save_prepared_map(
                image,
                target_maps[product],
                output_dir / product / input_path.name,
                runtime_config,
                overwrite,
            )
        if input_paths["tb"].stem.endswith(truth_frame):
            # The additive truth keeps the (clean + additive) * multiplier
            # convention: the time-constant residual the saved images carry.
            additives = {
                product: masks[product]
                - (0.0 if backgrounds is None else backgrounds[product] / multiplier)
                for product in PRODUCTS
            }
            multipliers = {product: multiplier for product in PRODUCTS}
            save_truth(
                truth_dir, runtime_config, additives, multipliers, valid, overwrite,
                masks=masks, backgrounds=backgrounds,
            )
            truth_written = True
    if not truth_written:
        raise FileNotFoundError(
            f"Could not save degradation truth: frame *{truth_frame}.fts is absent in {input_dir}."
        )


def main() -> None:
    args = parse_args()
    config = degradation_config(
        synthetic=args.synthetic,
        inner_rsun=args.inner_rsun,
        outer_rsun=args.outer_rsun,
        mask_fraction=args.mask_fraction,
        noise_fraction=args.noise_fraction,
        background_subtraction=not args.keep_background,
        calibration=args.calibration,
        azimuthal_gain=args.azimuthal_gain,
        inner_gain=args.inner_gain,
        outer_gain=args.outer_gain,
        gain_power=args.gain_power,
    )
    degrade_sequence(
        args.input_dir,
        args.output_dir,
        args.truth_dir,
        config,
        load_detector_masks(args.pb_mask, args.tb_mask),
        args.truth_frame,
        args.overwrite,
    )
    print(f"Degraded observer sequence: {args.output_dir}")


if __name__ == "__main__":
    main()
