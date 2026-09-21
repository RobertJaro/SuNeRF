#!/usr/bin/env python3
"""Prepare every downloaded PSI observer with one common clear-image recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm.auto import tqdm

from sunerf.data.psi.prep_psi_cme import load_fixed_map
from sunerf.data.ray_sampling import hpc_angular_separation, hpc_impact_parameter


PRODUCTS = ("pb", "tb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inner-rsun", type=float, default=2.0)
    parser.add_argument("--outer-rsun", type=float, default=30.0)
    parser.add_argument(
        "--resolution", nargs=2, type=int, default=(512, 512), metavar=("NX", "NY")
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_observers(input_dir: Path) -> list[str]:
    """Discover all directories containing non-empty pB and tB sequences."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"PSI input directory does not exist: {input_dir}")
    observers = [
        candidate.name
        for candidate in sorted(path for path in input_dir.iterdir() if path.is_dir())
        if all(any((candidate / product).glob("*.fts")) for product in PRODUCTS)
    ]
    if not observers:
        raise FileNotFoundError(
            f"No <observer>/pb and <observer>/tb FITS sequences found under {input_dir}."
        )
    return observers


def paired_product_files(input_dir: Path) -> list[dict[str, Path]]:
    product_files = {
        product: sorted((input_dir / product).glob("*.fts")) for product in PRODUCTS
    }
    counts = {product: len(paths) for product, paths in product_files.items()}
    if counts["tb"] != counts["pb"] or counts["tb"] == 0:
        raise ValueError(
            f"{input_dir} must contain equal nonzero tB/pB counts; got "
            f"{counts['tb']}/{counts['pb']}."
        )
    pairs = []
    for tb_path, pb_path in zip(product_files["tb"], product_files["pb"]):
        tb_key = tb_path.stem.replace("_tb", "_brightness", 1).lower()
        pb_key = pb_path.stem.replace("_pb", "_brightness", 1).lower()
        if tb_key != pb_key:
            raise ValueError(f"Unpaired PSI files: {tb_path.name}, {pb_path.name}")
        pairs.append({"tb": tb_path, "pb": pb_path})
    return pairs


def crop_to_fov(source_map: Map, outer_rsun: float, output_shape: tuple[int, int]) -> Map:
    """Crop a PSI map to a square projected-radius FOV and resample it."""
    # Elongation of a line of sight with an impact parameter of ``outer_rsun``.
    # Scaling the apparent solar radius linearly is a small-angle approximation
    # that clips the outermost valid pixels of wide fields of view.
    outer_fraction = (outer_rsun * u.R_sun / source_map.dsun).to_value(u.dimensionless_unscaled)
    if not 0 < outer_fraction < 1:
        raise ValueError("outer_rsun must lie between the Sun and the observer.")
    outer_fov_arcsec = (np.arcsin(outer_fraction) * u.rad).to_value(u.arcsec)
    bottom_left = SkyCoord(
        -outer_fov_arcsec * u.arcsec,
        -outer_fov_arcsec * u.arcsec,
        frame=source_map.coordinate_frame,
    )
    top_right = SkyCoord(
        outer_fov_arcsec * u.arcsec,
        outer_fov_arcsec * u.arcsec,
        frame=source_map.coordinate_frame,
    )
    cropped = source_map.submap(bottom_left, top_right=top_right)
    resampled = cropped.resample(np.asarray(output_shape[::-1]) * u.pixel)
    data = np.asarray(resampled.data, dtype=np.float32).copy()
    data[(~np.isfinite(data)) | (data <= 0)] = np.nan
    return Map(data, resampled.meta)


def radial_arcsec(s_map: Map) -> np.ndarray:
    """Exact angular separation of every pixel from Sun centre in arcsec."""
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    return hpc_angular_separation(coords.Tx, coords.Ty).to_value(u.arcsec).astype(np.float32)


def radial_rsun(s_map: Map) -> np.ndarray:
    """Line-of-sight impact parameter in R_sun, as used by the training loader."""
    coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    return hpc_impact_parameter(
        coords.Tx, coords.Ty, s_map.dsun
    ).to_value(u.R_sun).astype(np.float32)


def save_prepared_map(
    data: np.ndarray, template: Map, path: Path, metadata: dict, overwrite: bool
) -> None:
    if path.exists() and not overwrite:
        return
    meta = template.meta.copy()
    meta["SYNTH"] = metadata.get("synthetic", "CLEAR")
    meta["OCCMIN"] = float(metadata["inner_rsun"])
    meta["OCCMAX"] = float(metadata["outer_rsun"])
    meta["OCCUNIT"] = "R_sun"
    if "mask_fraction" in metadata:
        meta["MASKFR"] = float(metadata["mask_fraction"])
    if "inner_gain" in metadata:
        meta["GAININ"] = float(metadata["inner_gain"])
        meta["GAINOUT"] = float(metadata["outer_gain"])
        meta["GAINPOW"] = float(metadata["gain_power"])
    if "mask_fraction" in metadata and metadata.get("synthetic") != "CLEAR":
        meta["DEGFORM"] = "(clear+additive)*multiplier"
    path.parent.mkdir(parents=True, exist_ok=True)
    Map(np.asarray(data, dtype=np.float32), meta).save(path, overwrite=overwrite)


def prepare_observer(
    input_dir: Path,
    output_dir: Path,
    inner_rsun: float,
    outer_rsun: float,
    output_shape: tuple[int, int],
    overwrite: bool,
) -> None:
    for input_paths in tqdm(
        paired_product_files(input_dir), desc=f"Preparing clear {input_dir.name}"
    ):
        maps = {
            product: crop_to_fov(load_fixed_map(path), outer_rsun, output_shape)
            for product, path in input_paths.items()
        }
        if maps["tb"].date != maps["pb"].date:
            raise ValueError(
                f"tB/pB timestamps differ: {input_paths['tb'].name}, "
                f"{input_paths['pb'].name}"
            )
        radius = radial_rsun(maps["tb"])
        valid = (radius >= inner_rsun) & (radius <= outer_rsun)
        for product in PRODUCTS:
            valid &= np.isfinite(maps[product].data)
        metadata = {
            "synthetic": "CLEAR",
            "inner_rsun": inner_rsun,
            "outer_rsun": outer_rsun,
        }
        for product, input_path in input_paths.items():
            data = np.asarray(maps[product].data, dtype=np.float32).copy()
            data[~valid] = np.nan
            save_prepared_map(
                data,
                maps[product],
                output_dir / product / input_path.name,
                metadata,
                overwrite,
            )


def main() -> None:
    args = parse_args()
    if not 0 <= args.inner_rsun < args.outer_rsun:
        raise ValueError("Require 0 <= inner-rsun < outer-rsun.")
    output_shape = tuple(args.resolution)
    if any(size < 1 for size in output_shape):
        raise ValueError("--resolution values must be positive.")
    observers = discover_observers(args.input_dir)
    for observer in observers:
        prepare_observer(
            args.input_dir / observer,
            args.output_dir / observer,
            args.inner_rsun,
            args.outer_rsun,
            output_shape,
            args.overwrite,
        )
    print(f"Prepared {len(observers)} clear observer sequence(s): {args.output_dir}")


if __name__ == "__main__":
    main()
