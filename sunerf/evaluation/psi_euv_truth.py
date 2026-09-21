"""Compare an EUV plasma reconstruction with the PSI/MAS cube it was rendered from.

The synthetic observations of ``sunerf-render-psi`` and the reconstruction use
the same forward model, so the differences reported here are reconstruction
errors (limited views, network resolution, degeneracies), not physics errors.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from sunerf.data.psi.render_euv import MAS_DENSITY_UNIT_CM3, MAS_TEMPERATURE_UNIT_K
from sunerf.data.psi.spherical_grid import _frame_id, load_psi_grid, pair_psi_files
from sunerf.train.coordinate_transformation import spherical_to_cartesian


RADIAL_BINS = ((1.02, 1.1), (1.1, 1.3), (1.3, 1.5))


def evaluation_points(radius, latitude, longitude, time=0.0):
    """Cartesian ``(x, y, z, t)`` query points on a spherical evaluation grid."""
    rr, aa, ll = np.meshgrid(radius, latitude, longitude, indexing="ij")
    spherical = torch.as_tensor(np.stack([rr, aa, ll], axis=-1), dtype=torch.float32)
    xyz = spherical_to_cartesian(spherical, torch)
    return torch.cat([xyz, torch.full_like(xyz[..., :1], float(time))], dim=-1)


@torch.no_grad()
def query_model(model, points, batch_size=262144, opacity=None):
    """Query ``log10 n_e`` and ``log10 T``; with ``opacity`` also the absorber.

    The absorber is returned as the equivalent cool hydrogen density
    ``alpha / kappa`` of the first opacity channel, so a simulation whose
    chromosphere absorbs through equilibrium ion fractions and a reconstruction
    with a separate cool field are compared in the same unit.
    """
    flat = points.reshape(-1, 4)
    density, temperature, absorber = [], [], []
    for start in range(0, flat.shape[0], batch_size):
        output = model(flat[start:start + batch_size])
        density.append(output["total_log_ne"].reshape(-1).cpu())
        temperature.append(output["mean_log_T"].reshape(-1).cpu())
        if opacity is not None:
            state = opacity.opacity(
                total_ne=output["total_ne"].cpu(), total_log_ne=output["total_log_ne"].cpu(),
                mean_log_T=output["mean_log_T"].cpu(),
                **({"cool_hydrogen_density": output["cool_hydrogen_density"].cpu()}
                   if "cool_hydrogen_density" in output else {}),
            )
            kappa = opacity.cool_cross_section_per_hydrogen_cm2[0]
            absorber.append((state["alpha_cm_inverse"][..., 0] / kappa).reshape(-1))
    shape = points.shape[:-1]
    fields = (
        torch.cat(density).reshape(shape).numpy(),
        torch.cat(temperature).reshape(shape).numpy(),
    )
    if opacity is None:
        return fields
    return (*fields, torch.cat(absorber).reshape(shape).numpy())


def compare_absorber(truth, reconstruction, radius, latitude, radial_bins=RADIAL_BINS):
    """Shell-integrated equivalent cool hydrogen content, truth versus reconstruction."""
    solar_radius_cm = 6.957e10
    shell_volume = (
        radius[:, None, None] ** 2 * np.cos(latitude)[None, :, None] * solar_radius_cm**3
    )
    report = {}
    for lower, upper in ((radius[0], RADIAL_BINS[0][0]), *radial_bins):
        shell = ((radius >= lower) & (radius < upper))[:, None, None]
        true_total = float(np.nansum(np.where(shell, truth * shell_volume, 0.0)))
        total = float(np.nansum(np.where(shell, reconstruction * shell_volume, 0.0)))
        report[f"{lower:.2f}-{upper:.2f}"] = {
            "truth_relative_content": true_total,
            "reconstruction_relative_content": total,
            "ratio": total / true_total if true_total > 0 else None,
        }
    return report


def error_statistics(truth, reconstruction, mask):
    difference = (reconstruction - truth)[mask]
    if difference.size == 0:
        return None
    return {
        "count": int(difference.size),
        "median_bias_dex": float(np.median(difference)),
        "median_absolute_error_dex": float(np.median(np.abs(difference))),
        "p16_dex": float(np.percentile(difference, 16)),
        "p84_dex": float(np.percentile(difference, 84)),
    }


def compare(truth, reconstruction, radius, *, emitting_log_T=None, radial_bins=RADIAL_BINS):
    """Radially binned errors of ``log10 n_e`` and ``log10 T``.

    ``emitting_log_T`` restricts the comparison to plasma hotter than the
    emission cutoff: colder plasma without an absorption signature is not
    constrained by the observations and is reported separately.
    """
    true_density, true_temperature = truth
    density, temperature = reconstruction
    valid = np.isfinite(true_density) & np.isfinite(true_temperature) & (true_density > -20.0)
    emitting = valid if emitting_log_T is None else valid & (true_temperature >= emitting_log_T)
    report = {}
    for lower, upper in radial_bins:
        shell = ((radius >= lower) & (radius < upper))[:, None, None]
        report[f"{lower:.2f}-{upper:.2f}"] = {
            "log_ne_emitting": error_statistics(true_density, density, emitting & shell),
            "log_T_emitting": error_statistics(true_temperature, temperature, emitting & shell),
            "log_ne_all": error_statistics(true_density, density, valid & shell),
            "cool_fraction": float(np.mean(~emitting[valid & shell])) if np.any(valid & shell) else None,
        }
    return report


def plot_profiles(path, radius, truth, reconstruction):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, true_field, field, label in (
        (axes[0], truth[0], reconstruction[0], r"$\\log_{10} n_e$ [cm$^{-3}$]"),
        (axes[1], truth[1], reconstruction[1], r"$\\log_{10} T$ [K]"),
    ):
        for values, name, style in ((true_field, "PSI/MAS", "k-"), (field, "SuNeRF", "C1--")):
            flat = np.where(np.isfinite(values), values, np.nan).reshape(radius.size, -1)
            axis.plot(radius, np.nanmedian(flat, axis=1), style, label=name)
            axis.fill_between(
                radius, np.nanpercentile(flat, 16, axis=1), np.nanpercentile(flat, 84, axis=1),
                color="k" if name == "PSI/MAS" else "C1", alpha=0.15,
            )
        axis.set_xlabel(r"$r$ [$R_\\odot$]")
        axis.set_ylabel(label)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--psi-data", required=True, help="directory with rho/ and t/ cubes")
    parser.add_argument("--reconstruction", required=True, help="plasma .safe.pt artifact")
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--min-radius", type=float, default=1.0)
    parser.add_argument("--max-radius", type=float, default=1.5)
    parser.add_argument("--n-radius", type=int, default=64)
    parser.add_argument("--n-latitude", type=int, default=90)
    parser.add_argument("--n-longitude", type=int, default=180)
    parser.add_argument("--emitting-temperature-k", type=float, default=5.0e5,
                        help="Plasma colder than this is reported separately.")
    parser.add_argument(
        "--frame-id", type=int, default=None,
        help="PSI snapshot used as truth (default: the first snapshot).",
    )
    parser.add_argument(
        "--time", default=None,
        help="ISO snapshot (emission) time at which the reconstruction is evaluated; "
             "default: the reconstruction reference date. With light travel time "
             "enabled the model time axis is the emission time, not the detector time.",
    )
    parser.add_argument(
        "--absorption-artifact", default="builtin:h_he_photoionization",
        help="Bundle used to express truth and reconstruction absorbers in one unit; "
             "'none' skips the absorber comparison.",
    )
    parser.add_argument("--save-truth", action="store_true",
                        help="Also store the truth and reconstruction on the evaluation grid.")
    return parser


def main(argv=None) -> int:
    from sunerf.evaluation.loader import PlasmaSuNeRFLoader

    args = build_parser().parse_args(argv)
    output = Path(args.out_path)
    output.mkdir(parents=True, exist_ok=True)
    radius = np.linspace(args.min_radius, args.max_radius, args.n_radius)
    latitude = np.deg2rad(np.linspace(-89.0, 89.0, args.n_latitude))
    longitude = np.deg2rad(np.linspace(0.0, 360.0, args.n_longitude, endpoint=False))
    frame_ids = None if args.frame_id is None else [args.frame_id]
    density_path, _ = pair_psi_files(args.psi_data, frame_ids=frame_ids)[0]
    grid = load_psi_grid(
        args.psi_data,
        log_T=np.array([4.0, 8.0], dtype=np.float32),
        density_unit_scale_cm3=MAS_DENSITY_UNIT_CM3,
        temperature_unit_scale_K=MAS_TEMPERATURE_UNIT_K,
        reference_frame_id=_frame_id(density_path),
        longitude_frame="carrington",
        frame_ids=[_frame_id(density_path)],
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )
    opacity = None
    if str(args.absorption_artifact).lower() != "none":
        from sunerf.rendering.plasma import init_absorption_model

        opacity = init_absorption_model(
            {"type": "photoionization", "artifact": args.absorption_artifact},
            instrument_key="AIA", channels=["A171"],
        )
    # The truth grid is static (one snapshot), so its query time is irrelevant.
    truth = query_model(
        grid.model, evaluation_points(radius, latitude, longitude), opacity=opacity
    )

    loader = PlasmaSuNeRFLoader(args.reconstruction)
    model = loader.model.eval()
    model_time = 0.0 if args.time is None else float(
        loader.normalize_datetime(datetime.fromisoformat(args.time))
    )
    points = evaluation_points(radius, latitude, longitude, time=model_time)
    reconstruction = query_model(model, points.to(loader.device), opacity=opacity)

    report = {
        "psi_frame": int(_frame_id(density_path)),
        "evaluation_time": args.time,
        "reconstruction": str(Path(args.reconstruction).resolve()),
        "emitting_log_T": float(np.log10(args.emitting_temperature_k)),
        "radial_bins": compare(
            truth[:2], reconstruction[:2], radius,
            emitting_log_T=float(np.log10(args.emitting_temperature_k)),
        ),
    }
    if opacity is not None:
        # Equivalent cool hydrogen (alpha_171 / kappa_171) integrated over shells:
        # the chromospheric limb layer below 1.02 R_sun and the cool material above.
        report["absorber"] = compare_absorber(truth[2], reconstruction[2], radius, latitude)
    with open(output / "psi_euv_truth_comparison.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    plot_profiles(output / "psi_euv_radial_profiles.png", radius, truth[:2], reconstruction[:2])
    if args.save_truth:
        np.savez_compressed(
            output / "psi_euv_evaluation_grid.npz",
            radius=radius, latitude=latitude, longitude=longitude,
            true_log_ne=truth[0], true_log_T=truth[1],
            log_ne=reconstruction[0], log_T=reconstruction[1],
        )
    print(json.dumps(report["radial_bins"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
