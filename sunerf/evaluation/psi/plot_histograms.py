#!/usr/bin/env python3
"""Plot 1: 2D histograms of the SuNeRF density of every run against the GT density.

Reads the GT cube of one dump and the density exports of the runs
(``sunerf.evaluation.cme.export_density``).  The histograms and the statistics
use every grid point of the evaluated shell with a positive density in GT and
in the run; nothing is subsampled.

The absolute calibration of a reconstruction from degraded observations is
arbitrary.  The runs listed in ``--normalized-runs`` therefore also show the 1:1
line shifted by the normalization between the two distributions (median of the
log density ratio) and report the RMSE after this normalization; the other runs
show the 1:1 line and the plain RMSE.  Writes ``psi_histograms`` and all metrics
of every run to ``psi_histograms.txt``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, RADIUS_RANGE, add_colorbar, add_panel_metric, load_densities, save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True, help="GT density cubes rho*.hdf.")
    parser.add_argument("--dump", type=int, required=True, help="Simulation dump of the compared density.")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--normalized-runs", type=int, nargs="*", default=[],
                        help="Runs (counted from 1) evaluated after normalization, e.g. 3 4.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def density_metrics(truth, reconstruction) -> dict:
    """Statistics of the log10 densities of all points that are positive in both arrays."""
    valid = np.isfinite(truth) & np.isfinite(reconstruction) & (truth > 0) & (reconstruction > 0)
    log_truth = np.log10(truth[valid], dtype=np.float64)
    log_reconstruction = np.log10(reconstruction[valid], dtype=np.float64)
    slope, intercept = np.polyfit(log_truth, log_reconstruction, 1)
    offset = float(np.median(log_reconstruction - log_truth))
    residual = log_reconstruction - log_truth
    return {
        "n_points": int(valid.sum()),
        "n_excluded": int(valid.size - valid.sum()),
        "correlation_log10": float(np.corrcoef(log_truth, log_reconstruction)[0, 1]),
        "correlation_linear": float(np.corrcoef(truth[valid].astype(np.float64),
                                                reconstruction[valid].astype(np.float64))[0, 1]),
        "fit_slope_log10": float(slope),
        "fit_intercept_log10": float(intercept),
        "normalization_log10": offset,
        "normalization_factor": float(10 ** offset),
        "rmse_log10": float(np.sqrt(np.mean(residual ** 2))),
        "rmse_log10_normalized": float(np.sqrt(np.mean((residual - offset) ** 2))),
        "log_truth": log_truth,
        "log_reconstruction": log_reconstruction,
    }


def write_metrics(path: Path, dump, labels, metrics) -> None:
    """Metrics of every run for the unnormalized and for the normalized reconstruction."""
    def row(name, values):
        cells = [f"{value:>18d}" if isinstance(value, int) else f"{value:>18.4f}" for value in values]
        return f"{name:<22}" + "".join(f"  {cell}" for cell in cells)

    def column(key):
        return [entry[key] for entry in metrics]

    header = " " * 22 + "".join(f"  {label:>18}" for label in labels)
    lines = [
        f"SuNeRF density against GT, dump {dump}, {RADIUS_RANGE[0]:g}-{RADIUS_RANGE[1]:g} R_sun",
        "All grid points with a positive density in GT and in the run; equal weight per grid point.",
        "log10 of the electron density in cm^-3; fit: log10(SuNeRF) = slope * log10(GT) + intercept.",
        "Normalized: SuNeRF divided by normalization_factor, the median of SuNeRF / GT.",
        "",
        header,
        row("n_points", column("n_points")),
        row("n_excluded", column("n_excluded")),
        "",
        "Unnormalized",
        row("correlation_log10", column("correlation_log10")),
        row("correlation_linear", column("correlation_linear")),
        row("fit_slope_log10", column("fit_slope_log10")),
        row("fit_intercept_log10", column("fit_intercept_log10")),
        row("rmse_log10", column("rmse_log10")),
        "",
        "Normalized",
        row("normalization_factor", column("normalization_factor")),
        row("normalization_log10", column("normalization_log10")),
        row("correlation_log10", column("correlation_log10")),
        row("correlation_linear", column("correlation_linear")),
        row("fit_slope_log10", column("fit_slope_log10")),
        row("fit_intercept_log10", [entry["fit_intercept_log10"] - entry["normalization_log10"]
                                    for entry in metrics]),
        row("rmse_log10", column("rmse_log10_normalized")),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    truth, cubes, *_ = load_densities(args.density_dir, args.dump, args.runs)
    labels = [label for label, _ in args.runs]
    metrics = [density_metrics(truth, cube) for cube in cubes]
    limits = np.percentile(np.log10(truth[np.isfinite(truth) & (truth > 0)]), (0.2, 99.8))
    bins = np.linspace(*limits, 151)

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH / (len(cubes) + 0.3) + 0.5), constrained_layout=True)
    grid = GridSpec(1, len(cubes) + 1, figure=fig, width_ratios=(*[1] * len(cubes), 0.05))
    for column, (entry, label) in enumerate(zip(metrics, labels)):
        axis = fig.add_subplot(grid[0, column])
        histogram = axis.hist2d(entry["log_truth"], entry["log_reconstruction"], bins=(bins, bins), norm=LogNorm(),
                                cmap="magma", rasterized=True)
        axis.plot(limits, limits, color="white", linewidth=0.8, linestyle="--", label="1:1")
        normalized = column + 1 in args.normalized_runs
        if normalized:
            axis.plot(limits, limits + entry["normalization_log10"], color="cyan", linewidth=0.8, linestyle=":",
                      label="1:1 normalized")
            axis.legend(fontsize=5.5, loc="lower right", labelcolor="white", frameon=False)
        # the correlation does not depend on the normalization
        rmse = entry["rmse_log10_normalized"] if normalized else entry["rmse_log10"]
        superscript = "^{\\mathrm{norm}}" if normalized else ""
        add_panel_metric(axis, f"r$_{{\\log}}$ = {entry['correlation_log10']:.3f}\n"
                               f"RMSE$_{{\\log}}{superscript}$ = {rmse:.3f}")
        axis.set_title(label, fontsize=8)
        axis.set_xlim(*limits)
        axis.set_ylim(*limits)
        axis.set_aspect("equal")
        axis.set_facecolor("black")
        axis.tick_params(labelsize=6.5)
        axis.set_xlabel(r"log$_{10}$ $n_e$ GT [cm$^{-3}$]", fontsize=7)
        if column == 0:
            axis.set_ylabel(r"log$_{10}$ $n_e$ SuNeRF [cm$^{-3}$]", fontsize=7)
        else:
            axis.set_yticklabels([])
    add_colorbar(fig, histogram[3], grid[0, -1], "grid points per bin")
    print(save_figure(fig, args.output_dir, "psi_histograms"))
    write_metrics(args.output_dir / "psi_histograms.txt", args.dump, labels, metrics)
    print(args.output_dir / "psi_histograms.txt")


if __name__ == "__main__":
    main()
