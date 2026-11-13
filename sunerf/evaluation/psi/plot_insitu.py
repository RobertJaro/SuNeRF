#!/usr/bin/env python3
"""Plot 8: in-situ time series of the synthetic probes against every run.

One column per probe (the training probe through the CME body and the held-out
probe on the flank); rows show the electron density and the radial velocity
along the trajectory with the probe radius on top.  The ground truth is the
clean product, the grey dots are the noisy samples that entered the training,
and every run is the reconstruction sampled along the same trajectory
(``sunerf.evaluation.cme.export_insitu``).  The panel text gives the median
absolute log10 density error of every run over the eruption phase.

Reads the ``PSP_synth*.npz`` / ``*_truth.npz`` products and the in-situ exports.
Writes ``psi_insitu``.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sunerf.evaluation.cme.export_layout import insitu_file, labelled_path
from sunerf.evaluation.psi.common import (
    FIGURE_WIDTH, RUN_COLORS, TRUTH_COLOR, TRUTH_TITLE, format_time_axis, save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--insitu-dir", type=Path, required=True,
                        help="Directory with the synthetic probes (<probe>.npz and <probe>_truth.npz).")
    parser.add_argument("--probes", nargs="+", default=["PSP_synth", "PSP_synth_heldout"],
                        help="Probe file stems; the first one is the training probe.")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every run.")
    parser.add_argument("--event", nargs=2, default=["2021-10-28T15:30:00", "2021-10-29T03:30:00"],
                        help="Eruption phase shown (ISO times).")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def to_datetime(time_unix):
    return np.array([datetime.fromtimestamp(float(t), timezone.utc).replace(tzinfo=None) for t in time_unix])


def load_probe(insitu_dir: Path, probe: str, runs):
    noisy = dict(np.load(insitu_dir / f"{probe}.npz"))
    truth = dict(np.load(insitu_dir / f"{probe}_truth.npz"))
    exports = [dict(np.load(insitu_file(export_dir, probe))) for _, export_dir in runs]
    for (label, _), export in zip(runs, exports):
        if not np.array_equal(export["time_unix"], truth["time_unix"]):
            raise ValueError(f"The in-situ export of '{label}' does not match the samples of {probe}.")
    return truth, noisy, exports


def log_density_error(truth, export, mask):
    ratio = np.log10(export["density_cm3"][mask] / truth["density_cm3"][mask])
    return float(np.nanmedian(np.abs(ratio)))


def main() -> None:
    args = parse_args()
    start, end = (datetime.fromisoformat(t) for t in args.event)
    n_probes = len(args.probes)
    fig, axs = plt.subplots(3, n_probes, figsize=(FIGURE_WIDTH, 5.4), sharex="col", squeeze=False,
                            gridspec_kw={"height_ratios": [1, 2, 2]}, constrained_layout=True)
    for column, probe in enumerate(args.probes):
        truth, noisy, exports = load_probe(args.insitu_dir, probe, args.runs)
        times = to_datetime(truth["time_unix"])
        event = (times >= start) & (times <= end)
        title = "training probe" if column == 0 else "held-out probe"

        axis = axs[0, column]
        axis.plot(times[event], truth["radius_rsun"][event], color=TRUTH_COLOR, lw=1)
        axis.set_title(f"{probe} ({title})", fontsize=8)
        axis.set_ylabel(r"$r$ [$R_\odot$]", fontsize=7)

        axis = axs[1, column]
        axis.plot(times[event], noisy["density_cm3"][event], ".", color="0.6", ms=2.5, label="samples")
        axis.plot(times[event], truth["density_cm3"][event], color=TRUTH_COLOR, lw=1.2, label=TRUTH_TITLE)
        for (label, _), export, color in zip(args.runs, exports, RUN_COLORS):
            error = log_density_error(truth, export, event)
            axis.plot(times[event], export["density_cm3"][event], color=color, lw=1,
                      label=f"{label} ({error:.2f} dex)")
        axis.set_yscale("log")
        axis.set_ylabel(r"$n_e$ [cm$^{-3}$]", fontsize=7)
        axis.legend(fontsize=5.5, loc="best")

        axis = axs[2, column]
        has_velocity = noisy["has_velocity"][event] > 0.5
        axis.plot(times[event][has_velocity], noisy["velocity_radial_kms"][event][has_velocity], ".", color="0.6", ms=2.5)
        if np.any(truth["has_velocity"] > 0.5):
            axis.plot(times[event], truth["velocity_radial_kms"][event], color=TRUTH_COLOR, lw=1.2)
        else:
            axis.text(0.02, 0.92, "no velocity target", transform=axis.transAxes, fontsize=6, va="top")
        for (_, _), export, color in zip(args.runs, exports, RUN_COLORS):
            axis.plot(times[event], export["velocity_radial_kms"][event], color=color, lw=1)
        axis.set_ylabel(r"$v_r$ [km s$^{-1}$]", fontsize=7)
        axis.set_xlabel(f"{start:%Y-%m-%d} [UT]", fontsize=7)
        format_time_axis(axis)
        for axis in axs[:, column]:
            axis.tick_params(labelsize=6.5)
            axis.grid(alpha=0.3, lw=0.4)
    print(save_figure(fig, args.output_dir, "psi_insitu"))


if __name__ == "__main__":
    main()
