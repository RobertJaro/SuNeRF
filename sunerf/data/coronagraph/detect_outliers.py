#!/usr/bin/env python3
"""Flag invalid frames of a coronagraph sequence from per-frame statistics.

A post-processing step on the prepared frames. Every frame is reduced to a set
of scalar metrics, each of which forms a time series over the full sequence:

``mean``, ``median``, ``std``
    of the valid (finite) pixels, in log10 (a calibration factor is an offset);
``valid_fraction``
    finite pixels over all pixels (missing telemetry blocks, partial readouts);
    only a drop rejects;
``exposure``
    log10 of the exposure time (``EXPTIME``/``EXPOSURE``/``XPOSURE``);
``roll``, ``center_x``, ``center_y``
    roll angle (degrees) and Sun-center pixel position from the WCS (a broken
    pointing solution or crop).

A frame is an outlier when a metric deviates from the local behavior of its
series:

* baseline: the median of the metric over the other frames within
  ``window / 2`` hours of the frame;
* scale: the robust scatter (1.4826 MAD) of those frames around the baseline,
  never below the scatter of all residuals of the sequence nor below a fixed
  floor per metric (constant series such as the roll angle);
* ``z = (value - baseline) / scale``; ``|z| > threshold`` rejects the frame.

The flags are iterated: rejected frames are left out of the baselines and
scales of the next pass until the flags no longer change. The median baseline
ignores a run of consecutive invalid frames as long as the run fills less than
half of the window, and the iteration then removes its influence on the
neighboring frames.

Independent of the time series, a frame is rejected when it is unreadable, has
no valid pixels, or was taken in another observing mode than the majority of
the sequence (``MODE_KEYS``: filter, polarizer, binning, readout window).

The result is ``outliers.csv`` (``path``, ``status`` = reject/ok, metrics,
z-scores, reasons) and ``outliers.png`` (the series with their baselines) in
``--output``. ``python -m sunerf.data.frame_review apply <output>/outliers.csv``
moves the rejected files into a ``rejected/`` directory beside them, and
``python -m sunerf.data.frame_review restore <dir>`` moves them back.

Example::

    python -m sunerf.data.coronagraph.detect_outliers "prep/lasco_c2/tB/*.fits" \\
        --output prep/outliers/lasco_c2 --num_workers 16
"""

import argparse
import csv
import glob
import math
import multiprocessing
import os
import warnings
from collections import Counter

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from sunerf.data.coronagraph.prep_common import get_observation_time

# name: (log10 of the value, minimum scale, side of the deviation that rejects)
METRICS = {
    "mean": (True, 1.0e-4, "both"),
    "median": (True, 1.0e-4, "both"),
    "std": (True, 1.0e-4, "both"),
    "valid_fraction": (False, 2.0e-3, "low"),
    "exposure": (True, 1.0e-2, "both"),
    "roll": (False, 5.0e-2, "both"),
    "center_x": (False, 0.25, "both"),
    "center_y": (False, 0.25, "both"),
}
EXPOSURE_KEYS = ("EXPTIME", "EXPOSURE", "XPOSURE")
# Header keywords of the observing mode; missing keywords are ignored.
MODE_KEYS = ("FILTER", "POLAR", "LEBXSUM", "LEBYSUM", "SUMROW", "SUMCOL",
             "R1COL", "R2COL", "R1ROW", "R2ROW")
# Frames in the window (besides the frame itself) needed for a baseline; with
# fewer, the nearest valid frames in time are used instead.
MIN_NEIGHBOURS = 5
MAX_ITERATIONS = 20
MAD_TO_SIGMA = 1.4826


def _pointing(header):
    """Roll angle (degrees) and Sun-center pixel position from the WCS."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(header, naxis=2)
        pc = wcs.wcs.get_pc()
        roll = math.degrees(math.atan2(pc[1, 0], pc[0, 0]))
        center_x, center_y = wcs.wcs_world2pix(np.array([[0.0, 0.0]]), 0)[0]
    return roll, float(center_x), float(center_y)


def frame_statistics(path):
    record = {"path": os.path.abspath(path), "time": None, "error": ""}
    try:
        with fits.open(path, memmap=False) as hdul:
            hdu = next(hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim == 2)
            values = np.asarray(hdu.data, dtype=np.float64)
            header = hdu.header.copy()
        record["time"] = get_observation_time(path)
        measure_image(record, values, header)
    except Exception as exc:  # a defective file is a finding, not a failure
        record["error"] = f"unreadable: {type(exc).__name__}: {exc}"
    return record


def measure_image(record, values, header):
    """Measure one prepared coronagraph image on its native pixel grid."""
    record["mode"] = "|".join(f"{key}={header[key]}" for key in MODE_KEYS if key in header)
    exposure = next((header[key] for key in EXPOSURE_KEYS if key in header), None)
    exposure = float(exposure) if exposure is not None else math.nan
    record["exposure"] = exposure if np.isfinite(exposure) and exposure > 0 else math.nan
    try:
        record["roll"], record["center_x"], record["center_y"] = _pointing(header)
    except Exception:
        pass
    finite = np.isfinite(values)
    record["valid_fraction"] = float(finite.mean())
    values = values[finite]
    if values.size == 0:
        record["error"] = "no valid pixels"
        return
    record.update(mean=float(values.mean()), median=float(np.median(values)), std=float(values.std()))


def metric_series(records, key):
    log, _, _ = METRICS[key]
    values = np.array([r.get(key, math.nan) for r in records], dtype=np.float64)
    if log:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.where(values > 0, np.log10(values), np.nan)
    if key == "roll":
        finite = np.isfinite(values)
        values[finite] = np.unwrap(values[finite], period=360.0)
    return values


def local_scores(times, values, rejected, window, floor):
    """Baseline, scale and z-score of every frame from its non-rejected neighbors."""
    count = len(values)
    usable = np.isfinite(values) & ~rejected
    baseline = np.full(count, np.nan)
    local_scale = np.full(count, np.nan)
    half = 0.5 * window
    lower = np.searchsorted(times, times - half, side="left")
    upper = np.searchsorted(times, times + half, side="right")
    usable_index = np.flatnonzero(usable)
    for index in range(count):
        neighbours = np.arange(lower[index], upper[index])
        neighbours = neighbours[usable[neighbours] & (neighbours != index)]
        if neighbours.size < MIN_NEIGHBOURS:
            candidates = usable_index[usable_index != index]
            if candidates.size < MIN_NEIGHBOURS:
                continue
            nearest = np.argsort(np.abs(times[candidates] - times[index]), kind="stable")
            neighbours = candidates[nearest[:MIN_NEIGHBOURS]]
        reference = values[neighbours]
        baseline[index] = np.median(reference)
        local_scale[index] = MAD_TO_SIGMA * np.median(np.abs(reference - baseline[index]))
    residual = values - baseline
    global_residual = residual[usable & np.isfinite(residual)]
    global_scale = 0.0
    if global_residual.size:
        global_scale = MAD_TO_SIGMA * np.median(np.abs(global_residual - np.median(global_residual)))
    scale = np.fmax(np.fmax(local_scale, global_scale), floor)
    return baseline, scale, residual / scale


def majority_mode(records):
    """Reasons for frames whose observing mode differs from the most common one."""
    counts = Counter(r.get("mode") for r in records if not r["error"])
    if len(counts) > 1:
        print("Observing modes (frames):")
        for mode, count in counts.most_common():
            print(f"  {count:6d}  {mode or '(no mode keywords)'}")
    majority = counts.most_common(1)[0][0] if counts else None
    return [f"mode {r.get('mode')} (majority {majority})"
            if not r["error"] and r.get("mode") != majority else "" for r in records]


def detect_outliers(records, window, threshold):
    """Set ``status``, ``reasons``, baselines and z-scores of time-sorted records."""
    start = records[0]["time"]
    times = np.array([(r["time"] - start).total_seconds() / 3600.0 for r in records], dtype=np.float64)
    mode_reasons = majority_mode(records)
    fixed = np.array([bool(r["error"]) or bool(reason) for r, reason in zip(records, mode_reasons)])
    series = {key: metric_series(records, key) for key in METRICS}

    def exceeds(key, z):
        side = METRICS[key][2]
        with np.errstate(invalid="ignore"):
            return np.isfinite(z) & ((z < -threshold) if side == "low" else (np.abs(z) > threshold))

    rejected = fixed.copy()
    for iteration in range(MAX_ITERATIONS):
        scores = {key: local_scores(times, series[key], rejected, window, METRICS[key][1])
                  for key in METRICS}
        flagged = fixed.copy()
        for key in METRICS:
            flagged |= exceeds(key, scores[key][2])
        if np.array_equal(flagged, rejected):
            break
        rejected = flagged
    print(f"Converged after {iteration + 1} iterations.")

    exceeded = {key: exceeds(key, scores[key][2]) for key in METRICS}
    for index, record in enumerate(records):
        reasons = [reason for reason in (record["error"], mode_reasons[index]) if reason]
        for key in METRICS:
            baseline, scale, z = (scores[key][part][index] for part in range(3))
            record[f"series_{key}"] = series[key][index]
            record[f"baseline_{key}"] = baseline
            record[f"scale_{key}"] = scale
            record[f"z_{key}"] = z
            if exceeded[key][index]:
                reasons.append(f"{key} z={z:+.1f}")
        record["status"] = "reject" if rejected[index] else "ok"
        record["reasons"] = "; ".join(reasons)
    return records


CSV_COLUMNS = (
    ["path", "status", "time", "reasons", "mode"]
    + list(METRICS)
    + [f"z_{key}" for key in METRICS]
    + [f"{part}_{key}" for key in METRICS for part in ("series", "baseline", "scale")]
)


def write_csv(records, path):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["time"] = record["time"].isoformat() if record["time"] is not None else ""
            writer.writerow(row)


def plot_series(records, path, threshold, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [key for key in METRICS
            if np.isfinite([r[f"series_{key}"] for r in records]).any()]
    times = np.array([np.datetime64(r["time"]) for r in records])
    rejected = np.array([r["status"] == "reject" for r in records])
    fig, axes = plt.subplots(len(keys), 1, figsize=(14, 2.4 * len(keys)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, key in zip(axes, keys):
        value, baseline, scale = (np.array([r[f"{part}_{key}"] for r in records], dtype=np.float64)
                                  for part in ("series", "baseline", "scale"))
        ax.fill_between(times, baseline - threshold * scale, baseline + threshold * scale,
                        color="#9ca3af", alpha=0.3, linewidth=0, label=f"baseline ± {threshold:g} σ")
        ax.plot(times, baseline, color="#4b5563", linewidth=1.0, label="baseline")
        ax.plot(times[~rejected], value[~rejected], ".", color="#2563eb", markersize=3, label="kept")
        ax.plot(times[rejected], value[rejected], "x", color="#dc2626", markersize=6, label="rejected")
        ax.set_ylabel(f"log10 {key}" if METRICS[key][0] else key)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8, frameon=False)
    axes[0].set_title(f"{title}: {rejected.sum()} of {len(records)} frames rejected")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Glob pattern of the prepared FITS frames (quote it).")
    parser.add_argument("--output", required=True, help="Directory for outliers.csv and outliers.png.")
    parser.add_argument("--window", type=float, default=6.0,
                        help="Width of the local window in hours.")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Robust z-score above which a frame is rejected.")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    files = sorted(path for path in glob.glob(args.input) if os.path.isfile(path))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.input}")
    os.makedirs(args.output, exist_ok=True)

    with multiprocessing.Pool(args.num_workers) as pool:
        records = list(tqdm(pool.imap(frame_statistics, files, chunksize=8),
                            total=len(files), desc="Frame statistics"))

    undated = [r for r in records if r["time"] is None]
    for record in undated:
        record.update(status="reject", reasons=record["error"])
    records = sorted((r for r in records if r["time"] is not None), key=lambda r: r["time"])
    if records:
        records = detect_outliers(records, args.window, args.threshold)

    csv_path = os.path.join(args.output, "outliers.csv")
    write_csv(records + undated, csv_path)
    if records:
        plot_series(records, os.path.join(args.output, "outliers.png"), args.threshold,
                    os.path.basename(os.path.normpath(args.output)))

    n_rejected = sum(r["status"] == "reject" for r in records + undated)
    print(f"Rejected {n_rejected} of {len(files)} frames -> {csv_path}")
    for record in records + undated:
        if record["status"] == "reject":
            print(f"  {os.path.basename(record['path'])}: {record['reasons']}")


if __name__ == "__main__":
    main()
