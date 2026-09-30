#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
CLEAN_DIR="${DATA_ROOT}/clean"
DEGRADED_DIR="${DATA_ROOT}/degraded_gain"
DENSITY_DIR="${DATA_ROOT}/rho"
RUNS_ROOT="/glade/work/rjarolim/sunerf-cme-obs/psi_obs"
EXPORT_ROOT="/glade/work/rjarolim/sunerf-cme-obs/export"
FIGURE_DIR="/glade/work/rjarolim/sunerf-cme-obs/evaluation/psi"
PSI_SERIES="${EXPORT_ROOT}/psi/cme_series.npz"
# Compared runs: column title and export directory.
RUNS=(
  "2 views, clean=${EXPORT_ROOT}/clean_2view_v01"
  "3 views, clean=${EXPORT_ROOT}/clean_3view_v01"
  "2 views, degraded=${EXPORT_ROOT}/degraded_gain_2view_v02"
  "3 views, degraded=${EXPORT_ROOT}/degraded_gain_3view_v02"
)
# Reference frame of every view: frame 105 renders PSI dump 25 (17:30 UT), when the
# CME is inside the L4, L5 and P1 fields of view (frame = dump + 80). L5_wide
# (15-30 R_sun) stays at frame 153 (dump 73, 21:30 UT), when the CME has reached its field.
IMAGE_FRAMES="L4=105 L5=105 P1=105"
SETUP_FRAMES="L5_wide=153 L5=105 L4=105 L1=105"
# Dump of the density comparison and dumps of the tracking maps.
DUMP=73
TRACKING_DUMPS="20 35 50"

################################################################################
# Conversion (GPU): SuNeRF density of all dumps on the PSI grid (npy, cm^-3) and
# SuNeRF tB/pB images of the reference frame of every view (npy, MSB)
################################################################################
for run in clean_2view_v01 clean_3view_v01 degraded_gain_2view_v02 degraded_gain_3view_v02; do
python -m sunerf.evaluation.cme.export_density \
  --sunerf_path "${RUNS_ROOT}/${run}/save_state.snf" \
  --data_path "${DENSITY_DIR}/rho*.hdf" \
  --dump_times "${DATA_ROOT}/dump_times.json" \
  --out_path "${EXPORT_ROOT}/${run}" --trusted
python -m sunerf.evaluation.cme.export_images \
  --sunerf_path "${RUNS_ROOT}/${run}/save_state.snf" \
  --data_path "${CLEAN_DIR}" --frames ${IMAGE_FRAMES} \
  --out_path "${EXPORT_ROOT}/${run}" --trusted
done

################################################################################
# Conversion (CPU): CME series of PSI and every run inside the CME cone
# (mass per radius; body edge, body peak and cavity radius; density along the CME
# direction; no background subtraction)
################################################################################
python -m sunerf.evaluation.cme.export_cme_series \
  --density_path "${DENSITY_DIR}/rho*.hdf" --dump_times "${DATA_ROOT}/dump_times.json" \
  --out_file "${PSI_SERIES}"
for run in clean_2view_v01 clean_3view_v01 degraded_gain_2view_v02 degraded_gain_3view_v02; do
python -m sunerf.evaluation.cme.export_cme_series \
  --density_path "${EXPORT_ROOT}/${run}" --out_file "${EXPORT_ROOT}/${run}/cme_series.npz"
done

################################################################################
# Plot 1: 2D histograms of the density of every run against PSI; the degraded
# runs (3, 4) are evaluated after normalization, their calibration is arbitrary
################################################################################
python -m sunerf.evaluation.psi.plot_histograms \
  --density-dir "${DENSITY_DIR}" --dump "${DUMP}" --runs "${RUNS[@]}" --normalized-runs 3 4 \
  --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 2: pB images of PSI, the degraded observation and every run; PSNR and SSIM
################################################################################
python -m sunerf.evaluation.psi.plot_images \
  --clean-dir "${CLEAN_DIR}" --degraded-dir "${DEGRADED_DIR}" --frames ${IMAGE_FRAMES} \
  --runs "${RUNS[@]}" --separators 2 4 --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 3: density at constant radius, in the meridional and the equatorial plane
################################################################################
python -m sunerf.evaluation.psi.plot_density_cuts \
  --density-dir "${DENSITY_DIR}" --dump "${DUMP}" --runs "${RUNS[@]}" --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 4: setup overview (observer configuration and degraded tB/pB observations)
################################################################################
python -m sunerf.evaluation.psi.plot_setup \
  --degraded-dir "${DEGRADED_DIR}" --frames ${SETUP_FRAMES} --tb-only L5 --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 5: tracking of the CME leading edge, body and cavity (maps; height, speed
# and acceleration over time; fit values in psi_cme_tracking.txt); the leading
# edge is measured against the pre-event dump 1
################################################################################
python -m sunerf.evaluation.psi.plot_cme_tracking \
  --density-dir "${DENSITY_DIR}" --psi-series "${PSI_SERIES}" --dumps ${TRACKING_DUMPS} --reference-dump 1 \
  --runs "${RUNS[@]}" --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 6: mass propagation inside the CME cone (mass-distance-time with the tracks
# of the run and of PSI; shell masses with their correlation to PSI)
################################################################################
python -m sunerf.evaluation.psi.plot_mass_propagation \
  --psi-series "${PSI_SERIES}" --runs "${RUNS[@]}" --output-dir "${FIGURE_DIR}"

################################################################################
# Plot 7: J-maps of the CME period: naive construction from the clean tB images
# of every view (plane-of-sky height) against the 3D density along the CME direction
################################################################################
python -m sunerf.evaluation.psi.plot_jmap \
  --image-dir "${CLEAN_DIR}" --views L4 L5 L5_wide --psi-series "${PSI_SERIES}" \
  --runs "${RUNS[@]}" --output-dir "${FIGURE_DIR}"
