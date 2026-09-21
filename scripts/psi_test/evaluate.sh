#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_test"
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/psi_test/degraded_3view_v1"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
CHECKPOINT_PATH="${RUN_ROOT}/final.ckpt"
CLEAN_DIR="${DATA_ROOT}/clean"
DEGRADED_DIR="${DATA_ROOT}/degraded"
TRUTH_DIR="${DATA_ROOT}/degradation_truth"
DENSITY_PATH="${DATA_ROOT}/rho/rho000050.hdf"
REFERENCE_TB_PATH="${CLEAN_DIR}/L1/tb/L1_tb050.fts"
OUTPUT_DIR="${RUN_ROOT}/evaluation"

################################################################################
# Require the completed degraded reconstruction
################################################################################
test -f "${CHECKPOINT_PATH}"

################################################################################
# Verify PSI HDF4 support
################################################################################
python -c "from psi_io import read_hdf_data; import pyhdf"

################################################################################
# Compare clean, degraded, and recovered tB/pB images for all viewpoints
################################################################################
python -m sunerf.evaluation.psi_test_images \
  --sunerf-path "${SUNERF_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --clean-dir "${CLEAN_DIR}" \
  --degraded-dir "${DEGRADED_DIR}" \
  --truth-dir "${TRUTH_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --views L1 L4 L5 P1 \
  --degraded-views L1 L4 L5

################################################################################
# Compare reconstructed and PSI densities with correlations and radial slices
################################################################################
python -m sunerf.evaluation.psi_test_density \
  --sunerf-path "${SUNERF_PATH}" \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --density-path "${DENSITY_PATH}" \
  --reference-tb-path "${REFERENCE_TB_PATH}" \
  --output-dir "${OUTPUT_DIR}"
