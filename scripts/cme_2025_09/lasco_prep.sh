#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=06:00:00

set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
REPO_DIR="/glade/u/home/rjarolim/projects/SuNeRF"
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
WORKERS="16"
SSW_ROOT="/glade/work/rjarolim/ssw"

LASCO_RAW_DIR="${DATA_ROOT}/lasco"
LASCO_L1_DIR="${DATA_ROOT}/lasco_l1"
ANCIL_DIR="${DATA_ROOT}/soho_ancillary"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Calibrate level 0.5 to level 1 with SolarSoft REDUCE_LEVEL_1
################################################################################
csh "${REPO_DIR}/scripts/lasco_reduce_level_1.csh" \
  "${LASCO_RAW_DIR}/c2/level_05/*.fts" "${LASCO_L1_DIR}/c2" "${SSW_ROOT}" "${ANCIL_DIR}"

csh "${REPO_DIR}/scripts/lasco_reduce_level_1.csh" \
  "${LASCO_RAW_DIR}/c3/level_05/*.fts" "${LASCO_L1_DIR}/c3" "${SSW_ROOT}" "${ANCIL_DIR}"

################################################################################
# Prepare total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${LASCO_L1_DIR}/c2/*.fts" \
  --out_path "${PREP_DIR}/lasco_c2" \
  --occ_min 2100 \
  --occ_max 8000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite

python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${LASCO_L1_DIR}/c3/*.fts" \
  --out_path "${PREP_DIR}/lasco_c3" \
  --occ_min 4000 \
  --occ_max 29000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite

################################################################################
# Reject outlier frames (moved to tB/rejected/; undo: python -m sunerf.data.frame_review restore <tB dir>)
################################################################################
python -m sunerf.data.coronagraph.detect_outliers \
  "${PREP_DIR}/lasco_c2/tB/*.fits" \
  --output "${PREP_DIR}/outliers/lasco_c2" \
  --num_workers "${WORKERS}"
python -m sunerf.data.frame_review apply "${PREP_DIR}/outliers/lasco_c2/outliers.csv"

python -m sunerf.data.coronagraph.detect_outliers \
  "${PREP_DIR}/lasco_c3/tB/*.fits" \
  --output "${PREP_DIR}/outliers/lasco_c3" \
  --num_workers "${WORKERS}"
python -m sunerf.data.frame_review apply "${PREP_DIR}/outliers/lasco_c3/outliers.csv"

################################################################################
# Compute correction masks
################################################################################
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/lasco_c2/tB/*" \
  --output "${PREP_DIR}/masks/lasco_c2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/lasco_c3/tB/*" \
  --output "${PREP_DIR}/masks/lasco_c3_tB_correction.npy"

################################################################################
# Quicklook movies
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/lasco_c2" \
  --tb "${PREP_DIR}/lasco_c2/tB/*" \
  --tb_correction "${PREP_DIR}/macmesks/lasco_c2_tB_correction.npy" \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/lasco_c3" \
  --tb "${PREP_DIR}/lasco_c3/tB/*" \
  --tb_correction "${PREP_DIR}/masks/lasco_c3_tB_correction.npy" \
  --num_workers "${WORKERS}"
