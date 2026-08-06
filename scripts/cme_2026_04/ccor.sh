#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04"
START="2026-04-01T00:00:00"
END="2026-05-01T00:00:00"
CADENCE="all"
WORKERS="10"

CCOR_RAW_DIR="${DATA_ROOT}/ccor_l2"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Download CCOR level-2 science data
################################################################################
python -m sunerf.data.download.download_ccor \
  --start "${START}" \
  --end "${END}" \
  --cadence "${CADENCE}" \
  --out "${CCOR_RAW_DIR}" \
  --product-prefix "SWFO/GOES-19/CCOR-1/ccor1-l2_science"

################################################################################
# Prepare total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_ccor \
  --data_path "${CCOR_RAW_DIR}/*.fits" \
  --out_path "${PREP_DIR}/ccor" \
  --resize 512 512 \
  --filter_bright_background_objects \
  --num_workers "${WORKERS}"

################################################################################
# Compute correction mask
################################################################################
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/ccor/*" \
  --output "${PREP_DIR}/masks/ccor_tB_correction.npy"

################################################################################
# Quicklook movie
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/ccor" \
  --tb "${PREP_DIR}/ccor/*" \
  --tb_correction "${PREP_DIR}/masks/ccor_tB_correction.npy" \
  --num_workers "${WORKERS}"
