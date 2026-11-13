#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
START="2025-09-01T00:00:00"
END="2025-10-01T00:00:00"

COR_RAW_DIR="${DATA_ROOT}/cor"
COR_CLEAR_RAW_DIR="${COR_RAW_DIR}/clear"

################################################################################
# Download STEREO-A COR2 polarized triplets and total-brightness double exposures
################################################################################
python -m sunerf.data.download.download_cor \
  --start "${START}" \
  --end "${END}" \
  --output "${COR_RAW_DIR}" \
  --detector COR2

################################################################################
# Split into complete polarization triplets and clear images
################################################################################
python -m sunerf.data.cor.split_triplets \
  --glob "${COR_RAW_DIR}/*.fts" \
  --clear-dir "${COR_CLEAR_RAW_DIR}"
