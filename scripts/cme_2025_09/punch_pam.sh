#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/2025_09"
START="2025-09-01T00:00:00"
END="2025-10-01T00:00:00"
DOWNLOAD_CADENCE="all"
WORKERS="16"

PUNCH_PAM_RAW_DIR="${DATA_ROOT}/punch_pam"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Download PUNCH PAM data
################################################################################
python -m sunerf.data.download.download_punch \
  --level l3 \
  --product PAM \
  --ext 0l.fits \
  --start "${START}" \
  --end "${END}" \
  --cadence "${DOWNLOAD_CADENCE}" \
  --out "${PUNCH_PAM_RAW_DIR}"

################################################################################
# Prepare total- and polarized-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_punch_pam \
  --data_path "${PUNCH_PAM_RAW_DIR}/*.fits" \
  --out_path "${PREP_DIR}/punch_pam" \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --max_radius 90

################################################################################
# Quicklook movie
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/punch_pam" \
  --tb "${PREP_DIR}/punch_pam/tB/*" \
  --pb "${PREP_DIR}/punch_pam/pB/*" \
  --num_workers "${WORKERS}"
