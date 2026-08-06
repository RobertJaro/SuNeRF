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

PSP_RAW_DIR="${DATA_ROOT}/psp/raw"
PSP_PREP_DIR="${DATA_ROOT}/prep/psp"

################################################################################
# Download PSP in-situ data
################################################################################
python -m sunerf.data.download.download_psp_insitu \
  --start "${START}" \
  --end "${END}" \
  --out "${PSP_RAW_DIR}"

################################################################################
# Prepare PSP sparse trajectory constraints
################################################################################
python -m sunerf.data.prep.psp_insitu \
  --raw-dir "${PSP_RAW_DIR}" \
  --out "${PSP_PREP_DIR}/psp_insitu_20250901_20251001.npz" \
  --start "${START}" \
  --end "${END}" \
  --plot-dir "${PSP_PREP_DIR}/plots"
