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

SOLO_RAW_DIR="${DATA_ROOT}/solo/raw"
SOLO_PREP_DIR="${DATA_ROOT}/prep/solo"

################################################################################
# Download Solar Orbiter COHO in-situ data
################################################################################
python -m sunerf.data.download.download_solo_insitu \
  --start "${START}" \
  --end "${END}" \
  --out "${SOLO_RAW_DIR}"

################################################################################
# Prepare Solar Orbiter in-situ constraints
################################################################################
python -m sunerf.data.prep.solo_insitu \
  --raw-dir "${SOLO_RAW_DIR}" \
  --out "${SOLO_PREP_DIR}/solo_insitu_20250901_20251001.npz" \
  --start "${START}" \
  --end "${END}" \
  --electron-density-factor 1.0 \
  --plot-dir "${SOLO_PREP_DIR}/plots"
