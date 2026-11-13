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
CADENCE="all"

CCOR_RAW_DIR="${DATA_ROOT}/ccor_l2"

################################################################################
# Download CCOR level-2 science data
################################################################################
python -m sunerf.data.download.download_ccor \
  --start "${START}" \
  --end "${END}" \
  --cadence "${CADENCE}" \
  --output "${CCOR_RAW_DIR}" \
  --product-prefix "SWFO/GOES-19/CCOR-1/ccor1-l2_science"
