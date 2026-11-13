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
DOWNLOAD_CADENCE="all"

PUNCH_PAM_RAW_DIR="${DATA_ROOT}/punch_pam"

################################################################################
# Download PUNCH PAM data
################################################################################
python -m sunerf.data.download.download_punch \
  --level l3 \
  --product PAM \
  --ext 0m.fits \
  --start "${START}" \
  --end "${END}" \
  --cadence "${DOWNLOAD_CADENCE}" \
  --output "${PUNCH_PAM_RAW_DIR}"
