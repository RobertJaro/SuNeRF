#!/usr/bin/env bash
set -euo pipefail

cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# STEREO-A HI-1 for the qualitative comparison (plot_hi1_comparison): Level-0 FITS (geometry)
# and the SSC browse JPEGs (standard quicklook processing), every 4 hours
################################################################################
python -m sunerf.data.download.download_hi1 \
  --start "2025-09-21T00:00:00" \
  --end "2025-09-24T00:00:00" \
  --cadence 4h \
  --output "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/hi1"
