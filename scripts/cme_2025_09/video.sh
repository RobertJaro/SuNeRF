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
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2025_09_v34"
#RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2025_09_cme_v02"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
TIME_START="2025-09-21T00:00:00"
TIME_END="2025-09-24T00:00:00"

################################################################################
# Polar
################################################################################

python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_polar" \
  --lon_frame hci \
  --no_observer \
  --lat 89 --lon 0 --time 2025-09-10T00:00 --steps 1  --radius_min 3 --radius_max 50 \
  --lat 89 --lon 0 --time 2025-09-24T00:00 --steps 100  --radius_min 3 --radius_max 50

################################################################################
# Virtual Flight
################################################################################

python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_virtual" \
  --lon_frame hci \
  --no_observer \
  --lat 0 --lon 0 --time 2025-09-21T00:00 --steps 1  --radius_min 3 --radius_max 50 \
  --lat 0 --lon 0 --time 2025-09-22T11:00 --steps 10  --radius_min 3 --radius_max 50 \
  --lat 89 --lon 0 --time 2025-09-22T11:00 --steps 10  --radius_min 3 --radius_max 50 \
  --lat 89 --lon 0 --time 2025-09-24T00:00 --steps 10  --radius_min 3 --radius_max 50
