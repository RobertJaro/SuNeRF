#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2010_03"
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2010_03_cme_v02"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"

################################################################################
# Reference image series
################################################################################
python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_stereo_a" \
  --ref_pB_path "${DATA_ROOT}/prep/stereo_a_cor2/pB/*" \
  --ref_tB_path "${DATA_ROOT}/prep/stereo_a_cor2/tB/*" \
  --instrument_key STEREO_A_COR2


python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_stereo_b" \
  --ref_pB_path "${DATA_ROOT}/prep/stereo_b_cor2/pB/*" \
  --ref_tB_path "${DATA_ROOT}/prep/stereo_b_cor2/tB/*" \
  --instrument_key STEREO_B_COR2

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_lasco_c2" \
  --ref_pB_path "${DATA_ROOT}/prep/lasco_c2/pB/*" \
  --ref_tB_path "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --instrument_key LASCO_C2

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_lasco_c2_clear" \
  --ref_tB_path "${DATA_ROOT}/prep/lasco_c2_clear/tB/*" \
  --instrument_key LASCO_C2_CLEAR

################################################################################
# Tomography slices
################################################################################
python -m sunerf.evaluation.cme.plot_tomography \
  --sunerf_path "${SUNERF_PATH}" \
  --longitudes 80 90 100 110 120 130 \
  --time_range "2010-03-19T00:00" "2010-03-21T00:00"

################################################################################
# Radius maps
################################################################################
python -m sunerf.evaluation.cme.plot_radius_map \
  --sunerf_path "${SUNERF_PATH}" \
  --radius 5 8 12 15
