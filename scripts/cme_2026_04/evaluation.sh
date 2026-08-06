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
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2026_04_cme_v02"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
TIME_START="2026-04-23T14:00:00"
TIME_END="2026-04-24T00:00:00"

################################################################################
# Reference image series
################################################################################
python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_cor2_clear" \
  --ref_tB_path "${DATA_ROOT}/prep/cor2_clear/*" \
  --instrument_key STEREO_A_COR2_CLEAR \
  --time_range "${TIME_START}" "${TIME_END}"

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_cor2" \
  --ref_pB_path "${DATA_ROOT}/prep/cor2/pB/*" \
  --ref_tB_path "${DATA_ROOT}/prep/cor2/tB/*" \
  --instrument_key STEREO_A_COR2 \
  --time_range "${TIME_START}" "${TIME_END}"

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_ccor" \
  --ref_tB_path "${DATA_ROOT}/prep/ccor/*" \
  --instrument_key CCOR \
  --time_range "${TIME_START}" "${TIME_END}"

################################################################################
# CME video
################################################################################
python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video" \
  --lon_frame hci \
  --lat -4.660 --lon 140.137 --time 2026-04-23T00:00 --steps 1 --radius_min 3 --radius_max 20 \
  --lat -4.660 --lon 140.137 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660 --lon 200 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660 --lon 200 --time 2026-04-24T06:00 --steps 20 --radius_min 3 --radius_max 20

################################################################################
# Density line-of-sight profile
################################################################################
python -m sunerf.evaluation.cme.density_los_profile \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/density_los_1948" \
  --time 2026-04-23T19:48 \
  --point A 3000 5000 \
  --point B 2000 5000 \
  --observer earth \
  --occ_min 3 \
  --occ_max 20
