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
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2025_09_22_cme_v04"
#RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2025_09_cme_v02"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
TIME_START="2025-09-21T00:00:00"
TIME_END="2025-09-24T00:00:00"

################################################################################
# Reference image series
################################################################################
python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_punch_pam" \
  --ref_pB_path "${DATA_ROOT}/prep/punch_pam/pB/*" \
  --ref_tB_path "${DATA_ROOT}/prep/punch_pam/tB/*" \
  --time_range "${TIME_START}" "${TIME_END}"

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_cor2" \
  --ref_pB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/cor2/pB/*" \
  --ref_tB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/cor2/tB/*" \
  --instrument_key STEREO_A_COR2 \
  --time_range "2025-09-21T00:00:00" "2025-09-24T00:00:00"

python -m sunerf.evaluation.cme.plot_ref_series \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/ref_series_ccor" \
  --ref_tB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/ccor/*" \
  --instrument_key CCOR \
  --time_range "2025-09-21T00:00:00" "2025-09-24T00:00:00"

################################################################################
# Inertial longitude-radius latitude-integrated density video
################################################################################
python -m sunerf.evaluation.cme.plot_integrated_density \
  --sunerf_path "${SUNERF_PATH}" \
  --longitude_range 0 360 \
  --time_range "2025-09-21T00:00:00" "2025-09-24T00:00:00" \
  --radius_range 3 60 \
  --latitude_range -50 0 \
  --t_points 30 \
  --n_radius 80 \
  --n_latitude 64 \
  --n_longitude 64 \
  --dpi 150 \
  --fps 10

################################################################################
# Tomography slices
################################################################################
python -m sunerf.evaluation.cme.plot_tomography \
  --sunerf_path "${SUNERF_PATH}" \
  --longitudes 240 250 260 270 280 290 300 310 \
  --time_range "2025-09-21T00:00:00" "2025-09-24T00:00:00"  \
  --t_points 100 \
  --radius_range 3 60 \
  --latitude_range -90 270



python -m sunerf.evaluation.cme.plot_tomography \
  --sunerf_path "${SUNERF_PATH}" \
  --longitudes 130 140 150 160 170 180 \
  --time_range "2025-09-01T12:00:00" "2025-09-06T12:00:00"  \
  --radius_range 3 60 \
  --latitude_range -90 90 \
  --log_radius

python -m sunerf.evaluation.cme.plot_tomography \
  --sunerf_path "${SUNERF_PATH}" \
  --longitudes 100 110 120 130 140 150 160 170 \
  --time_range "2025-09-06T00:00:00" "2025-09-08T00:00:00" \
  --radius_range 3 60 \
  --latitude_range -90 270

python -m sunerf.evaluation.cme.plot_tomography \
  --sunerf_path "${SUNERF_PATH}" \
  --longitudes 30 60 90 120 150 180 \
  --time_range "${TIME_START}" "${TIME_END}" \
  --radius_range 3 60 \
  --latitude_range -90 270



################################################################################
# Radius maps
################################################################################
python -m sunerf.evaluation.cme.plot_radius_map \
  --sunerf_path "${SUNERF_PATH}" \
  --radius 6 9 12 15 50 80 \
  --time_range "2025-09-06T00:00:00" "2025-09-08T00:00:00"


python -m sunerf.evaluation.cme.video_psp_north \
  --sunerf_path "${SUNERF_PATH}" \
  --insitu_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/psp/psp_insitu_20250901_20251001.npz" \
  --out_path "${RUN_ROOT}/video_psp_north" \
  --N 10 \
  --occ_min 3 \
  --occ_max 80 \
  --max_plot_radius_rsun 60 \
  --time_start "${TIME_START}" \
  --time_end "${TIME_END}"
