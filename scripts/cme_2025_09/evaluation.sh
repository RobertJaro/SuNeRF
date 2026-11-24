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
RUN_ROOT="/glade/work/rjarolim/sunerf-cme-obs/2025_09_cme_v25"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
TIME_START="2025-09-21T00:00:00"
TIME_END="2025-09-24T00:00:00"

################################################################################
# Overview of the prepared input data in the CME period (cme_time of 2025_09_cme.yaml):
# 5 frames per observing sequence, one row per sequence, colorbars on the right
################################################################################
python -m sunerf.evaluation.cme.plot_input_data \
  --out_file "${RUN_ROOT}/evaluation/input_data_overview.png" \
  --time_range "2025-09-21T12:00:00" "2025-09-23T12:00:00" \
  --punch_pB "${DATA_ROOT}/prep_stash/punch_pam/pB/*" \
  --cor2_clear_tB "${DATA_ROOT}/prep/cor2_clear/*" \
  --cor2_clear_correction "${DATA_ROOT}/prep/masks/stereo_a_cor2_tB_clear_correction.npy" \
  --ccor_tB "${DATA_ROOT}/prep/ccor/*" \
  --lasco_c2_tB "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --lasco_c2_correction "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy" \
  --lasco_c3_tB "${DATA_ROOT}/prep/lasco_c3/tB/*" \
  --lasco_c3_correction "${DATA_ROOT}/prep/masks/lasco_c3_tB_correction.npy" \
  --num_workers 8

################################################################################
# Alignment check: PUNCH and LASCO C2 tB closest to --time reprojected into the LASCO C3 frame,
# stacked PUNCH, C3, C2 (bottom to top)
################################################################################
python -m sunerf.evaluation.cme.plot_overlay_alignment \
  --out_file "${RUN_ROOT}/evaluation/overlay_alignment.png" \
  --time "2025-09-22T08:00:00" \
  --punch "${DATA_ROOT}/prep_stash/punch_pam/tB/*" \
  --lasco_c2 "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --lasco_c3 "${DATA_ROOT}/prep/lasco_c3/tB/*" \
  --lasco_c2_correction "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy" \
  --lasco_c3_correction "${DATA_ROOT}/prep/masks/lasco_c3_tB_correction.npy" \
  --num_workers 8

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
# Clean renders of the combined field of view (3-100 R_sun) from Earth and STEREO-A
# (ephemeris positions at every frame, hourly over the primary CME period)
################################################################################
python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_earth" \
  --observer earth --panels tB pB density distance --resolution 1024 \
  --time 2025-09-22T00:00 --steps 1  --radius_min 3 --radius_max 100 \
  --time 2025-09-23T06:00 --steps 30 --radius_min 3 --radius_max 100
python -m sunerf.evaluation.image_sequence_to_video "${RUN_ROOT}/video_earth" "${RUN_ROOT}/video_earth.mp4" --fps 10

python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_stereo_a" \
  --observer stereo_a --panels tB pB density distance --resolution 1024 \
  --time 2025-09-22T00:00 --steps 1  --radius_min 3 --radius_max 100 \
  --time 2025-09-23T06:00 --steps 30 --radius_min 3 --radius_max 100
python -m sunerf.evaluation.image_sequence_to_video "${RUN_ROOT}/video_stereo_a" "${RUN_ROOT}/video_stereo_a.mp4" --fps 10

python -m sunerf.evaluation.cme.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_360" \
  --panels observer pB density distance --resolution 256 \
  --time 2025-09-02T00:00 --steps 1  --radius_min 3 --radius_max 100 --lat 0 --lon 60 \
  --time 2025-09-04T12:00 --steps 50 --radius_min 3 --radius_max 100 --lat 0 --lon 60 \
  --time 2025-09-04T12:00 --steps 50 --radius_min 3 --radius_max 100 --lat 0 --lon 420 \
  --time 2025-09-04T12:00 --steps 50 --radius_min 3 --radius_max 100 --lat 50 --lon 420 \
  --time 2025-09-06T00:00 --steps 50 --radius_min 3 --radius_max 100 --lat 50 --lon 420
python -m sunerf.evaluation.image_sequence_to_video "${RUN_ROOT}/video_360" "${RUN_ROOT}/video_360.mp4" --fps 10


################################################################################
# Density export on a spherical grid (basis of the CME evaluation) and the CME figures;
# VTK cubes for ParaView are rendered independently. CME direction: latitude -45, HCI longitude 10.
################################################################################
python -m sunerf.evaluation.cme.export_density_grid \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/export" \
  --time_range "2025-09-21T18:00:00" "2025-09-23T11:00:00" \
  --cadence_minutes 30 --radial_step 0.5 --angular_step 0.5

python -m sunerf.evaluation.cme.plot_cme_propagation \
  --export_path "${RUN_ROOT}/export" --out_path "${RUN_ROOT}/evaluation/cme_propagation.jpg" \
  --latitude -30 --longitude 5

python -m sunerf.evaluation.cme.plot_cme_mass_budget \
  --export_path "${RUN_ROOT}/export" --out_path "${RUN_ROOT}/evaluation/cme_mass_budget.jpg" \
  --latitude -30 --longitude 5 --fit_start_hours 7 --seed_min_radius 8 --median_profile --above_background 0.2 --continue_to_hours 40 --end_hours 40 --fit_max_radius 70

python -m sunerf.evaluation.cme.plot_cme_centroid_front \
  --export_path "${RUN_ROOT}/export" --out_path "${RUN_ROOT}/evaluation/cme_centroid_front.png" \
  --latitude -25 --longitude 5

python -m sunerf.evaluation.cme.plot_cme_front \
  --export_path "${RUN_ROOT}/export" --out_path "${RUN_ROOT}/evaluation/cme_front.jpg" \
  --latitude -30 --longitude 15

python -m sunerf.evaluation.cme.plot_cme_cuts \
  --export_path "${RUN_ROOT}/export" --out_path "${RUN_ROOT}/evaluation/cme_cuts.jpg" \
  --centroid --latitude -25 --longitude 5 \
  --times 2025-09-21T18:00 2025-09-22T04:00 2025-09-22T10:00 2025-09-22T16:00 2025-09-23T00:00

python -m sunerf.evaluation.cme.export_vtk \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/vtk" \
  --time_range "2025-09-22T00:00:00" "2025-09-23T06:00:00" \
  --background_time "2025-09-21T12:00:00" \
  --latitude -25 --longitude 5 --half_angle 30 --observers earth stereo_a

################################################################################
# Qualitative comparison with STEREO-A HI-1 (SSC browse images, geometry from the L0 headers)
################################################################################
python -m sunerf.evaluation.cme.plot_hi1_comparison \
  --sunerf_path "${SUNERF_PATH}" \
  --fits "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/hi1/L0/*.fts" \
  --browse_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/hi1/browse" \
  --out_path "${RUN_ROOT}/evaluation/hi1" \
  --time_range "2025-09-22T00:00:00" "2025-09-23T06:00:00"

################################################################################
# Degree of polarization pB/tB: input images against the forward-rendered reconstruction
################################################################################
python -m sunerf.evaluation.cme.plot_polarization \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/polarization" \
  --punch_tB "${DATA_ROOT}/prep/punch_pam/tB/*" \
  --punch_pB "${DATA_ROOT}/prep/punch_pam/pB/*" \
  --ccor_tB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/ccor/*" \
  --cor2_tB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/cor2/tB/*" \
  --cor2_pB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/cor2/pB/*" \
  --cor2_tB_correction "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/masks/stereo_a_cor2_tB_correction.npy" \
  --cor2_pB_correction "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/masks/stereo_a_cor2_pB_correction.npy" \
  --lasco_c2_tB "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --lasco_c2_correction "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy" \
  --lasco_c3_tB "${DATA_ROOT}/prep/lasco_c3/tB/*" \
  --lasco_c3_correction "${DATA_ROOT}/prep/masks/lasco_c3_tB_correction.npy" \
  --time "2025-09-22T08:00:00" \
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
  --longitudes 250 255 260 265 270 \
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
  --time_range "2025-09-21T00:00:00" "2025-09-24T00:00:00"


python -m sunerf.evaluation.cme.video_psp_north \
  --sunerf_path "${SUNERF_PATH}" \
  --insitu_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/psp/psp_insitu_20250901_20251001.npz" \
  --solo_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/solo/solo_insitu_20250901_20251001.npz" \
  --out_path "${RUN_ROOT}/video_psp_north" \
  --N 100 \
  --occ_min 3 \
  --occ_max 100

################################################################################
# CME figures on the 1-degree grid (3-80 R_sun) and 201^3 VTK cubes
################################################################################
python -m sunerf.evaluation.cme.export_density_grid \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/export_1deg" \
  --time_range "2025-09-21T18:00:00" "2025-09-23T11:00:00" \
  --cadence_minutes 30 --radius_range 3 80 --radial_step 1 --angular_step 1

python -m sunerf.evaluation.cme.plot_cme_mass_budget \
  --export_path "${RUN_ROOT}/export_1deg" --out_path "${RUN_ROOT}/evaluation/1deg/cme_mass_budget.jpg" \
  --latitude -30 --longitude 5 --fit_start_hours 7 --seed_min_radius 8 --median_profile --above_background 0.2 --continue_to_hours 40 --end_hours 40 --fit_max_radius 70

python -m sunerf.evaluation.cme.plot_cme_centroid_front \
  --export_path "${RUN_ROOT}/export_1deg" --out_path "${RUN_ROOT}/evaluation/1deg/cme_centroid_front.png" \
  --latitude -25 --longitude 5

python -m sunerf.evaluation.cme.plot_cme_front \
  --export_path "${RUN_ROOT}/export_1deg" --out_path "${RUN_ROOT}/evaluation/1deg/cme_front.jpg" \
  --latitude -30 --longitude 15

python -m sunerf.evaluation.cme.plot_cme_cuts \
  --export_path "${RUN_ROOT}/export_1deg" --out_path "${RUN_ROOT}/evaluation/1deg/cme_cuts.jpg" \
  --centroid --latitude -25 --longitude 5 \
  --times 2025-09-21T18:00 2025-09-22T04:00 2025-09-22T10:00 2025-09-22T16:00 2025-09-23T00:00

python -m sunerf.evaluation.cme.export_vtk \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/vtk_201" \
  --time_range "2025-09-22T00:00:00" "2025-09-23T06:00:00" \
  --background_time "2025-09-21T12:00:00" \
  --pixels_per_solRad 1 \
  --latitude -25 --longitude 5 --half_angle 30 --observers earth stereo_a
