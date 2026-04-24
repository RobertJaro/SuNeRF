#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Train - for script use
python -m sunerf.run_thomson --config "config/cme/2025_09.yaml"
#python -m sunerf.run_thomson --config "config/cme/2025_09_cor2_ccor.yaml"
exit

#################################################
# Prepare data for CME reconstruction

# STEREO/COR2
python -m sunerf.data.coronagraph.prep_stereo_cor \
  --tb_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/tB/*.fts" \
  --pb_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/pB/*.fts" \
  --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2" \
  --occ_min 4000 --occ_max 15000 --resize 512 512 --cadence 1h

# CCOR
python -m sunerf.data.coronagraph.prep_ccor --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/ccor_l2/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor" --resize 512 512 --value_min 1.0e-16

# PUNCH
python -m sunerf.data.coronagraph.prep_punch_triplets \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/punch_nfi/*.fits" \
  --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_nfi" \
  --resize 512 512 --reproject

# PUNCH - PAM
python -m sunerf.data.coronagraph.prep_punch_pam \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/punch_pam/*.fits" \
  --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam" \
  --resize 512 512 \
  --num_workers 16 \
  --max_radius 90 \
  --value_max 1.0e-11 --value_min 1.0e-16

#################################################
# check data
# COR2
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" --vmin 1.0e-12
# CCOR
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/ccor" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*" --vmin 1.0e-12
# PUNCH
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/punch" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch/tB/*"
# PUNCH - WFI1
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/punch_wfi1" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/wfi1/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/wfi1/tB/*"
# PUNCH - NFI
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/punch_nfi" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_nfi/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_nfi/tB/*"
# PUNCH - CAM
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/punch_cam" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_cam/*"
# PUNCH - PAM
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/punch_pam" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/tB/*" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/pB/*"


#################################################
# clean invalid files

python -m sunerf.data.coronagraph.prune_by_reference_range \
  --reference-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/tB/*" \
  --target-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" \
  --target-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" \
  --target-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*"

python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/**/*" --dry_run

#################################################
# compute correction masks
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/stereo_a_cor2_tB_correction.npy"
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/stereo_a_cor2_pB_correction.npy"
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/ccor_tB_correction.npy"
#python -m sunerf.data.coronagraph.compute_correction --type daily-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/tB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/punch_pam_tB_correction.npy"
#python -m sunerf.data.coronagraph.compute_correction --type daily-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/pB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/punch_pam_pB_correction.npy"


#################################################
# Train
python -m sunerf.run_thomson --config "config/cme/2025_09.yaml"
python -m sunerf.run_thomson --config "config/cme/2025_09_cor.yaml"



#################################################
# Download validation data
python -m sunerf.data.download.download_aia \
  --download_dir "/glade/work/rjarolim/data/sunerf-cme/aia_validation/2025_09" \
  --email robert.jarolim@uni-graz.at \
  --t_start 2025-09-01T00:00:00 \
  --t_end 2025-09-19T00:00:00 \
  --cadence 1d \
  --channel 193

#################################################
# Evaluation

# compare to Carrington maps
python -m sunerf.evaluation.cme.carrington_map_comparison \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" \
  --aia_glob "/glade/work/rjarolim/data/sunerf-cme/aia_validation/2025_09/*.fits"

python -m sunerf.evaluation.cme.video_polar --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v07/save_state.snf" --occ_range 3 100
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --occ_range 3 15
python -m sunerf.evaluation.cme.video_fixed_rotation --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --occ_range 3 100
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --longitudes 90 100 110 120 130 140 150 160 --time_range "2025-09-06T12:00" "2025-09-07T06:00"
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v04/save_state.snf" --radius 6 9 12 15 50 80
python -m sunerf.evaluation.cme.demo_video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v07/save_state.snf"

python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --longitudes 90 100 110 120 130 140 150 160 --time_range "2025-09-06T12:00" "2025-09-07T06:00"

# wiggle
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/wiggle" --longitudes 0 30 60 90 120 150 180 --time_range "2025-09-12T18:00" "2025-09-12T23:00"


python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/punch" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/pB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_fast_v02/cor2" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*"




python -m sunerf.evaluation.cme.plot_coverage \
  --tb-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" \
  --tb-label "STEREO-A/COR2" \
  --tb-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/tB/*" \
  --tb-label "PUNCH/WFI" \
  --tb-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*" \
  --tb-label "CCOR" \
  --pb-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" \
  --pb-label "STEREO-A/COR2" \
  --pb-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam/pB/*" \
  --pb-label "PUNCH/WFI" \
  --out-path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/coverage_2025_09.png" \
  --title "2025-09 coverage"
