#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Prepare data for CME reconstruction

## STEREO-A/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB" --occ_min 4000 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB" --occ_min 4000 --occ_max 15000 --resize 512 512
#
## STEREO-B/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB" --occ_min 4500 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB" --occ_min 4500 --occ_max 15000 --resize 512 512
#
#
## fix LASCO headers and basic prep
#python -m sunerf.data.lasco.fix_observer "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep/*" --out-dir "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep_fixed"
#
## SOHO/LASCO C2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep_fixed/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/lasco_c2" --occ_min 2100 --occ_max 8000 --resize 512 512 --clip_max 1e+5


#################################################
# check data
# COR2
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/stereo_a_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB/*" --vmin 1e-12 --vmax 1e-8
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/stereo_b_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB/*" --vmin 1e-12 --vmax 1e-8
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/lasco_c2" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/lasco_c2/*" --vmin 1e-12 --vmax 1e-8


#################################################
# clean invalid files
#python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/**/*" --dry_run

#################################################
# compute correction masks
#python -m sunerf.data.coronagraph.compute_correction --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_cor2_tB_correction.npy"
#python -m sunerf.data.coronagraph.compute_correction --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_cor2_pB_correction.npy"
#python -m sunerf.data.coronagraph.compute_correction --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_cor2_tB_correction.npy"
#python -m sunerf.data.coronagraph.compute_correction --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_cor2_pB_correction.npy"

#################################################
# train correction model
#python -i -m sunerf.train.fit_background --file_path '/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*' --out_dir '/glade/work/rjarolim/sunerf-cme-obs/background_correction/stereo_b_pB'
#python -i -m sunerf.train.fit_background --file_path '/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB/*' --out_dir '/glade/work/rjarolim/sunerf-cme-obs/background_correction/stereo_b_tB'

########### Fit scaling masks ###########
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" \
#  --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_pB_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_pB_fit.png" \
#  --degree 3
#
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB/*" \
#  --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_tB_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_a_tB_fit.png" \
#  --degree 3
#
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" \
#  --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_pB_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_pB_fit.png" \
#  --degree 3
#
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB/*" \
#  --output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_tB_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/masks/stereo_b_tB_fit.png" \
#  --degree 3

#################################################
# Train
#python -m sunerf.run_thomson --config "config/cme/2010_03_cor2_subtracted.yaml"
python -m sunerf.run_thomson --config "config/cme/2010_03_cor2_correction.yaml"
#python -m sunerf.run_thomson --config "config/cme/2010_03_cor2_physics.yaml"
#python -m sunerf.run_thomson --config "config/cme/2010_03_cor2_stereo_a.yaml"

exit

#################################################
# Download validation data
python -m sunerf.data.download.download_aia \
  --download_dir "/glade/work/rjarolim/data/sunerf-cme/aia_validation" \
  --email robert.jarolim@uni-graz.at \
  --t_start 2010-03-15T00:00:00 \
  --t_end 2010-04-07T00:00:00 \
  --cadence 1d \
  --channel 193

#################################################
# Evaluation

# ref series
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/ref_series_stereo_a" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/ref_series_stereo_b" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*"
# tomography
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf" --longitudes 0 30 60 90 120 150 180 --time_range "2010-03-19T00:00" "2010-03-20T00:00"
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/tomography_cme" --longitudes 80 90 100 110 120 130 --time_range "2010-03-19T00:00" "2010-03-21T00:00"
# radius map
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf" --radius 5 8 12 15
# video
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf"
python -m sunerf.evaluation.cme.video_polar --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf"
python -m sunerf.evaluation.cme.video_fixed_rotation --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v08/save_state.snf"
# export cubes
python -m sunerf.evaluation.cme.export_cubes --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v04/save_state.snf" --time_range "2010-03-19T00:00" "2010-03-21T00:00"

# STEREO A only
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v08/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v08/ref_series_stereo_a"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v08/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v08/ref_series_stereo_b"
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v08/save_state.snf"

# compare to Carrington maps
python -m sunerf.evaluation.cme.carrington_map_comparison \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_correction_v06/sunerf_state.pt" \
  --aia_glob "/glade/work/rjarolim/data/sunerf-cme/aia_validation/*.fits" \
  --out_path "/glade/work/rjarolim/data/sunerf-cme/aia_validation/carrington_comparison"