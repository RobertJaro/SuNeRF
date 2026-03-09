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

# STEREO-A/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/tB" --occ_min 3000 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/pB" --occ_min 3000 --occ_max 15000 --resize 512 512

# STEREO-B/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/tB" --occ_min 3000 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/pB" --occ_min 3000 --occ_max 15000 --resize 512 512


#################################################
# check data
# COR2
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/video/stereo_a_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/tB/*" --vmin 1e-12 --vmax 1e-8
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/video/stereo_b_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/tB/*" --vmin 1e-12 --vmax 1e-8

#################################################
# clean invalid files
#python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/**/*" --dry_run

#################################################
# Train
#python -m sunerf.run_thomson --config "config/cme/201208_cor2.yaml"

python -m sunerf.run_thomson --config "config/cme/201208_cor2_stereo_a.yaml"


#################################################
# Evaluation

python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/pB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/ref_series_stereo_b"

python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/save_state.snf"
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/save_state.snf" --longitudes 0 15 30 45 60 75 90
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_v05/save_state.snf" --radius 5 8 12 15


# STEREO A only
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_stereo_a_v04/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_a_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_stereo_a_v04/ref_series_stereo_a"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_stereo_a_v04/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2012_08/prep/stereo_b_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/201208_cor2_stereo_a_v04/ref_series_stereo_b"
