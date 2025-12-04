#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=32gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Data Preparation
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme1/data_fits/*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme1/prep_4view" --check_matching
# 2-view setup
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme1/data_fits/cme1_dcmer_030E_bang_0000_pB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme1/prep_2view"
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme1/data_fits/cme1_dcmer_030E_bang_0000_tB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme1/prep_2view"
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme1/data_fits/cme1_dcmer_090E_bang_0000_pB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme1/prep_2view"
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme1/data_fits/cme1_dcmer_090E_bang_0000_tB/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme1/prep_2view"


#################################################
# Training
#python -m sunerf.run_thomson --config "config/cme/hao_cme1_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme/hao_cme1_3view.yaml"
python -m sunerf.run_thomson --config "config/cme/hao_cme1_4view.yaml"


##################################################
# Evaluation
#python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme1_3view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme1/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude -25 --date0 "2010-05-27T07:04:39.000"
#python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme1_2view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme1/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude -25 --date0 "2010-05-27T07:04:39.000"
#python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme1_4view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme1/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude -25 --date0 "2010-05-27T07:04:39.000"
