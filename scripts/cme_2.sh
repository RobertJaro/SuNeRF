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
# Data download
#python -m sunerf.data.hao_cme.download_challenge --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits"


#################################################
# Data Preparation
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_4view" --check_matching

# 2 viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/cme2_dcmer_030E_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_2view" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/cme2_dcmer_090E_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_2view" --check_matching

# 3 viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/cme2_dcmer_030E_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_3view" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/cme2_dcmer_090E_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_3view" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/cme2/data_fits/cme2_dcmer_060W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/cme2/prep_3view" --check_matching


#################################################
# Training
#python -m sunerf.run_thomson --config "config/cme/hao_cme2_2view.yaml" --reload
#python -m sunerf.run_thomson --config "config/cme/hao_cme2_3view.yaml" --reload
python -m sunerf.run_thomson --config "config/cme/hao_cme2_4view.yaml" --reload

##################################################
# Evaluation
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme2_2view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme2/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude 35 --date0 "2010-05-27T07:04:39.000"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme2_3view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme2/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude 35 --date0 "2010-05-27T07:04:39.000"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/cme2_4view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/cme2/density_cube/*.sav" --min_longitude 0 --max_longitude 180 --plot_ground_truth --target_longitude 60 --target_latitude 35 --date0 "2010-05-27T07:04:39.000"
