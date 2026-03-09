#!/bin/bash -l

#PBS -N Noise
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

####################################################################
# Reconstruction
#python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.01
#python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.02
#python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.05
#python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.10
python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.20
#python -m sunerf.run_thomson --config "config/cme/noise_3view.yaml" --noise 0.30

####################################################################
# Evaluation
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.01_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.02_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.05_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.10_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.20_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.30_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/noise/3view_0.40_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
#
python -i -m sunerf.evaluation.cme.noise_comparison
python -i -m sunerf.evaluation.cme.plot_noise_sample