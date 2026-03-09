#!/bin/bash -l

#PBS -N SuNeRF
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
# Prep Data
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polarization_020" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_080W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polarization_080" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polarization_320" --check_matching

####################################################################
# Reconstruction
#python -m sunerf.run_thomson --config "config/cme/polarization/mixed_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme/polarization/mixed_3view.yaml"
python -m sunerf.run_thomson --config "config/cme/polarization/no_pol_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme/polarization/no_pol_3view.yaml"

####################################################################
# Evaluation
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/full_polarization_v02/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/mixed_3view_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/mixed_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/no_3view_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/no_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"


#python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/full_polarization_v02/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --plot_ground_truth
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/mixed_3view_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/mixed_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/no_3view_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polarization/no_polarization_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
