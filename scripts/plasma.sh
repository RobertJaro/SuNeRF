#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#python -m sunerf.data.euv.prep_aia_v2 --data_path "/glade/work/rjarolim/data/sunerf/2012_08/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/aia" --resolution 512
#python -m sunerf.data.euv.prep_euvi --data_path "/glade/work/rjarolim/data/sunerf/2012_08/euvi_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf/2012_08_prep/euvi" --resolution 512
#python -m sunerf.data.euv.prep_psi --data_path "/glade/work/rjarolim/data/sunerf/psi_data/psi_data/**/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/psi_data_prep" --resolution 1024

# prep 2010-06
#python -m sunerf.data.euv.prep_aia_v2 --data_path "/glade/work/rjarolim/data/sunerf/2010_06/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2010_06_prep/aia" --resolution 512
#python -m sunerf.data.euv.prep_euvi --data_path "/glade/work/rjarolim/data/sunerf/2010_06/euvi_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf/2010_06_prep/euvi" --resolution 512


#python -m sunerf.run_plasma --config "config/plasma/aia_2012_08.yaml"
python -m sunerf.run_plasma --config "config/plasma/all_2012_08.yaml"
#python -m sunerf.run_plasma --config "config/plasma/combined_2023_04.yaml"
#python -m sunerf.run_plasma --config "config/plasma/aia_euvi_2010_06.yaml"
#python -m sunerf.run_plasma --config "config/plasma/euvi_2010_06.yaml"

# Debugging
#python -i -m sunerf.run_plasma --config "config/psi_plasma_193.yaml"


########### EVALUATION ###########

#python -m  sunerf.evaluation.video --chk_path "/glade/work/rjarolim/sunerf/all_2012_08_v04/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2012_08_v04/evaluation/video"
#python -m  sunerf.evaluation.video_observer --chk_path "/glade/work/rjarolim/sunerf/all_2012_08_v01/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2012_08_v01/evaluation/video_observer"
#python -m  sunerf.evaluation.slices --chk_path "/glade/work/rjarolim/sunerf/all_2012_08_v01/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2012_08_v01/evaluation/slices"
#python -m  sunerf.evaluation.load_cube --chk_path "/glade/work/rjarolim/sunerf/aia_v01/save_state.snf" --out_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_3D_cube/sunerf_cube.npz"
#python -m  sunerf.evaluation.video_poles --chk_path "/glade/work/rjarolim/sunerf/all_2012_08_v05/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2012_08_v05/evaluation/video_poles"

