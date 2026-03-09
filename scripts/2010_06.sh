#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2010_06/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2010_06_prep/aia" --resolution 512
#python -m sunerf.data.euv.prep_euvi --data_path "/glade/work/rjarolim/data/sunerf/2010_06/euvi_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf/2010_06_prep/euvi" --resolution 512

python -i -m sunerf.run_plasma --config "config/all_2010_06.yaml"


########### EVALUATION ###########

#python -i -m  sunerf.evaluation.video_observer --chk_path "/glade/work/rjarolim/sunerf/all_2010_06_v01/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2010_06_v01/video"
#python -i -m  sunerf.evaluation.slices --chk_path "/glade/work/rjarolim/sunerf/all_2010_06_v01/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/all_2010_06_v01/slices"
