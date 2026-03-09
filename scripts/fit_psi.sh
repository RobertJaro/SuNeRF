#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

python -i -m sunerf.train.fit_psi --temperature_response_file "/glade/work/rjarolim/sunerf/response/aia_interpolated.npz" --data_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd" --work_directory "/glade/derecho/scratch/rjarolim/sunerf/psi" --out_path "/glade/work/rjarolim/sunerf/psi_cube_v2"

# generate video
python -i -m  sunerf.evaluation.video --chk_path "/glade/work/rjarolim/sunerf/psi_cube_v2/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/psi_cube_v2/video"
