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

python -m sunerf.data.psi.build_synthetic \
  --temperature-response-artifact "/glade/work/rjarolim/sunerf/responses/chianti_11.0.2_coronal_2021/aia_2012_08.sunerf.npz" \
  --data-path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd" \
  --out-path "/glade/work/rjarolim/sunerf/psi_cube_v4" \
  --source-density-scale-cm3 1e8 \
  --source-temperature-scale-k 2.807066716734894e7 \
  --reference-frame-id 1813 \
  --longitude-frame carrington \
  --workers 16

# Evaluation reads ordered channels and response metadata from the artifact.
python -m sunerf.evaluation.video \
  --chk_path "/glade/work/rjarolim/sunerf/psi_cube_v4/save_state.grid.pt" \
  --video_path "/glade/work/rjarolim/sunerf/psi_cube_v4/video" \
  --instrument-key PSI
