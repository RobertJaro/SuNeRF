#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Data Preparation
python -m sunerf.data.conditioned.convert_aia_data \
  --input "/glade/work/rjarolim/data/sunerf/2012_08_prep/aia/*.193.*.fits" \
  --output "/glade/work/rjarolim/data/sunerf-conditioned/aia_193_npz" \
  --nproc 16

#################################################
# Training
python -m sunerf.run_conditioned --config "config/conditioned/aia_193.yaml"
