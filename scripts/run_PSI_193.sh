#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q casper
#PBS -l select=1:ncpus=8:ngpus=2:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00


# Training from scratch
# init workspace
#conda activate sunerf
#cd /home/rjarolim/projects/SuNeRF

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# convert data
#python -m sunerf.data.prep.psi --psi_path "/glade/work/rjarolim/data/sunerf/psi_data/psi_data/193/*.fits" --output_path "/glade/work/rjarolim/data/sunerf/prep_psi/193"


python -m sunerf.run_emission --config "config/psi_193.yaml"