#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Prepare data for CME reconstruction

## STEREO-A/COR2
python -i -m sunerf.data.punch.download_wget --url "https://umbra.nascom.nasa.gov/punch/1/PZ1/2025/10/17/" --out-dir "/glade/work/rjarolim/data/sunerf-cme/punch"