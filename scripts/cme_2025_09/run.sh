#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Train coarse/background SuNeRF model
################################################################################
python -m sunerf.run_thomson --config "config/cme/2025_09.yaml"

################################################################################
# Fine-tune the CME window with fixed instrument corrections
################################################################################
#python -m sunerf.run_thomson --config "config/cme/2025_09_cme.yaml"
#python -m sunerf.run_thomson --config "config/cme/2025_09_22_cme.yaml"
#python -m sunerf.run_thomson --config "config/cme/2025_09_04_cme.ycaml"
