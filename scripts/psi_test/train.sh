#!/bin/bash -l
#PBS -N psi-fit
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Train the clean two-view baseline
################################################################################
python -m sunerf.run_thomson --config "config/cme/psi_clean_2view.yaml"
python -m sunerf.run_thomson --config "config/cme/psi_clean_3view.yaml"

python -m sunerf.run_thomson --config "config/cme/psi_degraded_2view.yaml"
python -m sunerf.run_thomson --config "config/cme/psi_degraded_3view.yaml"

################################################################################
# Train the degraded two-view reconstruction
################################################################################
#python -m sunerf.run_thomson --config "config/cme/psi_degraded_2view.yaml"
