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
# Clean views (no degradation)
################################################################################
#python -m sunerf.run_thomson --config "config/cme/psi/psi_clean_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme/psi/psi_clean_3view.yaml"

################################################################################
# Degraded views: additive mask, noise, background subtraction (no gain)
################################################################################
#python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_3view.yaml"

################################################################################
# Degraded views with a per-view multiplicative gain (tB_mul/pB_mul fields);
# see also train_gain_2view.sh and train_gain_3view.sh
################################################################################
#python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_gain_2view.yaml"
python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_gain_3view.yaml"
