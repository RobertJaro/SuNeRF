#!/bin/bash -l
#PBS -N psi-insitu
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Degraded views with gain plus the synthetic PSP probe
# (scripts/psi_test/prepare_insitu.sh must have written
# /glade/work/rjarolim/data/sunerf-cme/psi_obs/insitu/PSP_synth*.npz)
################################################################################
python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_gain_2view_insitu.yaml"
