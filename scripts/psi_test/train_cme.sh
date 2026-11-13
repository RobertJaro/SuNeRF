#!/bin/bash -l
#PBS -N psi-cme
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Stage 2 of the two-stage fit: continues from final.ckpt of train_background.sh
# (training.init_path: model weights only, fresh optimizer and step count) with
# the degradation fields fixed and two datasets per view, the 6 h background
# series and every frame of the eruption phase. Submit after stage 1 completed.
# Checkpoint-resumable: a resubmission continues from last.ckpt of this run and
# reads the stage-1 weights only on the first start.
################################################################################
python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_gain_2view_cme.yaml"
