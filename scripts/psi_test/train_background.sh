#!/bin/bash -l
#PBS -N psi-background
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Stage 1 of the two-stage fit: degraded+gain 2-view, background datasets only
# (fixed 6 h cadence over the full series, eruption phase included). Learns the
# corotating corona and the image-plane degradation fields. Checkpoint-resumable;
# resubmit if the fit reaches the walltime. Writes
# /glade/work/rjarolim/sunerf-cme-obs/psi_obs/degraded_gain_2view_background_v01/final.ckpt,
# the initialization of scripts/psi_test/train_cme.sh.
################################################################################
python -m sunerf.run_thomson --config "config/cme/psi/psi_degraded_gain_2view_background.yaml"
