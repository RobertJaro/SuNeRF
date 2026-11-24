#!/bin/bash -l

#PBS -N SuNeRF-polar-reversal
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=256gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Full reconstruction of the first window (config/plasma/polar_reversal.yaml),
# trained from scratch. A resubmitted job resumes from the last checkpoint. Its
# save_state.snf is the initial state of the series (run_series.sh).
#
# One image-scaling table for the whole 2011-2014 data set: the divisors are
# estimated from files sampled over all prepared data, so every series step
# shares them. An existing table is reused (--overwrite re-estimates).

python -m sunerf.data.euv.estimate_scaling --config config/plasma/polar_reversal.yaml --workers 16

python -m sunerf.run_plasma --config config/plasma/polar_reversal.yaml
