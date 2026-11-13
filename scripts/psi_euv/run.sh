#!/bin/bash -l

#PBS -N SuNeRF-psi-euv
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Closed-loop reconstruction of the synthetic PSI observations written by
# scripts/psi_euv/render.sh, with the shared plasma base configuration.

# Fixed per-channel loss divisors of each instrument, estimated once from the
# training observations; an existing table is reused (--overwrite re-estimates).
python -m sunerf.data.euv.estimate_scaling --config config/plasma/psi_observers.yaml --workers 16

python -m sunerf.run_plasma --config config/plasma/psi_observers.yaml
