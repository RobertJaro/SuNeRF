#!/bin/bash -l

#PBS -N SuNeRF-plasma-polar-2025-10
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Loss divisors: packaged table sunerf/resources/calibration/image_scaling.yaml
# (builtin:image_scaling), estimated from a 200-file sample of this campaign.
python -m sunerf.run_plasma --config config/plasma/polar_2025_10.yaml
