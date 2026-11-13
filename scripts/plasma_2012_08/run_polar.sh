#!/bin/bash -l

#PBS -N SuNeRF-plasma-polar-2012-08
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=256gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

# Polar reconstruction of the August 2012 data (config/plasma/polar_2012_08.yaml):
# the polar_2025_10 setup on the AIA, EUVI-A and EUVI-B data of this folder,
# with a rendered north-pole view in the validation plots. It shares the
# prepared data and the image-scaling table with run.sh; an existing table is
# reused (--overwrite re-estimates).
python -m sunerf.data.euv.estimate_scaling --config config/plasma/polar_2012_08.yaml --workers 16

python -m sunerf.run_plasma --config config/plasma/polar_2012_08.yaml
