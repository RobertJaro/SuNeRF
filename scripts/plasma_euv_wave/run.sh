#!/bin/bash -l

#PBS -N SuNeRF-plasma-euv-wave-2022-03
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=256gb
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Loss divisors: packaged table sunerf/resources/calibration/image_scaling.yaml
# (builtin:image_scaling, estimated on the polar 2025-10 campaign).
#python -m sunerf.run_plasma --config config/plasma/euv_wave_2022_03.yaml

python -m sunerf.run_plasma --config config/plasma/euv_wave_2022_03.yaml
