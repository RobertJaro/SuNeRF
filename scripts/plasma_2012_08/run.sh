#!/bin/bash -l

#PBS -N SuNeRF-plasma-2012-08
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

# Fixed per-channel loss divisors of AIA, EUVI-A, and EUVI-B, estimated once from
# the prepared training observations; an existing table is reused (--overwrite
# re-estimates).
python -m sunerf.data.euv.estimate_scaling --config config/plasma/all_2012_08.yaml --workers 16

python -m sunerf.run_plasma --config config/plasma/all_2012_08.yaml
