#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=06:00:00

set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
WORKERS="16"

CCOR_RAW_DIR="${DATA_ROOT}/ccor_l2"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Prepare total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_ccor \
  --data_path "${CCOR_RAW_DIR}/*.fits" \
  --out_path "${PREP_DIR}/ccor" \
  --resize 512 512 \
  --num_workers "${WORKERS}"
