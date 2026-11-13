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

################################################################################
# Compute correction mask
################################################################################
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/ccor/*" \
  --output "${PREP_DIR}/masks/ccor_tB_correction.npy"

################################################################################
# Quicklook movie
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/ccor" \
  --tb "${PREP_DIR}/ccor/*" \
  --tb_correction "${PREP_DIR}/masks/ccor_tB_correction.npy" \
  --num_workers "${WORKERS}"
