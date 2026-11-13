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

PUNCH_PAM_RAW_DIR="${DATA_ROOT}/punch_pam"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Prepare total- and polarized-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_punch_pam \
  --data_path "${PUNCH_PAM_RAW_DIR}/*.fits" \
  --out_path "${PREP_DIR}/punch_pam" \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --max_radius 120 \
  --overwrite

################################################################################
# Quicklook movie
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/punch_pam" \
  --tb "${PREP_DIR}/punch_pam/tB/*" \
  --pb "${PREP_DIR}/punch_pam/pB/*" \
  --num_workers "${WORKERS}"
