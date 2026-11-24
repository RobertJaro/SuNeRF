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
REPO_DIR="/glade/u/home/rjarolim/projects/SuNeRF"
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
WORKERS="16"

COR_RAW_DIR="${DATA_ROOT}/cor"
COR_PREP_DIR="${DATA_ROOT}/cor_prep"
COR_CLEAR_RAW_DIR="${COR_RAW_DIR}/clear"
PREP_DIR="${DATA_ROOT}/prep"
################################################################################
# Run SECCHI_PREP through SolarSoft/IDL
################################################################################
# Produces total-brightness, polarized-brightness, and clear FITS products.
csh "${REPO_DIR}/scripts/secchi_prep_triplets.csh" \
  "${COR_RAW_DIR}/*.fts" "${COR_PREP_DIR}/tB" "${COR_PREP_DIR}/pB"

csh "${REPO_DIR}/scripts/secchi_prep_clear.csh" \
  "${COR_CLEAR_RAW_DIR}/*.fts" "${COR_PREP_DIR}/clear"

################################################################################
# Prepare total- and polarized-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_stereo_cor \
  --tb_path "${COR_PREP_DIR}/tB/*.fts" \
  --pb_path "${COR_PREP_DIR}/pB/*.fts" \
  --out_path "${PREP_DIR}/cor2" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --num_workers "${WORKERS}"

################################################################################
# Prepare clear total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_coronagraph \
  --data_path "${COR_PREP_DIR}/clear/*.fts" \
  --out_path "${PREP_DIR}/cor2_clear" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --n_images 3 6 \
  --num_workers "${WORKERS}"
