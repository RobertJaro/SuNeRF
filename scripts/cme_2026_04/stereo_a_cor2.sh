#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
REPO_DIR="/glade/u/home/rjarolim/projects/SuNeRF"
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04"
START="2026-04-01T00:00:00"
END="2026-05-01T00:00:00"
WORKERS="10"

COR_RAW_DIR="${DATA_ROOT}/cor"
COR_PREP_DIR="${DATA_ROOT}/cor_prep"
COR_CLEAR_RAW_DIR="${COR_RAW_DIR}/clear"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Download STEREO-A COR2 polarized triplets
################################################################################
python -m sunerf.data.download.download_cor \
  --start "${START}" \
  --end "${END}" \
  --out "${COR_RAW_DIR}" \
  --detector COR2

################################################################################
# Remove incomplete or invalid polarization triplets
################################################################################
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "${COR_RAW_DIR}/*.fts" \
  --clear-dir "${COR_CLEAR_RAW_DIR}"

################################################################################
# Run SECCHI_PREP through SolarSoft/IDL
################################################################################
# Produces total-brightness, polarized-brightness, and clear FITS products.
export SECCHI_INPUT_GLOB="${COR_RAW_DIR}/*.fts"
export SECCHI_TB_OUT="${COR_PREP_DIR}/tB"
export SECCHI_PB_OUT="${COR_PREP_DIR}/pB"
csh "${REPO_DIR}/scripts/secchi_prep_triplets.csh"

export SECCHI_CLEAR_INPUT_GLOB="${COR_CLEAR_RAW_DIR}/*.fts"
export SECCHI_CLEAR_OUT="${COR_PREP_DIR}/clear"
csh "${REPO_DIR}/scripts/secchi_prep_clear.csh"

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
  --filter_bright_background_objects \
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
  --filter_bright_background_objects \
  --num_workers "${WORKERS}"

################################################################################
# Compute correction masks
################################################################################
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-percentile \
  --input "${PREP_DIR}/cor2/tB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-percentile \
  --input "${PREP_DIR}/cor2/pB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_pB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-percentile \
  --input "${PREP_DIR}/cor2_clear/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_clear_correction.npy"

################################################################################
# Quicklook movies
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/cor2" \
  --pb "${PREP_DIR}/cor2/pB/*" \
  --tb "${PREP_DIR}/cor2/tB/*" \
  --pb_correction "${PREP_DIR}/masks/stereo_a_cor2_pB_correction.npy" \
  --tb_correction "${PREP_DIR}/masks/stereo_a_cor2_tB_correction.npy" \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/cor2_clear" \
  --tb "${PREP_DIR}/cor2_clear/*" \
  --tb_correction "${PREP_DIR}/masks/stereo_a_cor2_tB_clear_correction.npy" \
  --num_workers "${WORKERS}"
