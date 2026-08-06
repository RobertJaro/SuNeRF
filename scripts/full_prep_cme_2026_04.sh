#!/bin/bash -l

set -euo pipefail

module load conda
module load cuda
conda activate lightning

REPO_DIR="/glade/u/home/rjarolim/projects/SuNeRF"
BASE_DIR="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04"

# The download helpers treat --end as exclusive. This downloads all of April and May 2026.
START_TS="2026-04-01T00:00:00"
END_TS="2026-05-01T00:00:00"

CCOR_RAW_DIR="${BASE_DIR}/ccor_l2"
COR_RAW_DIR="${BASE_DIR}/cor"
COR_PREP_DIR="${BASE_DIR}/cor_prep"
COR_CLEAR_RAW_DIR="${COR_RAW_DIR}/clear"
PREP_DIR="${BASE_DIR}/prep"

cd "${REPO_DIR}"

mkdir -p \
  "${CCOR_RAW_DIR}" \
  "${COR_RAW_DIR}" \
  "${COR_PREP_DIR}" \
  "${LASCO_RAW_DIR}" \
  "${LASCO_PREP_DIR}" \
  "${PREP_DIR}"

echo "Downloading CCOR full cadence"
python -m sunerf.data.download.download_ccor \
  --start "${START_TS}" \
  --end "${END_TS}" \
  --cadence all \
  --out "${CCOR_RAW_DIR}" \
  --product-prefix "SWFO/GOES-19/CCOR-1/ccor1-l2_science"

echo "Downloading STEREO-A/COR2 full cadence"
python -m sunerf.data.download.download_cor \
  --start "${START_TS}" \
  --end "${END_TS}" \
  --out "${COR_RAW_DIR}" \
  --detector COR2

echo "Checking COR2 triplets"
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "${COR_RAW_DIR}/*.fts" \
  --clear-dir "${COR_RAW_DIR}/clear"

echo "Running SECCHI_PREP for COR2"
export SECCHI_INPUT_GLOB="${COR_RAW_DIR}/*.fts"
export SECCHI_TB_OUT="${COR_PREP_DIR}/tB"
export SECCHI_PB_OUT="${COR_PREP_DIR}/pB"

csh "${REPO_DIR}/scripts/secchi_prep_triplets.csh"

echo "Running SECCHI_PREP for COR2 clear data"
export SECCHI_CLEAR_INPUT_GLOB="${COR_CLEAR_RAW_DIR}/*.fts"
export SECCHI_CLEAR_OUT="${COR_PREP_DIR}/clear"

csh "${REPO_DIR}/scripts/secchi_prep_clear.csh"

echo "Prepping COR2"
python -m sunerf.data.coronagraph.prep_stereo_cor \
  --tb_path "${COR_PREP_DIR}/tB/*.fts" \
  --pb_path "${COR_PREP_DIR}/pB/*.fts" \
  --out_path "${PREP_DIR}/cor2" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --filter_bright_background_objects

echo "Prepping COR2 clear data"
python -m sunerf.data.coronagraph.prep_coronagraph \
  --data_path "${COR_PREP_DIR}/clear/*.fts" \
  --out_path "${PREP_DIR}/cor2_clear" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --filter_bright_background_objects

echo "Prepping CCOR"
python -m sunerf.data.coronagraph.prep_ccor \
  --data_path "${CCOR_RAW_DIR}/*.fits" \
  --out_path "${PREP_DIR}/ccor" \
  --resize 512 512 \
  --filter_bright_background_objects

echo "Creating validation videos"
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/cor2" \
  --pb "${PREP_DIR}/cor2/pB/*" \
  --tb "${PREP_DIR}/cor2/tB/*"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/cor2_clear" \
  --tb "${PREP_DIR}/cor2_clear/*"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/ccor" \
  --tb "${PREP_DIR}/ccor/*"

python -m sunerf.data.coronagraph.clean_invalid \
  --invalid_files "${PREP_DIR}/invalid_files.txt" \
  --base_path "${PREP_DIR}/**/*"

echo "Computing correction masks"
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/cor2/tB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/cor2/pB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_pB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/cor2_clear/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_clear_correction.npy"


echo "Prep complete. Outputs are under ${PREP_DIR}"
