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
SSW_ROOT="/glade/work/rjarolim/ssw"

LASCO_RAW_DIR="${DATA_ROOT}/lasco"
LASCO_L1_DIR="${DATA_ROOT}/lasco_l1"
ANCIL_DIR="${DATA_ROOT}/soho_ancillary"
PREP_DIR="${DATA_ROOT}/prep"

################################################################################
# Calibrate level 0.5 to level 1 with SolarSoft REDUCE_LEVEL_1
################################################################################
csh "${REPO_DIR}/scripts/lasco_reduce_level_1.csh" \
  "${LASCO_RAW_DIR}/c2/level_05/*.fts" "${LASCO_L1_DIR}/c2" "${SSW_ROOT}" "${ANCIL_DIR}"

csh "${REPO_DIR}/scripts/lasco_reduce_level_1.csh" \
  "${LASCO_RAW_DIR}/c3/level_05/*.fts" "${LASCO_L1_DIR}/c3" "${SSW_ROOT}" "${ANCIL_DIR}"

################################################################################
# Prepare total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${LASCO_L1_DIR}/c2/*.fts" \
  --out_path "${PREP_DIR}/lasco_c2" \
  --occ_min 2100 \
  --occ_max 8000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite

python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${LASCO_L1_DIR}/c3/*.fts" \
  --out_path "${PREP_DIR}/lasco_c3" \
  --occ_min 4000 \
  --occ_max 29000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite
