#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/2010_03"
START="2010-03-15"
END="2010-04-15"
WORKERS="10"

################################################################################
# Download LASCO C2 archive data
################################################################################
# level_1 provides clear total-brightness images.
# polarized provides PB and %P products used to derive polarized tB.
python -m sunerf.data.download.download_lasco \
  --start "${START}" \
  --end "${END}" \
  --out "${DATA_ROOT}/lasco" \
  --instrument C2 \
  --product level_1 polarized \
  --workers "${WORKERS}"

################################################################################
# Prepare clear level-1 LASCO C2 tB
################################################################################
# Writes: ${DATA_ROOT}/prep/lasco_c2_clear/tB
python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${DATA_ROOT}/lasco/c2/level_1/*.fts*" \
  --out_path "${DATA_ROOT}/prep/lasco_c2_clear" \
  --occ_min 2100 \
  --occ_max 8000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite

################################################################################
# Prepare polarized LASCO C2 pB and derived tB
################################################################################
# PB is used directly as pB.
# %P is used with pB to compute tB = pB / (%P / 100).
# Writes: ${DATA_ROOT}/prep/lasco_c2/{tB,pB}
python -m sunerf.data.coronagraph.prep_lasco \
  --pb_path "${DATA_ROOT}/lasco/c2/polarized/C2-PB-*.fts" \
  --percent_path "${DATA_ROOT}/lasco/c2/polarized/C2-%25P-*.fts" \
  --out_path "${DATA_ROOT}/prep/lasco_c2" \
  --occ_min 2100 \
  --occ_max 8000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --filter_bright_objects \
  --overwrite

################################################################################
# Compute correction masks
################################################################################
# These masks are subtracted in quicklook rendering and can be referenced by
# training configs.
python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${DATA_ROOT}/prep/lasco_c2_clear/tB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_clear_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/lasco_c2/pB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_pB_correction.npy"

################################################################################
# Quicklook movies
################################################################################
# Produces frame folders plus zip archives for visual inspection.
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${DATA_ROOT}/prep/video/lasco_c2_clear" \
  --tb "${DATA_ROOT}/prep/lasco_c2_clear/tB/*" \
  --tb_correction "${DATA_ROOT}/prep/masks/lasco_c2_clear_tB_correction.npy" \
  --vmin 1e-12 \
  --vmax 1e-8 \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${DATA_ROOT}/prep/video/lasco_c2_polarized" \
  --tb "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --pb "${DATA_ROOT}/prep/lasco_c2/pB/*" \
  --tb_correction "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy" \
  --pb_correction "${DATA_ROOT}/prep/masks/lasco_c2_pB_correction.npy" \
  --vmin 1e-12 \
  --vmax 1e-8 \
  --num_workers "${WORKERS}"

################################################################################
# Invalid-file report
################################################################################
# Dry-run only: prints what would be removed based on invalid_files.txt.
python -m sunerf.data.coronagraph.clean_invalid \
  --invalid_files "${DATA_ROOT}/invalid_files.txt" \
  --base_path "${DATA_ROOT}/prep/**/*" \
  --dry_run
