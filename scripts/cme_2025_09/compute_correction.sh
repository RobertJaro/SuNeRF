#!/usr/bin/env bash
# Server, after frame_review apply. Compute the instrument backgrounds.
set -euo pipefail
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
PREP_DIR="${DATA_ROOT}/prep"
MASK_SMOOTH_ANGLE_DEG=60
MASK_SMOOTH_FRACTION=0.02
mkdir -p "${PREP_DIR}/masks"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/cor2/tB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_correction.npy" \
  --smooth-angle-deg "${MASK_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${MASK_SMOOTH_FRACTION}"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/cor2/pB/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_pB_correction.npy" \
  --smooth-angle-deg "${MASK_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${MASK_SMOOTH_FRACTION}"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/cor2_clear/*" \
  --output "${PREP_DIR}/masks/stereo_a_cor2_tB_clear_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${PREP_DIR}/ccor/*" \
  --output "${PREP_DIR}/masks/ccor_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/lasco_c2/tB/*" \
  --output "${PREP_DIR}/masks/lasco_c2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type daily-min \
  --input "${PREP_DIR}/lasco_c3/tB/*" \
  --output "${PREP_DIR}/masks/lasco_c3_tB_correction.npy"
