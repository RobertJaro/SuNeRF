#!/usr/bin/env bash
# Server, after compute_correction.sh. Generate quicklook movies.
set -euo pipefail
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
PREP_DIR="${DATA_ROOT}/prep"
PUNCH_DIR="${DATA_ROOT}/prep_stash/punch_pam"
WORKERS="16"

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

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/ccor" \
  --tb "${PREP_DIR}/ccor/*" \
  --tb_correction "${PREP_DIR}/masks/ccor_tB_correction.npy" \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/lasco_c2" \
  --tb "${PREP_DIR}/lasco_c2/tB/*" \
  --tb_correction "${PREP_DIR}/masks/lasco_c2_tB_correction.npy" \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/lasco_c3" \
  --tb "${PREP_DIR}/lasco_c3/tB/*" \
  --tb_correction "${PREP_DIR}/masks/lasco_c3_tB_correction.npy" \
  --num_workers "${WORKERS}"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PREP_DIR}/video/punch_pam" \
  --tb "${PUNCH_DIR}/tB/*" \
  --pb "${PUNCH_DIR}/pB/*" \
  --num_workers "${WORKERS}"
