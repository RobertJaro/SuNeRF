#!/usr/bin/env bash
# Compare the closed-loop reconstruction with the PSI/MAS cube it was rendered from.
# REFERENCE_DATE and LIMB_RADIUS must match scripts/psi_euv/render.sh.
REFERENCE_DATE="2026-01-01T00:00"
LIMB_RADIUS='1.007'
PSI_DATA="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd"
RESULT_ROOT="/glade/work/rjarolim/sunerf/sunerf/psi_euv"

python -m sunerf.evaluation.psi_euv_truth \
  --psi-data "${PSI_DATA}" \
  --reconstruction "${RESULT_ROOT}/save_state.safe.pt" \
  --time "${REFERENCE_DATE}" \
  --min-radius "${LIMB_RADIUS}" \
  --absorption-artifact none \
  --out-path "${RESULT_ROOT}/evaluation" --save-truth
