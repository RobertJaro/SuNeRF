#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
# Clean tB/pB sequences and density cubes come from scripts/psi_test/render_clean.sh,
# which must run first. Frame 0074 renders cube 50 after 24 pre-event frames.
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
PSI_CLEAN_DIR="${DATA_ROOT}/clean"
PSI_DEGRADED_DIR="${DATA_ROOT}/degraded"
PSI_MASK_DIR="${DATA_ROOT}/masks"
PSI_TRUTH_DIR="${DATA_ROOT}/degradation_truth"
PSI_PREVIEW_DIR="${DATA_ROOT}/degradation_previews"
TRUTH_FRAME="0074"
# The pB masks are smoothed around the Sun to suppress their streamer structure:
# Gaussian widths in position angle and, as a fraction of the image width, along
# the radius.
PB_SMOOTH_ANGLE_DEG=20
PB_SMOOTH_FRACTION=0.02
STEREO_PREP_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2010_03/prep"
LASCO_PREP_ROOT="/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/lasco_c2"
PUNCH_PREP_ROOT="/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch_pam"
CCOR_PREP_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/ccor"

################################################################################
# Derive each daily-min mask independently from the prepared observations of
# the instrument that the synthetic view imitates. The pB masks are smoothed.
################################################################################
python -m sunerf.data.psi.extract_noise_mask \
  --input "${LASCO_PREP_ROOT}/tB/*" \
  --output "${PSI_MASK_DIR}/LASCO_C2_tB_daily_min.fts" \
  --min-frames-per-day 4


python -m sunerf.data.psi.extract_noise_mask \
  --input "${LASCO_PREP_ROOT}/pB/*" \
  --output "${PSI_MASK_DIR}/LASCO_C2_pB_daily_min.fts" \
  --smooth-angle-deg "${PB_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${PB_SMOOTH_FRACTION}" \
  --min-frames-per-day 4

python -m sunerf.data.psi.extract_noise_mask \
  --input "${STEREO_PREP_ROOT}/stereo_a_cor2/tB/*" \
  --output "${PSI_MASK_DIR}/STEREO_A_COR2_tB_daily_min.fts"

python -m sunerf.data.psi.extract_noise_mask \
  --input "${STEREO_PREP_ROOT}/stereo_a_cor2/pB/*" \
  --output "${PSI_MASK_DIR}/STEREO_A_COR2_pB_daily_min.fts" \
  --smooth-angle-deg "${PB_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${PB_SMOOTH_FRACTION}"

python -m sunerf.data.psi.extract_noise_mask \
  --input "${STEREO_PREP_ROOT}/stereo_b_cor2/tB/*" \
  --output "${PSI_MASK_DIR}/STEREO_B_COR2_tB_daily_min.fts"

python -m sunerf.data.psi.extract_noise_mask \
  --input "${STEREO_PREP_ROOT}/stereo_b_cor2/pB/*" \
  --output "${PSI_MASK_DIR}/STEREO_B_COR2_pB_daily_min.fts" \
  --smooth-angle-deg "${PB_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${PB_SMOOTH_FRACTION}"

python -m sunerf.data.psi.extract_noise_mask \
  --input "${PUNCH_PREP_ROOT}/tB/*" \
  --output "${PSI_MASK_DIR}/PUNCH_tB_daily_min.fts"

python -m sunerf.data.psi.extract_noise_mask \
  --input "${PUNCH_PREP_ROOT}/pB/*" \
  --output "${PSI_MASK_DIR}/PUNCH_pB_daily_min.fts" \
  --smooth-angle-deg "${PB_SMOOTH_ANGLE_DEG}" \
  --smooth-fraction "${PB_SMOOTH_FRACTION}"

# CCOR observes total brightness only.
python -m sunerf.data.psi.extract_noise_mask \
  --input "${CCOR_PREP_ROOT}/*" \
  --output "${PSI_MASK_DIR}/CCOR_tB_daily_min.fts"

################################################################################
# Apply independent additive detector degradations (no multiplicative gain) over
# the rendered field of view of each training view. The Python routine processes one target and has no
# knowledge of how many viewpoints an experiment uses. P1 stays clear.
################################################################################
# L5_wide, 15--30 R_sun: STEREO-B COR2 pattern.
python -m sunerf.data.psi.degrade_psi \
  --input-dir "${PSI_CLEAN_DIR}/L5_wide" \
  --output-dir "${PSI_DEGRADED_DIR}/L5_wide" \
  --noise-instrument STEREO_B_COR2 \
  --mask-fraction 0.1 \
  --pb-mask "${PSI_MASK_DIR}/STEREO_B_COR2_pB_daily_min.fts" \
  --tb-mask "${PSI_MASK_DIR}/STEREO_B_COR2_tB_daily_min.fts" \
  --truth-dir "${PSI_TRUTH_DIR}/L5_wide" \
  --truth-frame "${TRUTH_FRAME}" \
  --overwrite

# L5, 3.7--17 R_sun: CCOR pattern. CCOR has no pB, so its tB pattern also
# degrades the pB product, which the CCOR-like training configs do not read.
python -m sunerf.data.psi.degrade_psi \
  --input-dir "${PSI_CLEAN_DIR}/L5" \
  --output-dir "${PSI_DEGRADED_DIR}/L5" \
  --noise-instrument CCOR \
  --mask-fraction 0.1 \
  --pb-mask "${PSI_MASK_DIR}/LASCO_C2_tB_daily_min.fts" \
  --tb-mask "${PSI_MASK_DIR}/LASCO_C2_tB_daily_min.fts" \
  --truth-dir "${PSI_TRUTH_DIR}/L5" \
  --truth-frame "${TRUTH_FRAME}" \
  --overwrite

# L4, 2.5--15 R_sun: STEREO-A COR2 pattern.
python -m sunerf.data.psi.degrade_psi \
  --input-dir "${PSI_CLEAN_DIR}/L4" \
  --output-dir "${PSI_DEGRADED_DIR}/L4" \
  --noise-instrument STEREO_A_COR2 \
  --mask-fraction 0.1 \
  --pb-mask "${PSI_MASK_DIR}/STEREO_A_COR2_pB_daily_min.fts" \
  --tb-mask "${PSI_MASK_DIR}/STEREO_A_COR2_tB_daily_min.fts" \
  --truth-dir "${PSI_TRUTH_DIR}/L4" \
  --truth-frame "${TRUTH_FRAME}" \
  --overwrite

# L1, 2.2--8.3 R_sun: LASCO C2 pattern.
python -m sunerf.data.psi.degrade_psi \
  --input-dir "${PSI_CLEAN_DIR}/L1" \
  --output-dir "${PSI_DEGRADED_DIR}/L1" \
  --noise-instrument LASCO_C2 \
  --mask-fraction 0.1 \
  --pb-mask "${PSI_MASK_DIR}/LASCO_C2_pB_daily_min.fts" \
  --tb-mask "${PSI_MASK_DIR}/LASCO_C2_tB_daily_min.fts" \
  --truth-dir "${PSI_TRUTH_DIR}/L1" \
  --truth-frame "${TRUTH_FRAME}" \
  --overwrite

################################################################################
# Plot three representative clean/degraded tB and pB frames for each view.
################################################################################
python -m sunerf.evaluation.plot_psi_degradation \
  --clean-dir "${PSI_CLEAN_DIR}" \
  --degraded-dir "${PSI_DEGRADED_DIR}" \
  --output-dir "${PSI_PREVIEW_DIR}" \
  --truth-dir "${PSI_TRUTH_DIR}" \
  --views L5_wide L5 L4 L1 \
  --sample-count 3
