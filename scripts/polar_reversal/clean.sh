#!/usr/bin/env bash
# Automatic frame cleaning of the prepared polar reversal data set. Run once
# after the last quarter is prepared (the jump flags need the full series).
#
# One review directory per instrument and year (frames.csv with statistics and
# automatic flags, JPEG previews, review page). The automatic rejections (header
# QUALITY, low valid-pixel fraction, level and structure jumps against the
# neighboring frames of the same channel) are then moved to rejected/ next to
# the prepared files, so the training globs no longer match them. Saturation
# never rejects a frame: run_plasma masks the pixels above
# instruments[].saturation_dn; every scan prints the recommended level.
#
# A dataset reviewed by hand (decisions.csv, see scripts/plasma_polar_2025_10/
# review_frames.sh) uses the decisions instead of the automatic flags. Undo with
#   python -m sunerf.data.frame_review restore <prepared dir>
set -euo pipefail

PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/prepared"
REVIEW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/review"
WORKERS="32"

for year in 2011 2012 2013 2014; do
  python -m sunerf.data.frame_review scan \
    "${PREPARED_ROOT}/aia/aia.lev1_euv_12s.${year}-*.prepared.fits" \
    --output "${REVIEW_ROOT}/aia_${year}" \
    --title "Polar reversal: AIA ${year}" \
    --workers "${WORKERS}"
  for dataset in euvi_a euvi_b; do
    python -m sunerf.data.frame_review scan \
      "${PREPARED_ROOT}/${dataset}/${year}*.prepared.fits" \
      --output "${REVIEW_ROOT}/${dataset}_${year}" \
      --title "Polar reversal: ${dataset} ${year}" \
      --workers "${WORKERS}"
  done
done

python -m sunerf.data.frame_review apply "${REVIEW_ROOT}" --dry-run
python -m sunerf.data.frame_review apply "${REVIEW_ROOT}"

echo "Review pages under ${REVIEW_ROOT}/<dataset>_<year>/index.html"
