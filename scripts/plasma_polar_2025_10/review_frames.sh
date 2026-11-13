#!/usr/bin/env bash
# Frame review of the prepared polar 2025-10 data set, step 1 (server).
#
# One review directory per prepared dataset (aia, euvi_a, eui) with frames.csv (statistics and automatic flags, one row
# per file), JPEG previews and the review page; no FITS data, a few kB per
# frame. The peak detector DN of every frame comes from its prepared header
# (PEAKDN): the scan prints the saturation plateau and the recommended
# instruments[].saturation_dn per dataset, and the review page selects it
# (saved to saturation.csv) for config/plasma/polar_2025_10.yaml. The EUI
# sequence is never trained on; a rejected FSI frame leaves the validation set.
#
# Step 2 (local): copy the review directories and review them; every decision
# is saved to <review dir>/decisions.csv:
#   rsync -av casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review/ data/plasma_polar_2025_10/review/
#   python -m sunerf.data.frame_review review data/plasma_polar_2025_10/review/aia
#   python -m sunerf.data.frame_review review data/plasma_polar_2025_10/review/euvi_a
#   python -m sunerf.data.frame_review review data/plasma_polar_2025_10/review/eui
# Step 3 (server): copy the decisions back and move the rejected files of
# every dataset out of the training glob; a dataset without decisions.csv
# applies the automatic flags of its frames.csv:
#   rsync -av --include='*/' --include='decisions.csv' --exclude='*' data/plasma_polar_2025_10/review/ casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review/
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review --dry-run
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review
# Undo with: python -m sunerf.data.frame_review restore <prepared dir>
set -euo pipefail

PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/prepared"
REVIEW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/review"
WORKERS="32"

for dataset in aia euvi_a eui; do
  python -m sunerf.data.frame_review scan \
    "${PREPARED_ROOT}/${dataset}/*.prepared.fits" \
    --output "${REVIEW_ROOT}/${dataset}" \
    --title "Polar 2025-10: ${dataset}" \
    --workers "${WORKERS}"
done

# Report only: what the automatic flags would move before any review.
python -m sunerf.data.frame_review apply "${REVIEW_ROOT}" --dry-run

echo "Review pages under ${REVIEW_ROOT}/<dataset>/index.html"
