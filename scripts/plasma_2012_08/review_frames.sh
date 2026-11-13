#!/usr/bin/env bash
# Frame review of the prepared August 2012 data set, step 1 (server).
#
# One review directory per prepared dataset (aia, euvi_a, euvi_b) with
# frames.csv (statistics and automatic flags, one row per file), JPEG previews
# and the review page; no FITS data, a few kB per frame. The peak detector DN
# of every frame comes from its prepared header (PEAKDN): the scan prints the
# saturation plateau and the recommended instruments[].saturation_dn per
# dataset, and the review page selects it (saved to saturation.csv) for
# config/plasma/all_2012_08.yaml.
#
# Step 2 (local): copy the review directories and review them; every decision
# is saved to <review dir>/decisions.csv:
#   rsync -av casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review/ data/plasma_2012_08/review/
#   python -m sunerf.data.frame_review review data/plasma_2012_08/review/aia
#   python -m sunerf.data.frame_review review data/plasma_2012_08/review/euvi_a
#   python -m sunerf.data.frame_review review data/plasma_2012_08/review/euvi_b
# Step 3 (server): copy the decisions back and move the rejected files of
# every dataset out of the training glob; a dataset without decisions.csv
# applies the automatic flags of its frames.csv:
#   rsync -av --include='*/' --include='decisions.csv' --exclude='*' data/plasma_2012_08/review/ casper:/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review/
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review --dry-run
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review
# Undo with: python -m sunerf.data.frame_review restore <prepared dir>
#
# run.sh reuses an existing image-scaling table; after the apply step
# re-estimate it without the rejected frames:
#   python -m sunerf.data.euv.estimate_scaling --config config/plasma/all_2012_08.yaml --workers 16 --overwrite
set -euo pipefail

PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/prepared"
REVIEW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/review"
WORKERS="16"

python -m sunerf.data.frame_review scan \
  "${PREPARED_ROOT}/aia/*.prepared.fits" \
  --output "${REVIEW_ROOT}/aia" \
  --title "Plasma 2012-08: aia" \
  --workers "${WORKERS}"

python -m sunerf.data.frame_review scan \
  "${PREPARED_ROOT}/euvi_a/*.prepared.fits" \
  --output "${REVIEW_ROOT}/euvi_a" \
  --title "Plasma 2012-08: euvi_a" \
  --workers "${WORKERS}"

python -m sunerf.data.frame_review scan \
  "${PREPARED_ROOT}/euvi_b/*.prepared.fits" \
  --output "${REVIEW_ROOT}/euvi_b" \
  --title "Plasma 2012-08: euvi_b" \
  --workers "${WORKERS}"

# Report only: what the automatic flags would move before any review.
python -m sunerf.data.frame_review apply "${REVIEW_ROOT}" --dry-run

echo "Review pages under ${REVIEW_ROOT}/<dataset>/index.html"
