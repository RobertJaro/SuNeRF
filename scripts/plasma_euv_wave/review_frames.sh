#!/usr/bin/env bash
# Frame review of the prepared EUV wave 2022-03 data set, step 1 (server).
#
# One review directory per prepared dataset with frames.csv (statistics and
# automatic flags, one row per file), JPEG previews and the review page; no
# FITS data, a few kB per frame. The peak detector DN of every frame comes
# from its prepared header (PEAKDN): the scan prints the saturation plateau and
# the recommended instruments[].saturation_dn per dataset, and the review page
# selects it (saved to saturation.csv) for the plasma config.
#
# Step 2 (local): copy the review directories and review them; every decision
# is saved to <review dir>/decisions.csv:
#   rsync -av casper:/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/review .
#   python -m sunerf.data.frame_review review review/aia_focus
# Step 3 (server): copy the decisions back and move the rejected files of
# every dataset out of the training glob; a dataset without decisions.csv
# applies the automatic flags of its frames.csv:
#   rsync -av --include='*/' --include='decisions.csv' --exclude='*' review/ casper:/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/review/
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/review --dry-run
#   python -m sunerf.data.frame_review apply /glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/review
# Undo with: python -m sunerf.data.frame_review restore <prepared dir>
set -euo pipefail

PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/prepared"
REVIEW_ROOT="/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/review"
WORKERS="16"

for dataset in aia aia_focus euvi_a euvi_a_focus eui eui_focus; do
  python -m sunerf.data.frame_review scan \
    "${PREPARED_ROOT}/${dataset}/*.prepared.fits" \
    --output "${REVIEW_ROOT}/${dataset}" \
    --title "EUV wave 2022-03: ${dataset}" \
    --workers "${WORKERS}"
done

python -m sunerf.data.frame_review review data/euv_wave/review/aia
python -m sunerf.data.frame_review review data/euv_wave/review/aia_focus
python -m sunerf.data.frame_review review data/euv_wave/review/euvi_a
python -m sunerf.data.frame_review review data/euv_wave/review/euvi_a_focus
python -m sunerf.data.frame_review review data/euv_wave/review/eui
python -m sunerf.data.frame_review review data/euv_wave/review/eui_focus


python -m sunerf.data.frame_review apply "${REVIEW_ROOT}" --dry-run

echo "Review pages under ${REVIEW_ROOT}/<dataset>/index.html"
