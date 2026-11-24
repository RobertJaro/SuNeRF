#!/usr/bin/env bash
# Server: run after preparation to scan the six CME image series.
#   bash scripts/cme_2025_09/review_frames.sh
set -euo pipefail

DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
PREP_DIR="$DATA_ROOT/prep"
REVIEW_ROOT="$DATA_ROOT/review"
PUNCH_DIR="$DATA_ROOT/prep_stash/punch_pam"
WORKERS=16

python -m sunerf.data.frame_review scan "$PREP_DIR/cor2/tB" "$PREP_DIR/cor2/pB" \
  --cme --workers "$WORKERS" --title "September 2025 · COR2" --output "$REVIEW_ROOT/cor2"

python -m sunerf.data.frame_review scan "$PREP_DIR/cor2_clear" \
  --cme --workers "$WORKERS" --title "September 2025 · COR2 clear" --output "$REVIEW_ROOT/cor2_clear"

python -m sunerf.data.frame_review scan "$PUNCH_DIR/tB" "$PUNCH_DIR/pB" \
  --cme --workers "$WORKERS" --title "September 2025 · PUNCH PAM" --output "$REVIEW_ROOT/punch_pam"

python -m sunerf.data.frame_review scan "$PREP_DIR/ccor" \
  --cme --workers "$WORKERS" --title "September 2025 · CCOR" --output "$REVIEW_ROOT/ccor"

python -m sunerf.data.frame_review scan "$PREP_DIR/lasco_c2/tB" \
  --cme --workers "$WORKERS" --title "September 2025 · LASCO C2" --output "$REVIEW_ROOT/lasco_c2"

python -m sunerf.data.frame_review scan "$PREP_DIR/lasco_c3/tB" \
  --cme --workers "$WORKERS" --title "September 2025 · LASCO C3" --output "$REVIEW_ROOT/lasco_c3"

# Local: repeat these commands for cor2, cor2_clear, punch_pam, ccor, lasco_c2, lasco_c3.
#   dataset=cor2
#   local_dir="data/2025_09/review/$dataset"
#   remote_dir="casper:/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/review/$dataset"
#
# Download the review.
#   mkdir -p "$local_dir/masks"
#   rsync -av --exclude='masks/' "$remote_dir/" "$local_dir/"
#   rsync -acv --delete "$remote_dir/masks/" "$local_dir/masks/"
#
# After downloading all six datasets, run their dashboards from the repository root.
#   bash scripts/cme_2025_09/review_local.sh
#
# Upload the decisions if present. Mirroring masks/ also propagates masks cleared in the dashboard.
#   if [ -f "$local_dir/decisions.csv" ]; then rsync -acv "$local_dir/decisions.csv" "$remote_dir/"; fi
#   rsync -acv --delete "$local_dir/masks/" "$remote_dir/masks/"
#
# Server: inspect the dry run, then apply decisions and install the training masks.
# Datasets without decisions.csv use their automatic rejection flags.
#   review_root="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/review"
#   python -m sunerf.data.frame_review apply "$review_root" --dry-run
#   python -m sunerf.data.frame_review apply "$review_root"
#   bash scripts/cme_2025_09/compute_correction.sh
#   bash scripts/cme_2025_09/review_products.sh
#   bash scripts/cme_2025_09/run.sh --reload
