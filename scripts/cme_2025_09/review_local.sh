#!/usr/bin/env bash
# Run from the repository root. Press Ctrl-C after saving each review to open the next dataset.
set -eu

python -m sunerf.data.frame_review review data/2025_09/review/cor2
python -m sunerf.data.frame_review review data/2025_09/review/cor2_clear
python -m sunerf.data.frame_review review data/2025_09/review/punch_pam
python -m sunerf.data.frame_review review data/2025_09/review/ccor
python -m sunerf.data.frame_review review data/2025_09/review/lasco_c2
python -m sunerf.data.frame_review review data/2025_09/review/lasco_c3
