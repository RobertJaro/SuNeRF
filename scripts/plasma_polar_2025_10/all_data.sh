#!/usr/bin/env bash
################################################################################
# Download observations, then prepare every instrument
################################################################################
set -euo pipefail
scripts/plasma_polar_2025_10/download.sh
scripts/plasma_polar_2025_10/prepare.sh
