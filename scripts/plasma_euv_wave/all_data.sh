#!/usr/bin/env bash
################################################################################
# Download observations, then prepare every instrument
################################################################################
set -euo pipefail
scripts/plasma_euv_wave/download.sh
scripts/plasma_euv_wave/prepare.sh
