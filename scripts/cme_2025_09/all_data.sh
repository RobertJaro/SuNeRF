#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Configuration
################################################################################
SCRIPT_DIR="/glade/u/home/rjarolim/projects/SuNeRF/scripts/cme_2025_09"

################################################################################
# Download and prepare all instruments
################################################################################
"${SCRIPT_DIR}/stereo_a_cor2.sh"
"${SCRIPT_DIR}/ccor.sh"
"${SCRIPT_DIR}/punch_pam.sh"
"${SCRIPT_DIR}/psp.sh"
"${SCRIPT_DIR}/solo.sh"
