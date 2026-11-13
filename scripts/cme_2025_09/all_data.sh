#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Configuration
################################################################################
SCRIPT_DIR="/glade/u/home/rjarolim/projects/SuNeRF/scripts/cme_2025_09"

################################################################################
# Download all instruments
################################################################################
"${SCRIPT_DIR}/stereo_a_cor2_download.sh"
"${SCRIPT_DIR}/ccor_download.sh"
"${SCRIPT_DIR}/lasco_download.sh"
"${SCRIPT_DIR}/punch_pam_download.sh"
"${SCRIPT_DIR}/hi1_download.sh"

################################################################################
# Prepare all instruments
################################################################################
"${SCRIPT_DIR}/stereo_a_cor2_prep.sh"
"${SCRIPT_DIR}/ccor_prep.sh"
"${SCRIPT_DIR}/lasco_prep.sh"
"${SCRIPT_DIR}/punch_pam_prep.sh"

################################################################################
# Download and prepare the in-situ data
################################################################################
"${SCRIPT_DIR}/psp.sh"
"${SCRIPT_DIR}/solo.sh"
