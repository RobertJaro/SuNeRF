#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Configuration
################################################################################
SCRIPT_DIR="/glade/u/home/rjarolim/projects/SuNeRF/scripts/cme_2010_03"

################################################################################
# Download and prepare all instruments
################################################################################
"${SCRIPT_DIR}/lasco_c2.sh"
"${SCRIPT_DIR}/stereo_a_cor2.sh"
"${SCRIPT_DIR}/stereo_b_cor2.sh"
