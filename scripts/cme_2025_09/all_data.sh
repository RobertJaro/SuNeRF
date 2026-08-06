#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Configuration
################################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

################################################################################
# Download and prepare all instruments
################################################################################
"${SCRIPT_DIR}/stereo_a_cor2.sh"
"${SCRIPT_DIR}/ccor.sh"
"${SCRIPT_DIR}/punch_pam.sh"
"${SCRIPT_DIR}/psp.sh"
"${SCRIPT_DIR}/solo.sh"
