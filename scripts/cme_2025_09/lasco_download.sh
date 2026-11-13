#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09"
START="2025-09-01T00:00:00"
END="2025-10-01T00:00:00"
WORKERS="16"
SSW_ROOT="/glade/work/rjarolim/ssw"

LASCO_RAW_DIR="${DATA_ROOT}/lasco"
ANCIL_DIR="${DATA_ROOT}/soho_ancillary"
EXPFAC_DIR="${SSW_ROOT}/soho/lasco/idl/expfac/data/2509"

################################################################################
# Download LASCO C2/C3 level-0.5 clear images
################################################################################
# NRL level_1 ends in 2017; level_05 is calibrated with REDUCE_LEVEL_1 in lasco_prep.sh.
python -m sunerf.data.download.download_lasco \
  --start "${START}" \
  --end "${END}" \
  --output "${LASCO_RAW_DIR}" \
  --instrument C2 C3 \
  --product level_05 \
  --workers "${WORKERS}"

################################################################################
# Download SOHO roll attitude
################################################################################
# Used by REDUCE_LEVEL_1 for the roll after the last NRL pointing-table day.
mkdir -p "${ANCIL_DIR}/attitude/roll" "${ANCIL_DIR}/attitude/predictive/2025"

wget -q -N -P "${ANCIL_DIR}/attitude/roll" \
  "https://soho.nascom.nasa.gov/data/ancillary/attitude/roll/nominal_roll_attitude.dat"

wget -q -N -r -np -nd -A "SO_AT_ROL_202509*.DAT" -P "${ANCIL_DIR}/attitude/predictive/2025" \
  "https://soho.nascom.nasa.gov/data/ancillary/attitude/predictive/2025/"

################################################################################
# Update LASCO exposure factors in SolarSoft
################################################################################
# NRL updates the expfac files after the fact; without them REDUCE_LEVEL_1 uses
# an exposure factor of 1.
mkdir -p "${EXPFAC_DIR}"
wget -q -N -r -np -nd -A "c2_expfactor_2509*.dat,c3_expfactor_2509*.dat" -P "${EXPFAC_DIR}" \
  "https://soho.nascom.nasa.gov/solarsoft/soho/lasco/idl/expfac/data/2509/"
