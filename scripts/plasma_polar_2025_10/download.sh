#!/usr/bin/env bash
# Solar Orbiter polar validation, north pass October 2025.
#
# Training views: SDO/AIA (Earth) and STEREO-A/EUVI (+48 deg from Earth).
# Held-out view: Solar Orbiter/EUI FSI 174 while the spacecraft is above +15 deg
# heliographic latitude (2025-10-04 .. 2025-10-28). FSI 304 is chromospheric and
# the plasma model carries no cool component, so it is not downloaded.
#
# The training views are sampled at 1 h over the high-latitude interval plus
# one week of margin on each side; FSI is sampled at 6 h. The FSI window is
# symmetric about the Solar Orbiter latitude maximum (+16.84 deg on
# 2025-10-14T09:00, 8.3 deg east of Earth, 0.59 au; JPL Horizons, queried
# 2026-10-14), the validation time (valid_datasets[].time) of the run. SOAR
# holds no FSI 174 frames on 2025-10-04.
set -euo pipefail

DRMS_EMAIL="robert.jarolim@uni-graz.at"
RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/raw"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2025_10.ecsv"

TRAIN_START="2025-09-27T00:00:00"
TRAIN_END="2025-11-05T00:00:00"
TRAIN_CADENCE="1h"
AIA_CHUNK="1d"
EUI_START="2025-10-05T00:00:00"
EUI_END="2025-10-23T18:00:00"
EUI_CADENCE="6h"

mkdir -p "${RAW_ROOT}/aia" "${RAW_ROOT}/euvi_a" "${RAW_ROOT}/eui"
mkdir -p "${AIA_CALIBRATION_ROOT}"

################################################################################
# AIA degradation and master-pointing tables used during preparation
################################################################################
python -m sunerf.data.euv.prepare_aia_calibration_tables \
  "${AIA_CORRECTION_TABLE}" \
  "${AIA_POINTING_TABLE}" \
  "${TRAIN_START}" \
  "${TRAIN_END}"

################################################################################
# SDO/AIA level-1 EUV channels (one JSOC export per chunk and all channels)
################################################################################
python -m sunerf.data.download.download_aia \
  --output "${RAW_ROOT}/aia" \
  --email "${DRMS_EMAIL}" \
  --start "${TRAIN_START}" \
  --end "${TRAIN_END}" \
  --cadence "${TRAIN_CADENCE}" \
  --chunk "${AIA_CHUNK}" \
  --channels 94 131 171 193 211 335

################################################################################
# STEREO-A/SECCHI EUVI level-0 inputs (queried one week at a time)
################################################################################
python -m sunerf.data.download.download_euvi \
  --output "${RAW_ROOT}/euvi_a" \
  --start "${TRAIN_START}" \
  --end "${TRAIN_END}" \
  --cadence "${TRAIN_CADENCE}" \
  --channels 171 195 284 \
  --sources STEREO_A

################################################################################
# Solar Orbiter/EUI FSI 174 level-2, held-out validation view
################################################################################
python -m sunerf.data.download.download_eui \
  --output "${RAW_ROOT}/eui" \
  --start "${EUI_START}" \
  --end "${EUI_END}" \
  --cadence "${EUI_CADENCE}" \
  --channels 174

echo "Downloaded raw EUV observations under ${RAW_ROOT}"
