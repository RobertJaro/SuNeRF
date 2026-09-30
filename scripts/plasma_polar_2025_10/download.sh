#!/usr/bin/env bash
# Solar Orbiter polar validation, north pass October 2025.
#
# Training views: SDO/AIA (Earth) and STEREO-A/EUVI (+48 deg from Earth), 1 h
# cadence over the high-latitude interval plus one week of margin on each side.
# Held-out view: Solar Orbiter/EUI FSI 174 while the spacecraft is above +15 deg
# heliographic latitude (2025-10-04 .. 2025-10-28; peak +16.8 deg on 10-14 at
# 0.59 au, 8 deg east of the Earth-Sun line). FSI 304 is chromospheric and the
# plasma model carries no cool component, so it is not downloaded.
set -euo pipefail

DRMS_EMAIL="robert.jarolim@uni-graz.at"
RAW_ROOT="/glade/work/rjarolim/data/sunerf/polar_2025_10/raw"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2025_10.ecsv"

TRAIN_START="2025-09-27T00:00:00"
TRAIN_END="2025-11-05T00:00:00"
TRAIN_CADENCE="1h"
AIA_CHUNK="1d"
EUI_START="2025-10-04T00:00:00"
EUI_END="2025-10-29T00:00:00"
EUI_CADENCE="6h"

mkdir -p "${RAW_ROOT}/aia" "${RAW_ROOT}/euvi_a" "${RAW_ROOT}/eui" "${AIA_CALIBRATION_ROOT}"

################################################################################
# AIA degradation and master-pointing tables used during preparation
################################################################################
python -m sunerf.data.euv.prepare_aia_calibration_tables \
  "${AIA_CORRECTION_TABLE}" \
  "${AIA_POINTING_TABLE}" \
  "${TRAIN_START}" \
  "${TRAIN_END}"

################################################################################
# SDO/AIA level-1 EUV channels
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
