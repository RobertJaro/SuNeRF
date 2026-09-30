#!/usr/bin/env bash
DRMS_EMAIL="robert.jarolim@uni-graz.at"

RAW_ROOT="/glade/work/rjarolim/data/sunerf/2012_08/raw"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2012_08.ecsv"

mkdir -p "${RAW_ROOT}/aia" "${RAW_ROOT}/euvi_a" "${RAW_ROOT}/euvi_b" "${AIA_CALIBRATION_ROOT}"

################################################################################
# AIA degradation and master-pointing tables used during preparation
################################################################################
python -m sunerf.data.euv.prepare_aia_calibration_tables \
  "${AIA_CORRECTION_TABLE}" \
  "${AIA_POINTING_TABLE}" \
  "2012-08-01T00:00:00" \
  "2012-09-01T00:00:00"

################################################################################
# SDO/AIA level-1 EUV channels
################################################################################
python -m sunerf.data.download.download_aia \
  --output "${RAW_ROOT}/aia" \
  --email "${DRMS_EMAIL}" \
  --start 2012-08-01T00:00:00 \
  --end 2012-09-01T00:00:00 \
  --cadence 6h \
  --channels 94 131 171 193 211 335

################################################################################
# STEREO/SECCHI EUVI level-0/1 inputs, kept separate by spacecraft
################################################################################
python -m sunerf.data.download.download_euvi \
  --output "${RAW_ROOT}/euvi_a" \
  --start 2012-08-01T00:00:00 \
  --end 2012-09-01T00:00:00 \
  --cadence 6h \
  --channels 171 195 284 \
  --sources STEREO_A

python -m sunerf.data.download.download_euvi \
  --output "${RAW_ROOT}/euvi_b" \
  --start 2012-08-01T00:00:00 \
  --end 2012-09-01T00:00:00 \
  --cadence 6h \
  --channels 171 195 284 \
  --sources STEREO_B

echo "Downloaded raw EUV observations under ${RAW_ROOT}"
