#!/usr/bin/env bash

RAW_ROOT="/glade/work/rjarolim/data/sunerf/2012_08/raw"
CALIBRATED_ROOT="/glade/work/rjarolim/data/sunerf/2012_08/calibrated"
PREPARED_ROOT="/glade/work/rjarolim/data/sunerf/2012_08_prepared_v2"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing.ecsv"
SSW_ROOT="/glade/u/home/rjarolim/ssw"

mkdir -p "${CALIBRATED_ROOT}/euvi_a" "${CALIBRATED_ROOT}/euvi_b"
mkdir -p "${PREPARED_ROOT}/aia" "${PREPARED_ROOT}/euvi_a" "${PREPARED_ROOT}/euvi_b"

################################################################################
# External SECCHI radiometric calibration
################################################################################
csh scripts/plasma_2012_08/secchi_prep.csh \
  "${RAW_ROOT}/euvi_a/*" \
  "${RAW_ROOT}/euvi_b/*" \
  "${CALIBRATED_ROOT}/euvi_a" \
  "${CALIBRATED_ROOT}/euvi_b" \
  "${SSW_ROOT}"

################################################################################
# Prepare calibrated FITS products
################################################################################
python -m sunerf.data.euv.prepare aia \
  --input "${RAW_ROOT}/aia/*" \
  --output-dir "${PREPARED_ROOT}/aia" \
  --correction-table "${AIA_CORRECTION_TABLE}" \
  --pointing-table "${AIA_POINTING_TABLE}" \
  --workers 16 \
  --shape 512 512 \
  --hpc-bounds -1560 -1560 1560 1560 \
  --overwrite

python -m sunerf.data.euv.prepare euvi \
  --input "${CALIBRATED_ROOT}/euvi_a/*" \
  --output-dir "${PREPARED_ROOT}/euvi_a" \
  --spacecraft A \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --workers 16 \
  --shape 512 512 \
  --hpc-bounds -1560 -1560 1560 1560 \
  --overwrite

python -m sunerf.data.euv.prepare euvi \
  --input "${CALIBRATED_ROOT}/euvi_b/*" \
  --output-dir "${PREPARED_ROOT}/euvi_b" \
  --spacecraft B \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --workers 16 \
  --shape 512 512 \
  --hpc-bounds -1560 -1560 1560 1560 \
  --overwrite

echo "Prepared FITS files under ${PREPARED_ROOT}"
