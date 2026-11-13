#!/usr/bin/env bash

RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/raw"
CALIBRATED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/calibrated"
PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2012_08/prepared"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2012_08.ecsv"
SSW_ROOT="/glade/work/rjarolim/ssw"

mkdir -p "${CALIBRATED_ROOT}/euvi_a" "${CALIBRATED_ROOT}/euvi_b"
mkdir -p "${PREPARED_ROOT}/aia" "${PREPARED_ROOT}/euvi_a" "${PREPARED_ROOT}/euvi_b"

################################################################################
# AIA degradation and master-pointing tables (no-op when download.sh made them)
################################################################################
python -m sunerf.data.euv.prepare_aia_calibration_tables \
  "${AIA_CORRECTION_TABLE}" \
  "${AIA_POINTING_TABLE}" \
  "2012-08-01T00:00:00" \
  "2012-09-01T00:00:00"

################################################################################
# External SECCHI radiometric calibration
################################################################################
csh scripts/secchi_prep.csh \
  "${RAW_ROOT}/euvi_a/*" \
  "${CALIBRATED_ROOT}/euvi_a" \
  "${SSW_ROOT}"

csh scripts/secchi_prep.csh \
  "${RAW_ROOT}/euvi_b/*" \
  "${CALIBRATED_ROOT}/euvi_b" \
  "${SSW_ROOT}"

################################################################################
# Prepare calibrated FITS products
################################################################################
# Same 1.4 R_sun field of view as every plasma campaign: it sits inside the
# 1.7 R_sun sampling shell and matches model.supervised_radius_rsun of
# config/plasma/all_2012_08.yaml. A dataset can narrow it further at load time
# (data.*_datasets[].max_radius_rsun).
python -m sunerf.data.euv.prepare aia \
  --input "${RAW_ROOT}/aia/*" \
  --output-dir "${PREPARED_ROOT}/aia" \
  --correction-table "${AIA_CORRECTION_TABLE}" \
  --pointing-table "${AIA_POINTING_TABLE}" \
  --workers 16 \
  --shape 512 512 \
  --max-radius-rsun 1.4 \
  --overwrite

python -m sunerf.data.euv.prepare euvi \
  --input "${CALIBRATED_ROOT}/euvi_a/*" \
  --output-dir "${PREPARED_ROOT}/euvi_a" \
  --spacecraft A \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --workers 16 \
  --shape 512 512 \
  --max-radius-rsun 1.4 \
  --overwrite

python -m sunerf.data.euv.prepare euvi \
  --input "${CALIBRATED_ROOT}/euvi_b/*" \
  --output-dir "${PREPARED_ROOT}/euvi_b" \
  --spacecraft B \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --workers 16 \
  --shape 512 512 \
  --max-radius-rsun 1.4 \
  --overwrite

echo "Prepared FITS files under ${PREPARED_ROOT}"
