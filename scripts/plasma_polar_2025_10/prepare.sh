#!/usr/bin/env bash
# Radiometric calibration and common geometry for the polar 2025-10 dataset.
#
# AIA is calibrated here with pinned aiapy tables; EUVI is calibrated
# externally by SolarSoft (secchi_prep.csh) and EUI arrives as level 2, so both
# are only validated and regridded. The EUI calibration id comes from the
# level-2 header.
set -euo pipefail

RAW_ROOT="/glade/work/rjarolim/data/sunerf/polar_2025_10/raw"
CALIBRATED_ROOT="/glade/work/rjarolim/data/sunerf/polar_2025_10/calibrated"
PREPARED_ROOT="/glade/work/rjarolim/data/sunerf/polar_2025_10_prepared"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2025_10.ecsv"
SSW_ROOT="/glade/work/rjarolim/ssw"
WORKERS="32"

# Same field of view in solar radii for every instrument, on one 512x512 grid.
# The angular half-width follows from each image's own apparent solar radius
# (DSUN_OBS, RSUN_REF), so the closer Solar Orbiter view covers the same physical
# extent as AIA and EUVI instead of the same angular extent. Pixels outside
# 1.4 R_sun, and outside the detector, are NaN with VALID_MASK=0.
#
# 1.4 R_sun sits inside the 1.7 R_sun sampling shell of
# config/plasma/polar_2025_10.yaml, which keeps a full integration chord under
# every supervised pixel; a field of view reaching the shell edge would leave
# those rays with a sliver of path length. It must not grow past the shell
# either: rays that miss it are rendered and only then dropped from the loss.
# AIA reaches 1.28 R_sun on its axes (1229" at 962"/R_sun), so its outer rim is
# NaN there and only EUVI-A and FSI constrain 1.28 .. 1.4 R_sun.
GEOMETRY=(--shape 512 512 --max-radius-rsun 1.4)

mkdir -p "${CALIBRATED_ROOT}/euvi_a"
mkdir -p "${PREPARED_ROOT}/aia" "${PREPARED_ROOT}/euvi_a" "${PREPARED_ROOT}/eui"

################################################################################
# External SECCHI radiometric calibration
################################################################################
csh scripts/plasma_polar_2025_10/secchi_prep.csh \
  "${RAW_ROOT}/euvi_a/*" \
  "${CALIBRATED_ROOT}/euvi_a" \
  "${SSW_ROOT}"

################################################################################
# Prepare calibrated FITS products
################################################################################
python -m sunerf.data.euv.prepare aia \
  --input "${RAW_ROOT}/aia/*" \
  --output-dir "${PREPARED_ROOT}/aia" \
  --correction-table "${AIA_CORRECTION_TABLE}" \
  --pointing-table "${AIA_POINTING_TABLE}" \
  --workers "${WORKERS}" "${GEOMETRY[@]}"

python -m sunerf.data.euv.prepare euvi \
  --input "${CALIBRATED_ROOT}/euvi_a/*" \
  --output-dir "${PREPARED_ROOT}/euvi_a" \
  --spacecraft A \
  --product-level SECCHI-L1-calibrated \
  --sensitivity-convention static_assumed \
  --workers "${WORKERS}" "${GEOMETRY[@]}"

python -m sunerf.data.euv.prepare eui \
  --input "${RAW_ROOT}/eui/*" \
  --output-dir "${PREPARED_ROOT}/eui" \
  --sensitivity-convention static_assumed \
  --workers "${WORKERS}" "${GEOMETRY[@]}"

################################################################################
# Observer overview of the prepared set: positions, view directions, coverage
################################################################################
python -m sunerf.evaluation.plot_euv_observers "${PREPARED_ROOT}" \
  --output "${PREPARED_ROOT}/observer_overview.jpg"

echo "Prepared FITS files under ${PREPARED_ROOT}"
