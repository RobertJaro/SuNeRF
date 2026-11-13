#!/usr/bin/env bash
# EUV wave of the M4.0 flare of 2022-03-28 (AR 12975, wave interval 11:05–11:45 UT).
#
# Training views: SDO/AIA (Earth), STEREO-A/EUVI (~33 deg east of Earth),
# and Solar Orbiter/EUI FSI 174 (~83.5 deg west, 0.325 au).
# FSI 304 is chromospheric and excluded from this plasma model.
#
# Two sampling tiers in separate directories (<instrument>, <instrument>_focus):
#   background: 1 h, from 7 days before the focus start to 7 days after its end;
#               includes the focus interval.
#   focus: 2022-03-28 10:30 .. 12:30 UT, requested cadence 1 min.
#          AIA uses all six channels; EUVI 171/195/284 sets are anchored on 284;
#          EUVI and FSI sampling is limited by the available native cadence.
set -euo pipefail

DRMS_EMAIL="robert.jarolim@uni-graz.at"
RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/EUV_WAVE/data/raw"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2022_03.ecsv"

# Background tier
TRAIN_START="2022-03-21T10:30:00"
TRAIN_END="2022-04-04T12:30:00"
TRAIN_CADENCE="1h"
AIA_CHUNK="1d"

# Focus tier
FOCUS_START="2022-03-28T10:30:00"
FOCUS_END="2022-03-28T12:30:00"
FOCUS_CADENCE="1m"
FOCUS_AIA_CHUNK="1h"

mkdir -p "${RAW_ROOT}/aia" "${RAW_ROOT}/aia_focus"
mkdir -p "${RAW_ROOT}/euvi_a" "${RAW_ROOT}/euvi_a_focus"
mkdir -p "${RAW_ROOT}/eui" "${RAW_ROOT}/eui_focus"
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
download_aia() {
  local output="$1" start="$2" end="$3" cadence="$4" chunk="$5"
  python -m sunerf.data.download.download_aia \
    --output "${output}" \
    --email "${DRMS_EMAIL}" \
    --start "${start}" \
    --end "${end}" \
    --cadence "${cadence}" \
    --chunk "${chunk}" \
    --channels 94 131 171 193 211 335
}

download_aia "${RAW_ROOT}/aia" "${TRAIN_START}" "${TRAIN_END}" "${TRAIN_CADENCE}" "${AIA_CHUNK}"
download_aia "${RAW_ROOT}/aia_focus" "${FOCUS_START}" "${FOCUS_END}" "${FOCUS_CADENCE}" "${FOCUS_AIA_CHUNK}"

################################################################################
# STEREO-A/SECCHI EUVI level-0 inputs (queried one week at a time); complete
# 171/195/284 sets anchored on 284 in both tiers
################################################################################
download_euvi() {
  local output="$1" start="$2" end="$3" cadence="$4"
  python -m sunerf.data.download.download_euvi \
    --output "${output}" \
    --start "${start}" \
    --end "${end}" \
    --cadence "${cadence}" \
    --channels 171 195 284 \
    --sources STEREO_A
}

download_euvi "${RAW_ROOT}/euvi_a" "${TRAIN_START}" "${TRAIN_END}" "${TRAIN_CADENCE}"
download_euvi "${RAW_ROOT}/euvi_a_focus" "${FOCUS_START}" "${FOCUS_END}" "${FOCUS_CADENCE}"

################################################################################
# Solar Orbiter/EUI FSI 174 level-2, near-quadrature training view
################################################################################
download_eui() {
  local output="$1" start="$2" end="$3" cadence="$4"
  python -m sunerf.data.download.download_eui \
    --output "${output}" \
    --start "${start}" \
    --end "${end}" \
    --cadence "${cadence}" \
    --channels 174
}

download_eui "${RAW_ROOT}/eui" "${TRAIN_START}" "${TRAIN_END}" "${TRAIN_CADENCE}"
download_eui "${RAW_ROOT}/eui_focus" "${FOCUS_START}" "${FOCUS_END}" "${FOCUS_CADENCE}"

echo "Downloaded raw EUV observations under ${RAW_ROOT}"
