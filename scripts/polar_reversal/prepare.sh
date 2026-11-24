#!/usr/bin/env bash
# Calibrate and prepare every raw file present: SECCHI_PREP of the EUVI level
# 0.5 files, then AIA and EUVI preparation (512x512, 1.4 R_sun, PEAK_DN) as in
# every plasma campaign. Existing calibrated files and products are skipped, so
# it runs after every download.sh. SECCHI_PREP of STEREO-A and STEREO-B runs in
# parallel (logs in logs/secchi_prep_euvi_{a,b}.log). The AIA degradation and
# pointing tables cover the whole interval and are generated on the first run.
set -euo pipefail

RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/raw"
CALIBRATED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/calibrated"
PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/prepared"
LOG_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/logs"
AIA_CALIBRATION_ROOT="data/response_calibration/instruments/aia"
AIA_CORRECTION_TABLE="${AIA_CALIBRATION_ROOT}/aia_correction.ecsv"
AIA_POINTING_TABLE="${AIA_CALIBRATION_ROOT}/aia_pointing_2011_2014.ecsv"
SSW_ROOT="/glade/work/rjarolim/ssw"
WORKERS="8"
GEOMETRY=(--shape 512 512 --max-radius-rsun 1.4)

has_files() {
  [ -n "$(find "$1" -maxdepth 1 -name "$2" -print -quit)" ]
}

mkdir -p "${CALIBRATED_ROOT}/euvi_a" "${CALIBRATED_ROOT}/euvi_b" "${AIA_CALIBRATION_ROOT}" "${LOG_ROOT}"
for directory in aia euvi_a euvi_b; do
  mkdir -p "${PREPARED_ROOT}/${directory}"
done

if [ ! -f "${AIA_CORRECTION_TABLE}" ] || [ ! -f "${AIA_POINTING_TABLE}" ]; then
  python -m sunerf.data.euv.prepare_aia_calibration_tables \
    "${AIA_CORRECTION_TABLE}" \
    "${AIA_POINTING_TABLE}" \
    "2011-02-06T00:00:00" \
    "2014-10-01T00:00:00"
fi

# Wait for both before failing, so no SECCHI_PREP keeps running behind a failed one.
pids=()
logs=()
for directory in euvi_a euvi_b; do
  if has_files "${RAW_ROOT}/${directory}" '*.fts'; then
    log="${LOG_ROOT}/secchi_prep_${directory}.log"
    csh scripts/secchi_prep.csh \
      "${RAW_ROOT}/${directory}/*.fts" \
      "${CALIBRATED_ROOT}/${directory}" \
      "${SSW_ROOT}" >> "${log}" 2>&1 &
    pids+=("$!")
    logs+=("${log}")
  fi
done
status=0
for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "${pid}" || status=1
done
for log in "${logs[@]+"${logs[@]}"}"; do
  grep -h "already calibrated" "${log}" | tail -n 1 || true
done
if [ "${status}" -ne 0 ]; then
  echo "SECCHI_PREP failed, see ${logs[*]}" >&2
  exit 1
fi

if has_files "${RAW_ROOT}/aia" '*.fits'; then
  python -m sunerf.data.euv.prepare aia \
    --input "${RAW_ROOT}/aia/*.fits" \
    --output-dir "${PREPARED_ROOT}/aia" \
    --correction-table "${AIA_CORRECTION_TABLE}" \
    --pointing-table "${AIA_POINTING_TABLE}" \
    --workers "${WORKERS}" "${GEOMETRY[@]}"
fi

for spacecraft in A B; do
  if [ "${spacecraft}" = "A" ]; then directory="euvi_a"; else directory="euvi_b"; fi
  if has_files "${CALIBRATED_ROOT}/${directory}" '*.fts'; then
    python -m sunerf.data.euv.prepare euvi \
      --input "${CALIBRATED_ROOT}/${directory}/*.fts" \
      --output-dir "${PREPARED_ROOT}/${directory}" \
      --spacecraft "${spacecraft}" \
      --product-level SECCHI-L1-calibrated \
      --sensitivity-convention static_assumed \
      --workers "${WORKERS}" "${GEOMETRY[@]}"
  fi
done
