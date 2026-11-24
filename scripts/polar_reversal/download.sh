#!/usr/bin/env bash
# Download AIA level 1 (JSOC) and EUVI-A/B level 0.5 (VSO) at 2 h for one
# interval, e.g. one quarter:
#   scripts/polar_reversal/download.sh 2011-02-06T00:00:00 2011-04-01T00:00:00
#
# Every record at the 2 h slots is downloaded; flagged AIA frames (eclipse,
# dark, calibration) are skipped by prepare.sh and leave a gap. Weeks without
# EUVI records are listed at the end of the EUVI logs. Records with a raw file
# or a prepared product (also one in rejected/) are skipped, so a rerun resumes
# and files deleted by clear.sh are not downloaded again; an AIA chunk whose
# records are all present is not exported.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 START END" >&2
  exit 1
fi
START="$1"
END="$2"

DRMS_EMAIL="robert.jarolim@uni-graz.at"
RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/raw"
PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/prepared"
LOG_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/logs"
LABEL="${START:0:10}_${END:0:10}"

mkdir -p "${RAW_ROOT}/aia" "${RAW_ROOT}/euvi_a" "${RAW_ROOT}/euvi_b" "${LOG_ROOT}"

python -u -m sunerf.data.download.download_aia \
  --output "${RAW_ROOT}/aia" \
  --prepared-dir "${PREPARED_ROOT}/aia" \
  --email "${DRMS_EMAIL}" \
  --start "${START}" --end "${END}" \
  --cadence 2h --chunk 7d \
  --channels 94 131 171 193 211 335 \
  > "${LOG_ROOT}/${LABEL}_download_aia.log" 2>&1 &
aia_pid=$!
python -u -m sunerf.data.download.download_euvi \
  --output "${RAW_ROOT}/euvi_a" \
  --prepared-dir "${PREPARED_ROOT}/euvi_a" \
  --start "${START}" --end "${END}" \
  --cadence 2h --chunk 7d \
  --channels 171 195 284 \
  --sources STEREO_A \
  > "${LOG_ROOT}/${LABEL}_download_euvi_a.log" 2>&1 &
euvi_a_pid=$!
python -u -m sunerf.data.download.download_euvi \
  --output "${RAW_ROOT}/euvi_b" \
  --prepared-dir "${PREPARED_ROOT}/euvi_b" \
  --start "${START}" --end "${END}" \
  --cadence 2h --chunk 7d \
  --channels 171 195 284 \
  --sources STEREO_B \
  > "${LOG_ROOT}/${LABEL}_download_euvi_b.log" 2>&1 &
euvi_b_pid=$!

# Wait for all three before failing, so no download keeps running behind a
# failed one.
status=0
wait "${aia_pid}" || status=1
wait "${euvi_a_pid}" || status=1
wait "${euvi_b_pid}" || status=1

tail -n 20 "${LOG_ROOT}/${LABEL}_download_aia.log"
tail -n 5 "${LOG_ROOT}/${LABEL}_download_euvi_a.log" "${LOG_ROOT}/${LABEL}_download_euvi_b.log"
exit "${status}"
