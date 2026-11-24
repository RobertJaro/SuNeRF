#!/usr/bin/env bash
# Delete the raw and calibrated files that have a prepared product (also one
# moved to rejected/ by clean.sh). Files that failed to prepare stay for
# inspection. Run after prepare.sh; a dry run lists what would be removed:
#   python -m sunerf.data.euv.clear_sources "<raw dir>/aia/*.fits" --prepared-dir <prepared dir>/aia --dry-run
set -euo pipefail

RAW_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/raw"
CALIBRATED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/calibrated"
PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/prepared"

python -m sunerf.data.euv.clear_sources "${RAW_ROOT}/aia/*.fits" \
  --prepared-dir "${PREPARED_ROOT}/aia"
for directory in euvi_a euvi_b; do
  python -m sunerf.data.euv.clear_sources "${RAW_ROOT}/${directory}/*.fts" \
    --prepared-dir "${PREPARED_ROOT}/${directory}" --secchi-level-0
  python -m sunerf.data.euv.clear_sources "${CALIBRATED_ROOT}/${directory}/*.fts" \
    --prepared-dir "${PREPARED_ROOT}/${directory}"
done
