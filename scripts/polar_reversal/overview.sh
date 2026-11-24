#!/usr/bin/env bash
# Observer overview of the prepared (and cleaned) set: positions, view
# directions and coverage per instrument. Run last, after clean.sh.
set -euo pipefail

PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/prepared"

python -m sunerf.evaluation.plot_euv_observers "${PREPARED_ROOT}" \
  --output "${PREPARED_ROOT}/observer_overview.jpg"

echo "Observer overview: ${PREPARED_ROOT}/observer_overview.jpg"
