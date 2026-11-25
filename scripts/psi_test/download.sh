#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
CORHEL_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs_corhel"

# Coronal cubes and their five-minute observation times.
python -m sunerf.data.psi.download_psi_density \
  --url "https://www.predsci.com/~epalmerio/issi_rho3d/20211028/cor/" \
  --out-dir "${DATA_ROOT}/rho"

python -m sunerf.data.psi.dump_times \
  --url "https://www.predsci.com/~epalmerio/getpb/20211028/fakeC3/fits_L1/pb/" \
  --out-file "${DATA_ROOT}/dump_times.json"

# CORHEL cubes and their roughly hourly heliospheric observation times.
python -m sunerf.data.psi.download_psi_density \
  --url "https://www.predsci.com/~epalmerio/issi_rho3d/20211028/corhel/" \
  --out-dir "${CORHEL_ROOT}/rho"

python -m sunerf.data.psi.dump_times \
  --url "https://www.predsci.com/~epalmerio/getpb/20211028/fakeHI/fits_L1e/pb/" \
  --out-file "${CORHEL_ROOT}/dump_times.json"
