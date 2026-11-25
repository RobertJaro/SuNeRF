#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
CORHEL_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs_corhel"
RENDER_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs_render"

render() {
  local view="$1"
  local out_dir="${RENDER_ROOT}/clean/${view}"
  shift
  python -m sunerf.data.psi.render_psi_thomson \
    --key "${view}" --out-dir "${out_dir}" --resolution 512 \
    --pre-duration 20d --pre-cadence 6h "$@"
  python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
    "${out_dir}/jpg" --tb "${out_dir}/tb/*" --pb "${out_dir}/pb/*" --a 1e-6
}

# Coronal views share the same density cubes and dump times.
coronal=(
  --density-dir "${DATA_ROOT}/rho"
  --dump-times "${DATA_ROOT}/dump_times.json"
)

# CCOR, STEREO-A COR2, LASCO C2, and the held-out polar view.
render L5 "${coronal[@]}" --observer L5 --inner-rsun 3.7 --outer-rsun 17
render L4 "${coronal[@]}" --observer L4 --inner-rsun 2.5 --outer-rsun 15
render L1 "${coronal[@]}" --observer L1 --inner-rsun 2.2 --outer-rsun 8.3
render P1 "${coronal[@]}" --observer 0 80 1 --inner-rsun 2 --outer-rsun 30

# At 1 AU the 20,000″ inner cutoff is about 21 solar radii.
render PUNCH --observer L5 --inner-rsun 21 --outer-rsun 120 \
  --density-dir "${CORHEL_ROOT}/rho" --dump-times "${CORHEL_ROOT}/dump_times.json" \
  --n-samples 512 --ray-batch 512 --overwrite
