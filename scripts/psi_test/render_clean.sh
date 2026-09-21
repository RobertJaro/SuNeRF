#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
PSI_DENSITY_DIR="${DATA_ROOT}/rho"
PSI_CLEAN_DIR="${DATA_ROOT}/clean"
PSI_DENSITY_URL="https://www.predsci.com/~epalmerio/issi_rho3d/20211028/cor/"
# The cubes carry no time stamp. PSI's synthetic images of the same run hold the
# observation time of every dump in their headers, which fixes the cadence.
PSI_IMAGE_URL="https://www.predsci.com/~epalmerio/getpb/20211028/fakeC3/fits_L1/pb/"
PSI_DUMP_TIMES="${DATA_ROOT}/dump_times.json"
# Frames before the first cube, the relaxed pre-eruption state, render that cube
# in rigid corotation.
PRE_DURATION="20d"
PRE_CADENCE="6h"

################################################################################
# Download the complete PSI density-cube sequence (about 6.5 GB)
################################################################################
python -m sunerf.data.psi.download_psi_density \
  --out-dir "${PSI_DENSITY_DIR}" \
  --url "${PSI_DENSITY_URL}"

################################################################################
# Tabulate the observation time of every dump from the PSI image headers (the
# image data are not transferred)
################################################################################
python -m sunerf.data.psi.dump_times \
  --url "${PSI_IMAGE_URL}" \
  --out-file "${PSI_DUMP_TIMES}"

################################################################################
# Render clean tB/pB sequences in the field of view of the instrument that each
# view imitates; scripts/psi_test/prepare_data.sh adds the detector effects.
################################################################################
# L5_wide: PUNCH-like outer field from L5.
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir "${PSI_DENSITY_DIR}" \
  --out-dir "${PSI_CLEAN_DIR}/L5_wide" \
  --key L5_wide \
  --observer L5 \
  --inner-rsun 15 \
  --outer-rsun 30 \
  --resolution 512 \
  --dump-times "${PSI_DUMP_TIMES}" \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}"

# L5: CCOR-like field from L5.
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir "${PSI_DENSITY_DIR}" \
  --out-dir "${PSI_CLEAN_DIR}/L5" \
  --key L5 \
  --observer L5 \
  --inner-rsun 3.7 \
  --outer-rsun 17 \
  --resolution 512 \
  --dump-times "${PSI_DUMP_TIMES}" \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}"

# L4: STEREO COR2-like field from L4.
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir "${PSI_DENSITY_DIR}" \
  --out-dir "${PSI_CLEAN_DIR}/L4" \
  --key L4 \
  --observer L4 \
  --inner-rsun 2.5 \
  --outer-rsun 15 \
  --resolution 512 \
  --dump-times "${PSI_DUMP_TIMES}" \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}"

# L1: LASCO C2-like field from L1.
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir "${PSI_DENSITY_DIR}" \
  --out-dir "${PSI_CLEAN_DIR}/L1" \
  --key L1 \
  --observer L1 \
  --inner-rsun 2.2 \
  --outer-rsun 8.3 \
  --resolution 512 \
  --dump-times "${PSI_DUMP_TIMES}" \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}"

# P1: polar viewpoint at 80 degrees Stonyhurst latitude above the Sun--Earth
# line, kept clear over the full cube for held-out validation.
python -m sunerf.data.psi.render_psi_thomson \
  --density-dir "${PSI_DENSITY_DIR}" \
  --out-dir "${PSI_CLEAN_DIR}/P1" \
  --key P1 \
  --observer 0 80 1 \
  --inner-rsun 2 \
  --outer-rsun 30 \
  --resolution 512 \
  --dump-times "${PSI_DUMP_TIMES}" \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}"

################################################################################
# Write a tB/pB quicklook JPG of every rendered frame
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PSI_CLEAN_DIR}/L5_wide/jpg" \
  --tb "${PSI_CLEAN_DIR}/L5_wide/tb/*" \
  --pb "${PSI_CLEAN_DIR}/L5_wide/pb/*" \
  --a 1e-6

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PSI_CLEAN_DIR}/L5/jpg" \
  --tb "${PSI_CLEAN_DIR}/L5/tb/*" \
  --pb "${PSI_CLEAN_DIR}/L5/pb/*" \
  --a 1e-6

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PSI_CLEAN_DIR}/L4/jpg" \
  --tb "${PSI_CLEAN_DIR}/L4/tb/*" \
  --pb "${PSI_CLEAN_DIR}/L4/pb/*" \
  --a 1e-6

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PSI_CLEAN_DIR}/L1/jpg" \
  --tb "${PSI_CLEAN_DIR}/L1/tb/*" \
  --pb "${PSI_CLEAN_DIR}/L1/pb/*" \
  --a 1e-6

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${PSI_CLEAN_DIR}/P1/jpg" \
  --tb "${PSI_CLEAN_DIR}/P1/tb/*" \
  --pb "${PSI_CLEAN_DIR}/P1/pb/*" \
  --a 1e-6
