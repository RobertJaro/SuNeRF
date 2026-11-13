#!/bin/bash -l

#PBS -N SuNeRF-plasma-polar-2025-10-eval
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb
#PBS -l walltime=02:00:00

set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
RUN_ROOT="/glade/work/rjarolim/sunerf/polar_2025_10_plasma_absorption_v03"
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
PREPARED_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2025_10/prepared"
# Hourly training frames closest to the EUI frame at 06:00.
AIA_FILE=$(ls "${PREPARED_ROOT}"/aia/aia.lev1_euv_12s.2025-10-14T06*.171.image_lev1.prepared.fits | head -n 1)
EUVI_FILE=$(ls "${PREPARED_ROOT}"/euvi_a/20251014_06*.prepared.fits | head -n 1)

################################################################################
# Solar Orbiter validation figures at the held-out EUI frame closest to the
# latitude maximum (2025-10-14T09:00, the validation frame of the run), with the
# AIA and EUVI-A 171 training inputs closest in time:
#   comparison.png  SuNeRF FSI 174 (mean and std in log10 counts matched to EUI),
#                   EUI FSI 174, AIA 171 and EUVI-A 171 reprojected to Solar Orbiter
#   thermal.png     AIA 193 render from Solar Orbiter, emission measure in two
#                   temperature bins over Stonyhurst +-60 deg, temperature and
#                   density in the Stonyhurst 0 deg meridional plane
################################################################################
python -m sunerf.evaluation.plasma.solo_comparison \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/solo_comparison" \
  --eui_file "${PREPARED_ROOT}/eui/solo_L2_eui-fsi174-image_20251014T060055315_V01.prepared.fits" \
  --aia_file "${AIA_FILE}" \
  --euvi_file "${EUVI_FILE}" \
  --em_edges 5.7 6.2 7.0 \
  --batch_size 8192
