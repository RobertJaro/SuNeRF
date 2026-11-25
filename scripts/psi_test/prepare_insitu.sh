#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
# Density cubes and dump times come from scripts/psi_test/download.sh.
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/psi_obs"
PSI_DENSITY_DIR="${DATA_ROOT}/rho"
PSI_DUMP_TIMES="${DATA_ROOT}/dump_times.json"
INSITU_DIR="${DATA_ROOT}/insitu"
# MAS velocity cubes (vr/vt/vp<dump>.hdf) are not part of the public release;
# once available, add "--velocity-dir ${DATA_ROOT}/velocity" to both calls.
# Pre-event samples mirror the 20 pre-event days of the images at 1 h cadence;
# the probe is inside the cubes only during the last two days of that span.
PRE_DURATION="20d"
PRE_CADENCE="1h"
# The probe follows PSP around the encounter-25 perihelion (2025-09-15T20:18,
# 9.86 R_sun), placed at 2021-10-27T20:00 so that the outbound probe is at
# 17-20 R_sun during the eruption, and rotated so that it sits at the CME
# longitude (Carrington 245 deg) at 20:00 UT of the eruption day.
PERIHELION_TIME="2021-10-27T20:00:00"
ALIGN_TIME="2021-10-28T20:00:00"
CME_LONGITUDE=245
# Measurement model: lognormal density scatter (QTN-SPC level), 25 km/s
# velocity errors, 20 % of the 30 min windows without velocity.
DENSITY_NOISE=0.15
VELOCITY_NOISE=25
GAP_FRACTION=0.2

################################################################################
# Training probe through the CME body
################################################################################
python -m sunerf.data.psi.synthetic_insitu \
  --density-dir "${PSI_DENSITY_DIR}" \
  --dump-times "${PSI_DUMP_TIMES}" \
  --out-file "${INSITU_DIR}/PSP_synth.npz" \
  --key PSP_synth \
  --perihelion-time "${PERIHELION_TIME}" \
  --align-time "${ALIGN_TIME}" \
  --cme-longitude-deg "${CME_LONGITUDE}" \
  --longitude-offset-deg 0 \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}" \
  --density-noise "${DENSITY_NOISE}" \
  --velocity-noise-kms "${VELOCITY_NOISE}" \
  --gap-fraction "${GAP_FRACTION}" \
  --seed 0 \
  --plot-dir "${INSITU_DIR}/qa"

################################################################################
# Held-out probe on the same orbit, 30 deg east of the CME direction (flank)
################################################################################
python -m sunerf.data.psi.synthetic_insitu \
  --density-dir "${PSI_DENSITY_DIR}" \
  --dump-times "${PSI_DUMP_TIMES}" \
  --out-file "${INSITU_DIR}/PSP_synth_heldout.npz" \
  --key PSP_synth_heldout \
  --perihelion-time "${PERIHELION_TIME}" \
  --align-time "${ALIGN_TIME}" \
  --cme-longitude-deg "${CME_LONGITUDE}" \
  --longitude-offset-deg 30 \
  --pre-duration "${PRE_DURATION}" \
  --pre-cadence "${PRE_CADENCE}" \
  --density-noise "${DENSITY_NOISE}" \
  --velocity-noise-kms "${VELOCITY_NOISE}" \
  --gap-fraction "${GAP_FRACTION}" \
  --seed 1 \
  --plot-dir "${INSITU_DIR}/qa"
