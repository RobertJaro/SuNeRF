#!/usr/bin/env bash

# PSI_DATA contains the 'rho' and 't' folders with the paired snapshots
# (rho/rho001813.h5, t/t001813.h5, ...). One frame is rendered per pair.
REFERENCE_DATE="2026-01-01T00:00"
CADENCE_SECONDS='3600'
PSI_DATA="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd"
SYNTHETIC_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/psi_euv/observers"
RESOLUTION='128'
# First model: the synthesis uses the same limb treatment as the reconstruction
# (config/plasma/psi_observers.yaml): optically thin emission with the TR cutoff,
# no H/He absorption, and rays that stop at an opaque sphere at the EUV limb.
# Remove --no-absorption and --min-radius to render the physical limb layer.
# FITS timestamps are the snapshot times (--no-light-travel-time); the
# reconstruction uses module.light_travel_time: false accordingly.
LIMB_RADIUS='1.007'

python -m sunerf.resources.build verify

################################################################################
# One call per observer and instrument. LOCATION is L1..L5, mercury..neptune,
# or Stonyhurst 'longitude_deg,latitude_deg,distance_au'.
################################################################################
python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/aia_earth" \
  --instrument AIA --location earth \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time

python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/euvi_a_l4" \
  --instrument EUVI-A --location L4 \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time

python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/euvi_b_l5" \
  --instrument EUVI-B --location L5 \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time

python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/aia_l3" \
  --instrument AIA --location L3 \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time

python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/euvi_a_north" \
  --instrument EUVI-A --location 0,60,1.0 \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time

python -m sunerf.data.psi.render_euv \
  --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/euvi_b_south" \
  --instrument EUVI-B --location 0,-60,1.0 \
  --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
  --resolution "${RESOLUTION}" \
  --no-absorption --min-radius "${LIMB_RADIUS}" --no-light-travel-time
