#!/bin/bash -l

#PBS -N SuNeRF-psi-euv-render
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb
#PBS -l walltime=06:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# PSI_DATA contains the 'rho' and 't' folders with the paired snapshots
# (rho/rho001813.h5, t/t001813.h5, ...). One frame is rendered per pair.
REFERENCE_DATE="2026-01-01T00:00"
CADENCE_SECONDS='3600'
PSI_DATA="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd"
SYNTHETIC_ROOT="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/psi_euv/observers_v02"
# Geometry of the prepared observations of every plasma campaign: 512x512 pixels
# over +/-1.4 R_sun (pixels beyond 1.4 R_sun are NaN), so the reconstruction
# (config/plasma/psi_observers.yaml) sees the pixel scale and field of view it
# sees on real data. The rays are integrated over the same 1.007..1.7 R_sun
# shell as the reconstruction.
RESOLUTION='512'
FOV_RSUN='2.8'
LIMB_RADIUS='1.007'
SHELL_RADIUS='1.7'
# The synthesis keeps the PSI transition-region cutoff (default of render_euv)
# and, like the reconstruction, includes no H/He absorption and stops rays at
# the opaque sphere at the EUV limb. Remove --no-absorption and --min-radius to
# render the physical limb layer. FITS timestamps are detector times (snapshot
# time plus light travel time), as for real data; the reconstruction maps them
# back with module.light_travel_time: true.

python -m sunerf.resources.build verify

render () {
  python -m sunerf.data.psi.render_euv \
    --psi-data "${PSI_DATA}" --out-path "${SYNTHETIC_ROOT}/$1" \
    --instrument "$2" --location "$3" \
    --reference-date "${REFERENCE_DATE}" --cadence-seconds "${CADENCE_SECONDS}" \
    --resolution "${RESOLUTION}" --fov-rsun "${FOV_RSUN}" \
    --no-absorption --min-radius "${LIMB_RADIUS}" --max-radius "${SHELL_RADIUS}"
}

################################################################################
# One call per observer and instrument. LOCATION is L1..L5, mercury..neptune,
# or Stonyhurst 'longitude_deg,latitude_deg,distance_au'.
################################################################################
# Training views (ecliptic).
render aia_earth AIA earth
render euvi_a_l4 EUVI-A L4
render euvi_b_l5 EUVI-B L5
# Validation-only views (config/plasma/psi_observers.yaml uses euvi_a_north).
render euvi_a_north EUVI-A 0,60,1.0
render aia_l3 AIA L3
render euvi_b_south EUVI-B 0,-60,1.0
