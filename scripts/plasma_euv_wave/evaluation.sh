#!/bin/bash -l

#PBS -N SuNeRF-plasma-euv-wave-2022-03-eval
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb
#PBS -l walltime=06:00:00

# Videos of the EUV wave reconstruction over the focus window
# (2022-03-28 10:30 .. 12:30, the 1 min tier of scripts/plasma_euv_wave/download.sh).
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
RUN_ROOT="/glade/work/rjarolim/sunerf/euv_wave_2022_03_plasma_v04"
# Weights-only save state of the run (overwritten after every validation); model, response
# calibration and data configuration are reconstructed from that file alone.
SUNERF_PATH="${RUN_ROOT}/save_state.snf"
# focus window at 1 min: 121 frames
FOCUS_START="2022-03-28T10:30:00"
FOCUS_END="2022-03-28T12:30:00"
FOCUS_STEPS="20"

################################################################################
# EUV wave from Earth (SDO view): AIA 193, column density, three emission-measure
# temperature bins over the diagnostic grid, observer geometry. The color scales are
# fixed by the first frame (pre-flare), so the wave and the flare brightening show as
# changes against the quiet corona.
################################################################################
python -m sunerf.evaluation.plasma.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_earth_focus" \
  --observer earth --resolution 256 --fov 1.4 \
  --subframe -0.3 0.3 0 0.6 \
  --panels aia193 density em observer \
  --time "${FOCUS_START}" --steps 1 \
  --time "${FOCUS_END}" --steps "${FOCUS_STEPS}" \
  --fps 10 --overwrite

################################################################################
# Solar Orbiter near-quadrature view (EUI FSI 174, ~83.5 deg west of Earth),
# with the emission-weighted temperature.
################################################################################
python -m sunerf.evaluation.plasma.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_solo_focus" \
  --observer solo --resolution 512 --fov 1.4 \
  --panels eui174 aia193 aia211 density temperature em observer \
  --time "${FOCUS_START}" --steps 1 \
  --time "${FOCUS_END}" --steps "${FOCUS_STEPS}" \
  --fps 10

################################################################################
# Moving view through the wave: from the Earth view at the focus start to the Solar Orbiter view at the end of the focus window, with
# time and viewpoint advancing together (1 min per frame, shortest arc in
# longitude between the two spacecraft positions). All three training channels,
# density and the emission-weighted distance from the observer.
################################################################################
python -m sunerf.evaluation.plasma.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_earth_to_solo" \
  --resolution 256 --fov 1.4 \
  --panels aia193 density em observer \
  --observer earth --time "${FOCUS_START}" --steps 1 \
  --observer solo  --time "${FOCUS_END}"        --steps "${FOCUS_STEPS}" \
  --fps 10 --overwrite

python -m sunerf.evaluation.plasma.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/high_latitude" \
  --resolution 256 --fov 1.4 \
  --panels aia193 density em observer \
  --lon 110 --lat 0 --time "${FOCUS_START}" --steps 1 \
  --lon 110 --lat 45  --time "${FOCUS_END}"        --steps "${FOCUS_STEPS}" \
  --fps 10 --overwrite

################################################################################
# Radially integrated emission measure per temperature bin and column density on
# the Carrington grid over the focus window (1 min): the wave as a change of the
# emission measure in each temperature bin, independent of any viewpoint.
################################################################################
python -m sunerf.evaluation.plasma.video_em_maps \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${RUN_ROOT}/video_em_maps_focus" \
  --time_start "${FOCUS_START}" --time_end "${FOCUS_END}" --cadence_minutes 1 \
  --n_radius 256 --n_lat 180 --n_lon 360 \
  --em_bins 3 --fps 10
