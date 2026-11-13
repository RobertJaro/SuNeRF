#!/usr/bin/env bash
SUNERF_PATH="/glade/work/rjarolim/sunerf/all_2012_08_plasma_absorption_v01/save_state.snf"
EVALUATION_ROOT="/glade/work/rjarolim/sunerf/all_2012_08_plasma_absorption_v01/evaluation"

################################################################################
# Artifact-driven virtual-observer movies for every reconstructed instrument
################################################################################
python -m sunerf.evaluation.video \
  --chk-path "${SUNERF_PATH}" \
  --video-path "${EVALUATION_ROOT}/aia" \
  --instrument-key AIA \
  --channels A94 A131 A171 A193 A211 A335 \
  --resolution 512 \
  --batch-size 1024 \
  --fps 20

python -m sunerf.evaluation.video \
  --chk-path "${SUNERF_PATH}" \
  --video-path "${EVALUATION_ROOT}/euvi_a" \
  --instrument-key EUVI-A \
  --channels 171 195 284 \
  --resolution 512 \
  --batch-size 1024 \
  --fps 20

python -m sunerf.evaluation.video \
  --chk-path "${SUNERF_PATH}" \
  --video-path "${EVALUATION_ROOT}/euvi_b" \
  --instrument-key EUVI-B \
  --channels 171 195 284 \
  --resolution 512 \
  --batch-size 1024 \
  --fps 20

################################################################################
# Polar tour over the full reconstruction window (one frame per 6 h at the
# poles), fixed in Carrington longitude so the poles corotate with the Sun:
# Earth view at the window start (B0 5.8, L0 185.6) -> north pole -> forward in
# time at the north pole -> back in time while moving to the south pole ->
# forward in time at the south pole.
################################################################################
python -m sunerf.evaluation.plasma.video \
  --sunerf_path "${SUNERF_PATH}" \
  --out_path "${EVALUATION_ROOT}/video_polar_v2" \
  --resolution 256 --fov 1.4 \
  --panels aia171 aia193 aia211 em cool_column observer \
  --lat 0  --lon 185.6 --time 2012-08-01T00:00:00 --steps 1 \
  --lat -89   --lon 185.6 --time 2012-08-01T00:00:00 --steps 40 \
  --lat -89   --lon 185.6 --time 2012-08-31T18:00:00 --steps 256 \
  --fps 10

echo "Evaluation products written to ${EVALUATION_ROOT}"
