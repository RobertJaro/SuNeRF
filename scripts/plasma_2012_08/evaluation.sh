#!/usr/bin/env bash
SUNERF_PATH="/glade/work/rjarolim/sunerf/all_2012_08_plasma_v2/save_state.safe.pt"
EVALUATION_ROOT="/glade/work/rjarolim/sunerf/all_2012_08_plasma_v2/evaluation"

mkdir -p "${EVALUATION_ROOT}"

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

echo "Evaluation products written to ${EVALUATION_ROOT}"
