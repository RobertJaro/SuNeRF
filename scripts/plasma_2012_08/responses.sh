#!/usr/bin/env bash
RESPONSE_CALIBRATION_ROOT="data/response_calibration"
RESPONSE_SPECTRAL_EMISSIVITY="${RESPONSE_CALIBRATION_ROOT}/responses/chianti_coronal_2021.spectral.npz"
RESPONSE_ROOT="/glade/work/rjarolim/sunerf/responses/chianti_11.0.2_coronal_2021"

mkdir -p "${RESPONSE_ROOT}"

scripts/generate_spectral_emissivity.sh "${RESPONSE_CALIBRATION_ROOT}"

python -m sunerf.response.pipeline \
  --root "${RESPONSE_CALIBRATION_ROOT}" \
  prepare \
  --instrument aia \
  --instrument euvi_a \
  --instrument euvi_b

python -m sunerf.response.pipeline \
  --root "${RESPONSE_CALIBRATION_ROOT}" \
  build \
  --spectral-emissivity "${RESPONSE_SPECTRAL_EMISSIVITY}" \
  --output-dir "${RESPONSE_ROOT}" \
  --label 2012_08 \
  --instrument aia \
  --instrument euvi_a \
  --instrument euvi_b

echo "Published response artifacts under ${RESPONSE_ROOT}"
