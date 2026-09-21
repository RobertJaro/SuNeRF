#!/usr/bin/env bash
set -euo pipefail

python -m sunerf.response.emissivity --root "${1:-data/response_calibration}"
