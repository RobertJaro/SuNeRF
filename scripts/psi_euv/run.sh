#!/usr/bin/env bash
# First closed-loop reconstruction of the synthetic PSI observations.

# Fixed per-channel loss divisors of each instrument, estimated once from the
# training observations; an existing table is reused (--overwrite re-estimates).
python -m sunerf.data.euv.estimate_scaling --config config/plasma/psi_observers.yaml

python -m sunerf.run_plasma --config config/plasma/psi_observers.yaml
