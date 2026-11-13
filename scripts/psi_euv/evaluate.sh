#!/bin/bash -l

#PBS -N SuNeRF-psi-euv-evaluate
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=1:mem=128gb
#PBS -l walltime=02:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Compare the closed-loop reconstruction with the PSI/MAS cube it was rendered from.
# REFERENCE_DATE and LIMB_RADIUS must match scripts/psi_euv/render.sh; --time is
# the emission (snapshot) time of the first snapshot.
REFERENCE_DATE="2026-01-01T00:00"
LIMB_RADIUS='1.007'
PSI_DATA="/glade/campaign/hao/radmhd/rjarolim/SuNeRF_2023_03/psi_data/mhd"
RESULT_ROOT="/glade/work/rjarolim/sunerf/psi_euv_v02"

python -m sunerf.evaluation.psi_euv_truth \
  --psi-data "${PSI_DATA}" \
  --reconstruction "${RESULT_ROOT}/save_state.snf" \
  --time "${REFERENCE_DATE}" \
  --min-radius "${LIMB_RADIUS}" \
  --absorption-artifact none \
  --out-path "${RESULT_ROOT}/evaluation" --save-truth
