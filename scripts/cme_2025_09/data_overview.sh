#!/usr/bin/env bash
set -euo pipefail

cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Overview of the observations (raw PUNCH and CCOR tB, prepared COR2 pB) in the CME period (cme_time of 2025_09_cme.yaml):
# one folder per instrument, image plane in projected solar radii, transparent background
################################################################################
python -m sunerf.evaluation.cme.plot_data_overview \
  --out_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/data_overview" \
  --time_range "2025-09-21T12:00:00" "2025-09-23T12:00:00" \
  --punch_tB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/punch_pam/*.fits" \
  --ccor_tB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/ccor_l2/*.fits" \
  --cor2_pB "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/prep/cor2/pB/*" \
  --num_workers 4
