#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Repository
################################################################################
cd /glade/u/home/rjarolim/projects/SuNeRF

################################################################################
# Configuration
################################################################################
DATA_ROOT="/glade/work/rjarolim/data/sunerf-cme/2010_03"
START="2010-03-15"
END="2010-04-15"
WORKERS="10"
IDL_BATCH="/tmp/sunerf_stereo_a_cor2_$$.pro"

################################################################################
# Download STEREO-A COR2 polarized triplets
################################################################################
python -m sunerf.data.download.download_cor \
  --start "${START}" \
  --end "${END}" \
  --out "${DATA_ROOT}/stereo_a_cor2" \
  --detector COR2 \
  --source STEREO_A

################################################################################
# Remove incomplete or invalid polarization triplets
################################################################################
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "${DATA_ROOT}/stereo_a_cor2/*.fts"

################################################################################
# Run SECCHI_PREP through SolarSoft/IDL
################################################################################
# Produces total-brightness and polarized-brightness FITS products.
cat > "${IDL_BATCH}" <<IDL
filenames = file_search('${DATA_ROOT}/stereo_a_cor2/*.fts')
FILE_MKDIR, '${DATA_ROOT}/stereo_a_cor2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='${DATA_ROOT}/stereo_a_cor2_prep/tB'
FILE_MKDIR, '${DATA_ROOT}/stereo_a_cor2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='${DATA_ROOT}/stereo_a_cor2_prep/pB'
exit
IDL

csh <<CSH
cd \$HOME
module load idl
setenv SSW_INSTR "SECCHI LASCO STEREO SOHO"
setenv SSW \$HOME/ssw
setenv NRL_LIB \$SSW/soho/lasco
source \$SSW/gen/setup/setup.ssw
sswidl < "${IDL_BATCH}"
CSH
rm -f "${IDL_BATCH}"

################################################################################
# Prepare total-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_coronagraph \
  --data_path "${DATA_ROOT}/stereo_a_cor2_prep/tB/*.fts" \
  --out_path "${DATA_ROOT}/prep/stereo_a_cor2/tB" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --overwrite

################################################################################
# Prepare polarized-brightness maps
################################################################################
python -m sunerf.data.coronagraph.prep_coronagraph \
  --data_path "${DATA_ROOT}/stereo_a_cor2_prep/pB/*.fts" \
  --out_path "${DATA_ROOT}/prep/stereo_a_cor2/pB" \
  --occ_min 4000 \
  --occ_max 15000 \
  --resize 512 512 \
  --num_workers "${WORKERS}" \
  --overwrite

################################################################################
# Compute correction masks
################################################################################
python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/stereo_a_cor2/tB/*" \
  --output "${DATA_ROOT}/prep/masks/stereo_a_cor2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/stereo_a_cor2/pB/*" \
  --output "${DATA_ROOT}/prep/masks/stereo_a_cor2_pB_correction.npy"

################################################################################
# Quicklook movie
################################################################################
python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${DATA_ROOT}/prep/video/stereo_a_cor2" \
  --tb "${DATA_ROOT}/prep/stereo_a_cor2/tB/*" \
  --pb "${DATA_ROOT}/prep/stereo_a_cor2/pB/*" \
  --tb_correction "${DATA_ROOT}/prep/masks/stereo_a_cor2_tB_correction.npy" \
  --pb_correction "${DATA_ROOT}/prep/masks/stereo_a_cor2_pB_correction.npy" \
  --vmin 1e-12 \
  --vmax 1e-8 \
  --num_workers "${WORKERS}"
