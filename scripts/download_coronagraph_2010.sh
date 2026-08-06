module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

DATA_ROOT=/glade/work/rjarolim/data/sunerf-cme/2010_03

# download prepped SOHO/LASCO C2 total-brightness and polarized archive data
python -m sunerf.data.download.download_lasco \
  --start 2010-03-15 \
  --end 2010-04-15 \
  --out ${DATA_ROOT}/lasco \
  --instrument C2 \
  --product level_1 polarized \
  --workers 10

python -m sunerf.data.coronagraph.prep_lasco \
  --clear_path "${DATA_ROOT}/lasco/c2/level_1/*.fts*" \
  --out_path "${DATA_ROOT}/prep/lasco_c2_clear" \
  --occ_min 2100 \
  --occ_max 6000 \
  --resize 512 512 \
  --num_workers 10 \
  --filter_bright_objects \
  --overwrite

python -m sunerf.data.coronagraph.prep_lasco \
  --pb_path "${DATA_ROOT}/lasco/c2/polarized/C2-PB-*.fts" \
  --percent_path "${DATA_ROOT}/lasco/c2/polarized/C2-%25P-*.fts" \
  --out_path "${DATA_ROOT}/prep/lasco_c2" \
  --resize 512 512 \
  --num_workers 10 \
  --filter_bright_objects \
  --overwrite

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/lasco_c2_clear/tB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_clear_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_tB_correction.npy"

python -m sunerf.data.coronagraph.compute_correction \
  --type full-min \
  --input "${DATA_ROOT}/prep/lasco_c2/pB/*" \
  --output "${DATA_ROOT}/prep/masks/lasco_c2_pB_correction.npy"

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${DATA_ROOT}/prep/video/lasco_c2_clear" \
  --tb "${DATA_ROOT}/prep/lasco_c2_clear/tB/*" \
  --vmin 1e-12 \
  --vmax 1e-8 \
  --num_workers 10

python -m sunerf.data.coronagraph.quicklook_coronagraph_video \
  "${DATA_ROOT}/prep/video/lasco_c2_polarized" \
  --tb "${DATA_ROOT}/prep/lasco_c2/tB/*" \
  --pb "${DATA_ROOT}/prep/lasco_c2/pB/*" \
  --vmin 1e-12 \
  --vmax 1e-8 \
  --num_workers 10

python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2010_03/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/**/*" --dry_run

# download stereo cor2
python -m sunerf.data.download.download_cor --start 2010-03-15  --end 2010-04-15 --out ${DATA_ROOT}/stereo_a_cor2 --detector COR2 --source STEREO_A
python -m sunerf.data.download.download_cor --start 2010-03-15  --end 2010-04-15 --out ${DATA_ROOT}/stereo_b_cor2 --detector COR2 --source STEREO_B

# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "${DATA_ROOT}/stereo_a_cor2/*.fts"
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "${DATA_ROOT}/stereo_b_cor2/*.fts"

# prep COR files
csh

cd $HOME
module load idl

setenv SSW_INSTR "SECCHI LASCO STEREO SOHO"
setenv SSW $HOME/ssw
setenv NRL_LIB $SSW/soho/lasco
source $SSW/gen/setup/setup.ssw
sswidl

# prep STEREO A COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/pB'

# prep STEREO B COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/pB'
