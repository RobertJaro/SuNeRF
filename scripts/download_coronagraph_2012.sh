module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF


# download stereo cor2
python -m sunerf.data.download.download_cor --start 2012-08-01  --end 2012-09-01 --out /glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2 --detector COR2 --source STEREO_A
python -m sunerf.data.download.download_cor --start 2012-08-01  --end 2012-09-01 --out /glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2 --detector COR2 --source STEREO_B

# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2/*.fts" --dry-run
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2/*.fts" --dry-run

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
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_a_cor2_prep/pB'

# prep STEREO B COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2012_08/stereo_b_cor2_prep/pB'