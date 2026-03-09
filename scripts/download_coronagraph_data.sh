module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF


# download soho lasco c2
python -m sunerf.data.download.download_lasco  --start 2024-09-15  --end 2024-10-02 --out /glade/work/rjarolim/data/sunerf-cme/2024_10/lasco --detector C2
# download stereo cor2
python -m sunerf.data.download.download_cor --start 2024-09-15  --end 2024-10-02 --out /glade/work/rjarolim/data/sunerf-cme/2024_10/cor --detector COR2
# download stereo cor1
python -m sunerf.data.download.download_cor --start 2024-09-15  --end 2024-10-02 --out /glade/work/rjarolim/data/sunerf-cme/2024_10/cor --detector COR1
# download lasco c3
python -m sunerf.data.download.download_lasco  --start 2024-09-15  --end 2024-10-02 --out /glade/work/rjarolim/data/sunerf-cme/2024_10/lasco --detector C3


# download PUNCH NFI data
python -m sunerf.data.download.download_punch  --start 2025-09-01  --end 2025-09-02 --out /glade/work/rjarolim/data/sunerf-cme/2025_09/punch


# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1/*.fts" --dry-run
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2/*.fts" --dry-run

# fix LASCO headers and basic prep
python -m sunerf.data.lasco.fix_observer "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2_prep/*" --out-dir "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2_prep_fixed"
python -m sunerf.data.lasco.fix_observer "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C3/*" --out-dir "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C3_prep"



# prep COR files
csh

cd $HOME
module load idl

setenv SSW_INSTR "SECCHI LASCO STEREO SOHO"
setenv SSW $HOME/ssw
setenv NRL_LIB $SSW/soho/lasco
source $SSW/gen/setup/setup.ssw
sswidl

# prep COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/pB'


# prep COR1 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR1_prep/pB'

# prep LASCO files - not working
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2_prep'

filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2/*')
FOREACH element, filenames DO reduce_level_1, element, header, ima, savedir='/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2_prep', /NOROLL_CORRECT
