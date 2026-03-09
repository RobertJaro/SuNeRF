module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# download ccor
python -m sunerf.data.download.download_ccor \
  --start 2025-09-01T00:00:00 \
  --end 2025-09-20T00:00:00 \
  --cadence 1h \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/ccor

# download stereo cor2
python -m sunerf.data.download.download_cor --start 2025-09-01T00:00:00  --end 2025-09-20T00:00:00 --out /glade/work/rjarolim/data/sunerf-cme/2025_09/cor --detector COR2


# download PUNCH NFI data
python -m sunerf.data.download.download_punch  --start 2025-09-01T00:00:00  --end 2025-09-20T00:00:00 --out /glade/work/rjarolim/data/sunerf-cme/2025_09/punch


# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor/*.fts" --dry-run



# prep COR files
csh

cd $HOME
module load idl

setenv SSW_INSTR "SECCHI STEREO"
setenv SSW $HOME/ssw
source $SSW/gen/setup/setup.ssw
sswidl

# prep COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2025_09/cor/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/pB'
