module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF


# download soho lasco c2
python -m sunerf.data.download.download_lasco  --start 2010-03-15  --end 2010-04-15 --out /glade/work/rjarolim/data/sunerf-cme/2010_03/lasco --detector C2


# download stereo cor2
python -m sunerf.data.download.download_cor --start 2010-03-15  --end 2010-04-15 --out /glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2 --detector COR2 --source STEREO_A
python -m sunerf.data.download.download_cor --start 2010-03-15  --end 2010-04-15 --out /glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2 --detector COR2 --source STEREO_B

# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2/*.fts"
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2/*.fts"

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


# prep LASCO files - not working
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep'

filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2/*')

FOREACH element, filenames DO reduce_level_05, element, header, ima, savedir='/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep', /NOROLL_CORRECT


FOR i=0, N_ELEMENTS(filenames)-1 DO BEGIN & $
  CATCH, err & $
  IF err NE 0 THEN BEGIN & $
    PRINT, 'Error in reduce_level_1 for: ', filenames[i] & $
    PRINT, !ERROR_STATE.MSG & $
    CATCH, /CANCEL & $
  ENDIF ELSE BEGIN & $
    reduce_level_1, filenames[i], header, ima, savedir='/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep' $
    CATCH, /CANCEL & $
  ENDELSE & $
ENDFOR