module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# download ccor
python -m sunerf.data.download.download_ccor \
  --start 2025-09-01T00:00:00 \
  --end 2025-10-01T00:00:00 \
  --cadence 1h \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/ccor_l2 \
  --product-prefix "SWFO/GOES-19/CCOR-1/ccor1-l2_science"

# download stereo cor2
python -m sunerf.data.download.download_cor --start 2025-09-10T00:00:00  --end 2025-09-20T00:00:00 --out /glade/work/rjarolim/data/sunerf-cme/2025_09/cor --detector COR2

# clean up COR triplets
python -m sunerf.data.cor.remove_invalid_triplets \
  --glob "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor/*.fts" --dry-run

# download PUNCH NFI data
#python -m sunerf.data.download.download_punch  \
#  --level l1 --instrument nfi --ext 0j.fits \
#  --start 2025-09-01T00:00:00  --end 2025-09-20T00:00:00 \
#  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/punch_nfi \

# download PUNCH - CAM
#python -m sunerf.data.download.download_punch \
#  --level l3 --product CAM \
#  --start 2025-09-01T00:00:00 --end 2025-10-01T00:00:00 \
#  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/punch_cam

# download PUNCH - PAM
python -m sunerf.data.download.download_punch \
  --level l3 --product PAM --ext 0l.fits \
  --start 2025-09-01T00:00:00 --end 2025-10-01T00:00:00 \
  --out /glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2025_09/punch_pam \
  --download-mode day-by-day

# download PSP in-situ data for the full month
python -m sunerf.data.download.download_psp_insitu \
  --start 2025-09-01T00:00:00 \
  --end 2025-10-01T00:00:00 \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/psp/raw

# prep PSP in-situ data into sparse SuNeRF trajectory constraints
python -m sunerf.data.prep.psp_insitu \
  --raw-dir /glade/work/rjarolim/data/sunerf-cme/2025_09/psp/raw \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/prep/psp/psp_insitu_20250901_20251001.npz \
  --start 2025-09-01T00:00:00 \
  --end 2025-10-01T00:00:00 \
  --plot-dir /glade/work/rjarolim/data/sunerf-cme/2025_09/prep/psp/plots

# download Solar Orbiter COHO in-situ data for the full month
python -m sunerf.data.download.download_solo_insitu \
  --start 2025-09-01T00:00:00 \
  --end 2025-10-01T00:00:00 \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/solo/raw

# prep Solar Orbiter COHO proton density and RTN velocity into in-situ constraints
python -m sunerf.data.prep.solo_insitu \
  --raw-dir /glade/work/rjarolim/data/sunerf-cme/2025_09/solo/raw \
  --out /glade/work/rjarolim/data/sunerf-cme/2025_09/prep/solo/solo_insitu_20250901_20251001.npz \
  --start 2025-09-01T00:00:00 \
  --end 2025-10-01T00:00:00 \
  --electron-density-factor 1.0 \
  --plot-dir /glade/work/rjarolim/data/sunerf-cme/2025_09/prep/solo/plots

# prep COR files
csh <<'CSH'
cd $HOME
module load idl

setenv SSW_INSTR "SECCHI STEREO"
setenv SSW $HOME/ssw
source $SSW/gen/setup/setup.ssw
sswidl <<'IDL'

# prep COR2 files
filenames = file_search('/glade/work/rjarolim/data/sunerf-cme/2025_09/cor/*.fts')
FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB'

FILE_MKDIR, '/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB'
SECCHI_PREP, filenames, /write_fts, /polariz_on, /pB, SAVEPATH='/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB'
exit
IDL
CSH
