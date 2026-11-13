#!/bin/csh -f
# SECCHI_PREP of COR polarization triplets into total and polarized brightness.
#
# Usage: csh scripts/secchi_prep_triplets.csh INPUT_GLOB TB_OUT PB_OUT

if ($#argv != 3) then
  echo "Usage: csh scripts/secchi_prep_triplets.csh INPUT_GLOB TB_OUT PB_OUT"
  exit 2
endif

setenv SECCHI_INPUT_GLOB "$argv[1]"
setenv SECCHI_TB_OUT "$argv[2]"
setenv SECCHI_PB_OUT "$argv[3]"

cd "${HOME}"
# idl/9.2.0 is only available after ncarenv/25.10 on Casper.
module load ncarenv/25.10 idl/9.2.0
if (! $?IDL_DIR) then
  echo "IDL_DIR is not set: 'module load ncarenv/25.10 idl/9.2.0' failed."
  exit 3
endif

setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "/glade/work/rjarolim/ssw"
# setup.ssw reads instrument names from argv; the input glob must not reach it.
set argv = ()
source $SSW/gen/setup/setup.ssw

sswidl << EOF
files = file_search(getenv('SECCHI_INPUT_GLOB'))
if files[0] EQ '' then message, 'No SECCHI files found for ' + getenv('SECCHI_INPUT_GLOB')

resolve_routine, 'SECCHI_PREP'

tb_out = getenv('SECCHI_TB_OUT')
pb_out = getenv('SECCHI_PB_OUT')

FILE_MKDIR, tb_out
SECCHI_PREP, files, /write_fts, /polariz_on, SAVEPATH=tb_out

FILE_MKDIR, pb_out
SECCHI_PREP, files, /write_fts, /polariz_on, /pB, SAVEPATH=pb_out

exit
EOF
