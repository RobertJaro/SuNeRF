#!/bin/csh -f
# SECCHI_PREP of COR clear and total-brightness double-exposure images.
#
# Usage: csh scripts/secchi_prep_clear.csh INPUT_GLOB OUT_DIR

if ($#argv != 2) then
  echo "Usage: csh scripts/secchi_prep_clear.csh INPUT_GLOB OUT_DIR"
  exit 2
endif

setenv SECCHI_CLEAR_INPUT_GLOB "$argv[1]"
setenv SECCHI_CLEAR_OUT "$argv[2]"

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
files = file_search(getenv('SECCHI_CLEAR_INPUT_GLOB'))
if files[0] EQ '' then message, 'No SECCHI clear files found for ' + getenv('SECCHI_CLEAR_INPUT_GLOB')

resolve_routine, 'SECCHI_PREP'

clear_out = getenv('SECCHI_CLEAR_OUT')

FILE_MKDIR, clear_out
SECCHI_PREP, files, /write_fts, SAVEPATH=clear_out

exit
EOF
