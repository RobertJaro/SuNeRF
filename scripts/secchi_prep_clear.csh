#!/bin/csh -f

if (! $?SECCHI_CLEAR_INPUT_GLOB) then
  echo "SECCHI_CLEAR_INPUT_GLOB is not set"
  exit 1
endif

if (! $?SECCHI_CLEAR_OUT) then
  echo "SECCHI_CLEAR_OUT is not set"
  exit 1
endif

cd "${HOME}"
module load idl

setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "${HOME}/ssw"
source $SSW/gen/setup/setup.ssw

sswidl << EOF
files = file_search(getenv('SECCHI_CLEAR_INPUT_GLOB'))
if N_ELEMENTS(files) EQ 0 then message, 'No SECCHI clear files found for ' + getenv('SECCHI_CLEAR_INPUT_GLOB')

resolve_routine, 'SECCHI_PREP'

clear_out = getenv('SECCHI_CLEAR_OUT')

FILE_MKDIR, clear_out
SECCHI_PREP, files, /write_fts, SAVEPATH=clear_out

exit
EOF
