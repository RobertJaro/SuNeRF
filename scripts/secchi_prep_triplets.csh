#!/bin/csh -f

if (! $?SECCHI_INPUT_GLOB) then
  echo "SECCHI_INPUT_GLOB is not set"
  exit 1
endif

if (! $?SECCHI_TB_OUT) then
  echo "SECCHI_TB_OUT is not set"
  exit 1
endif

if (! $?SECCHI_PB_OUT) then
  echo "SECCHI_PB_OUT is not set"
  exit 1
endif

cd "${HOME}"
module load idl

setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "${HOME}/ssw"
source $SSW/gen/setup/setup.ssw

sswidl << EOF
files = file_search(getenv('SECCHI_INPUT_GLOB'))
if N_ELEMENTS(files) EQ 0 then message, 'No SECCHI files found for ' + getenv('SECCHI_INPUT_GLOB')

resolve_routine, 'SECCHI_PREP'

tb_out = getenv('SECCHI_TB_OUT')
pb_out = getenv('SECCHI_PB_OUT')

FILE_MKDIR, tb_out
SECCHI_PREP, files, /write_fts, /polariz_on, SAVEPATH=tb_out

FILE_MKDIR, pb_out
SECCHI_PREP, files, /write_fts, /polariz_on, /pB, SAVEPATH=pb_out

exit
EOF
