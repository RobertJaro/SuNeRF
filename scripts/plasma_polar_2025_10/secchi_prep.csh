#!/bin/csh -f
# Radiometric calibration of one STEREO/EUVI spacecraft with SolarSoft SECCHI_PREP.

if ($#argv != 3) then
  echo "Usage: csh secchi_prep.csh INPUT_GLOB OUT_DIR SSW_ROOT"
  exit 2
endif

setenv SECCHI_EUVI_INPUT_GLOB "$argv[1]"
setenv SECCHI_EUVI_OUT "$argv[2]"

module load idl
setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "$argv[3]"
source "$SSW/gen/setup/setup.ssw"

sswidl << IDL
resolve_routine, 'SECCHI_PREP'

files = file_search(getenv('SECCHI_EUVI_INPUT_GLOB'))
if N_ELEMENTS(files) EQ 0 then message, 'No EUVI inputs found'
out = getenv('SECCHI_EUVI_OUT')
FILE_MKDIR, out
SECCHI_PREP, files, /write_fts, /NORMAL_OFF, SAVEPATH=out

exit
IDL

set idl_status = $status
exit $idl_status
