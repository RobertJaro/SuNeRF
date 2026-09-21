#!/bin/csh -f

if ($#argv != 5) then
  echo "Usage: csh secchi_prep.csh A_GLOB B_GLOB A_OUT B_OUT SSW_ROOT"
  exit 2
endif

setenv SECCHI_EUVI_A_INPUT_GLOB "$argv[1]"
setenv SECCHI_EUVI_B_INPUT_GLOB "$argv[2]"
setenv SECCHI_EUVI_A_OUT "$argv[3]"
setenv SECCHI_EUVI_B_OUT "$argv[4]"

module load idl
setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "$argv[5]"
source "$SSW/gen/setup/setup.ssw"

sswidl << IDL
resolve_routine, 'SECCHI_PREP'

files_a = file_search(getenv('SECCHI_EUVI_A_INPUT_GLOB'))
if N_ELEMENTS(files_a) EQ 0 then message, 'No STEREO-A/EUVI inputs found'
out_a = getenv('SECCHI_EUVI_A_OUT')
FILE_MKDIR, out_a
SECCHI_PREP, files_a, /write_fts, /NORMAL_OFF, SAVEPATH=out_a

files_b = file_search(getenv('SECCHI_EUVI_B_INPUT_GLOB'))
if N_ELEMENTS(files_b) EQ 0 then message, 'No STEREO-B/EUVI inputs found'
out_b = getenv('SECCHI_EUVI_B_OUT')
FILE_MKDIR, out_b
SECCHI_PREP, files_b, /write_fts, /NORMAL_OFF, SAVEPATH=out_b

exit
IDL

set idl_status = $status
exit $idl_status
