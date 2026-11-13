#!/bin/csh -f
# Radiometric calibration of one STEREO/EUVI input set with SolarSoft SECCHI_PREP,
# shared by every plasma campaign (one call per spacecraft and sampling tier).
# /NORMAL_OFF matches the exported S1 throughput of the EUVI responses.
#
# Usage: csh scripts/secchi_prep.csh INPUT_GLOB OUT_DIR SSW_ROOT

if ($#argv != 3) then
  echo "Usage: csh scripts/secchi_prep.csh INPUT_GLOB OUT_DIR SSW_ROOT"
  exit 2
endif

setenv SECCHI_EUVI_INPUT_GLOB "$argv[1]"
setenv SECCHI_EUVI_OUT "$argv[2]"

# idl/9.2.0 is only available after ncarenv/25.10 on Casper.
module load ncarenv/25.10 idl/9.2.0
if (! $?IDL_DIR) then
  echo "IDL_DIR is not set: 'module load ncarenv/25.10 idl/9.2.0' failed."
  exit 3
endif
setenv SSW_INSTR "SECCHI STEREO"
setenv SSW "$argv[3]"
# setup.ssw reads instrument names from argv; the input glob must not reach it.
set argv = ()
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
