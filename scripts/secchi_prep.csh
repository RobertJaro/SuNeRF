#!/bin/csh -f
# Radiometric calibration of one STEREO/EUVI input set with SolarSoft SECCHI_PREP,
# shared by every plasma campaign (one call per spacecraft and sampling tier).
# /NORMAL_OFF matches the exported S1 throughput of the EUVI responses.
#
# Inputs whose calibrated file (level-0.5 name YYYYMMDD_HHMMSS_n4euA.fts, or
# another image-type letter such as s4euB -> YYYYMMDD_HHMMSS_14euA.fts) already
# exists in OUT_DIR are skipped, so a rerun
# resumes. SECCHI_PREP writes into OUT_DIR/.secchi_prep_staging and every
# finished batch is moved into OUT_DIR: a run cut off mid-write leaves no
# truncated file there (the staging directory is cleared on the next run).
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

files = file_search(getenv('SECCHI_EUVI_INPUT_GLOB'), count=n_files)
if n_files EQ 0 then message, 'No EUVI inputs found'
out = getenv('SECCHI_EUVI_OUT')
staging = out + '/.secchi_prep_staging'
FILE_MKDIR, out
FILE_DELETE, staging, /RECURSIVE, /ALLOW_NONEXISTENT
FILE_MKDIR, staging

names = file_basename(files)
targets = names
level_0 = where(stregex(strmid(names, 16, 1), '^[a-zA-Z]', /BOOLEAN), n_level_0)
if n_level_0 GT 0 then targets[level_0] = strmid(names[level_0], 0, 16) + '1' + strmid(names[level_0], 17)
todo = where(file_test(out + '/' + targets) EQ 0, n_todo)
print, n_files - n_todo, n_files, format='(I0, " of ", I0, " EUVI inputs already calibrated")'
if n_todo GT 0 then files = files[todo]

batch = 100L
for i = 0L, n_todo - 1, batch do begin & SECCHI_PREP, files[i:((i + batch - 1) < (n_todo - 1))], /write_fts, /NORMAL_OFF, SAVEPATH=staging & done = file_search(staging + '/*.fts', count=n_done) & if n_done GT 0 then FILE_MOVE, done, out, /OVERWRITE & endfor

FILE_DELETE, staging, /RECURSIVE, /ALLOW_NONEXISTENT
exit
IDL

set idl_status = $status
exit $idl_status
