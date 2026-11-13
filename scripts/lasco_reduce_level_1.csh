#!/bin/csh -f
# Calibration of LASCO level-0.5 images to level 1 with SolarSoft REDUCE_LEVEL_1.
# After the last NRL pointing-table day (2024-05-15) the roll comes from the SOHO
# predictive roll files under ANCIL_DIR/attitude, and the sun center is the fixed
# post-2005 (C2) / post-2009 (C3) estimate.
#
# Files whose level-1 output already exists in OUT_DIR are skipped.
#
# Usage: csh scripts/lasco_reduce_level_1.csh INPUT_GLOB OUT_DIR SSW_ROOT ANCIL_DIR

if ($#argv != 4) then
  echo "Usage: csh scripts/lasco_reduce_level_1.csh INPUT_GLOB OUT_DIR SSW_ROOT ANCIL_DIR"
  exit 2
endif

setenv LASCO_INPUT_GLOB "$argv[1]"
setenv LASCO_L1_OUT "$argv[2]"
setenv LASCO_PRO_DIR `dirname "$0"`

# idl/9.2.0 is only available after ncarenv/25.10 on Casper.
module load ncarenv/25.10 idl/9.2.0
if (! $?IDL_DIR) then
  echo "IDL_DIR is not set: 'module load ncarenv/25.10 idl/9.2.0' failed."
  exit 3
endif
# LASCO has no IDL_STARTUP that adds the NRL package (GET_ROLL_OR_XY, GET_SUN_CENTER);
# GET_SUN_CENTER calls the SECCHI function SCC_SUN_CENTER.
setenv SSW_INSTR "LASCO SOHO NRL SECCHI STEREO"
setenv SSW "$argv[3]"
set ancil_dir = "$argv[4]"
# setup.ssw reads instrument names from argv; the input glob must not reach it.
set argv = ()
source "$SSW/gen/setup/setup.ssw"
# The site setup.lasco_env points LASCO_DATA to $NRL_LIB/lasco/data, which does not exist.
setenv NRL_LIB "$SSW/soho/lasco"
setenv LASCO_DATA "$NRL_LIB/idl/data"
setenv ANCIL_DATA "$ancil_dir"

foreach required ( \
    "$LASCO_DATA/calib/c2vig_final.fts" \
    "$LASCO_DATA/calib/c3vig_postint_final.fts" \
    "$LASCO_DATA/calib/c3_cl_mask_lvl1.fts" \
    "$LASCO_DATA/calib/C3ramp.fts" \
    "$LASCO_DATA/calib/c2_c3_last_post_recovery_day.sav" \
    "$NRL_LIB/idl/expfac/data" \
    "$ANCIL_DATA/attitude/roll/nominal_roll_attitude.dat" \
    "$ANCIL_DATA/attitude/predictive" \
    "$SSW/packages/nrl/idl" \
    "$SSW/stereo/secchi/idl/display/movie/scc_sun_center.pro" )
  if (! -e "$required") then
    echo "Missing: $required"
    exit 4
  endif
end

sswidl << IDL
!path = getenv('LASCO_PRO_DIR') + ':' + !path
resolve_routine, 'REDUCE_LEVEL_1'
resolve_routine, 'GET_ROLL_OR_XY', /IS_FUNCTION
resolve_routine, 'READ_SO_AT_ROLL_DAT', /IS_FUNCTION
resolve_routine, 'SCC_SUN_CENTER', /IS_FUNCTION
resolve_routine, 'GET_SUN_CENTER', /IS_FUNCTION
resolve_routine, 'LASCO_REDUCE_FILES'

files = file_search(getenv('LASCO_INPUT_GLOB'))
if files[0] EQ '' then message, 'No LASCO level-0.5 inputs found'
out = getenv('LASCO_L1_OUT')
FILE_MKDIR, out
cd, out
LASCO_REDUCE_FILES, files, out

exit
IDL

set idl_status = $status
exit $idl_status
