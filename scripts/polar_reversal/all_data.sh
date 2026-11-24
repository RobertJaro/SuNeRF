#!/bin/bash -l

#PBS -N SuNeRF-polar-reversal-data
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF
conda activate sunerf

# Quarterly data processing of the polar reversal set: download.sh,
# prepare.sh and clear.sh per quarter, so only one quarter of raw and
# calibrated files exists at a time. A finished quarter is marked with
# logs/<quarter>.done and skipped without any archive query.
#
# The job chains itself: until every quarter is marked it first submits its
# successor, which starts when this job ends (also at the walltime). A quarter
# cut by the walltime resumes in the next job: downloads skip records with a
# raw file or a prepared product, SECCHI_PREP skips existing calibrated files
# and the preparation existing products. If a quarter fails, the successor is
# removed so a broken quarter does not loop.
#
# Afterwards: clean.sh, then overview.sh.
LOG_ROOT="/glade/campaign/hao/radmhd/rjarolim/SUNERF_POLAR/data_2011_2014/logs"

QUARTERS=(
  "2011-02-06T00:00:00 2011-04-01T00:00:00"
  "2011-04-01T00:00:00 2011-07-01T00:00:00"
  "2011-07-01T00:00:00 2011-10-01T00:00:00"
  "2011-10-01T00:00:00 2012-01-01T00:00:00"
  "2012-01-01T00:00:00 2012-04-01T00:00:00"
  "2012-04-01T00:00:00 2012-07-01T00:00:00"
  "2012-07-01T00:00:00 2012-10-01T00:00:00"
  "2012-10-01T00:00:00 2013-01-01T00:00:00"
  "2013-01-01T00:00:00 2013-04-01T00:00:00"
  "2013-04-01T00:00:00 2013-07-01T00:00:00"
  "2013-07-01T00:00:00 2013-10-01T00:00:00"
  "2013-10-01T00:00:00 2014-01-01T00:00:00"
  "2014-01-01T00:00:00 2014-04-01T00:00:00"
  "2014-04-01T00:00:00 2014-07-01T00:00:00"
  "2014-07-01T00:00:00 2014-10-01T00:00:00"
)

mkdir -p "${LOG_ROOT}"

remaining=0
for quarter in "${QUARTERS[@]}"; do
  read -r start end <<< "${quarter}"
  if [ ! -f "${LOG_ROOT}/${start:0:10}_${end:0:10}.done" ]; then
    remaining=$((remaining + 1))
  fi
done
echo "${remaining} of ${#QUARTERS[@]} quarters remaining"
if [ "${remaining}" -eq 0 ]; then
  exit 0
fi

NEXT_JOB=$(qsub -W depend=afterany:"${PBS_JOBID}" scripts/polar_reversal/all_data.sh)

for quarter in "${QUARTERS[@]}"; do
  read -r start end <<< "${quarter}"
  label="${start:0:10}_${end:0:10}"
  if [ -f "${LOG_ROOT}/${label}.done" ]; then
    continue
  fi
  echo "Quarter ${label}"
  if ! { bash scripts/polar_reversal/download.sh "${start}" "${end}" &&
         bash scripts/polar_reversal/prepare.sh &&
         bash scripts/polar_reversal/clear.sh; }; then
    echo "Quarter ${label} failed"
    if [ -n "${NEXT_JOB}" ]; then
      qdel "${NEXT_JOB}"
    fi
    exit 1
  fi
  date -u +%Y-%m-%dT%H:%M:%S > "${LOG_ROOT}/${label}.done"
done

if [ -n "${NEXT_JOB}" ]; then
  qdel "${NEXT_JOB}"
fi
echo "All quarters prepared; next: clean.sh, then overview.sh"
