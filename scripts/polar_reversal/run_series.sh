#!/bin/bash -l

#PBS -N SuNeRF-polar-reversal-series
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=64:ngpus=4:mem=256gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

cd /glade/u/home/rjarolim/projects/SuNeRF

# Series reconstruction (config/plasma/polar_reversal_series.yaml): the window
# of run.sh moves forward in 5-day steps, each step warm-started from the one
# before; save states under <series base_path>/steps/<index>_<focus center>/.
#
# The job chains itself: until the series is complete it first submits its
# successor, which starts when this job ends (also at the walltime). A step cut
# by the walltime resumes from its last checkpoint in the next job. If the
# series fails, the successor is removed so a broken step does not loop.
SERIES_PATH="/glade/work/rjarolim/sunerf/polar_reversal_series_v01"

NEXT_JOB=""
if [ ! -f "${SERIES_PATH}/series.done" ]; then
  NEXT_JOB=$(qsub -W depend=afterany:"${PBS_JOBID}" scripts/polar_reversal/run_series.sh)
fi

if ! python -m sunerf.run_plasma_series --config config/plasma/polar_reversal_series.yaml; then
  if [ -n "${NEXT_JOB}" ]; then
    qdel "${NEXT_JOB}"
  fi
  exit 1
fi
