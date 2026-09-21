#!/bin/bash -l

#PBS -N sst_13392
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=24:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF



#################### Download Data ####################
python -m sunerf.data.download.download_aia --output '/glade/work/rjarolim/data/sunerf/polar/aia' --email 'robert.jarolim@uni-graz.at' --start '2025-03-01T00:00:00' --end '2025-05-01T00:00:00' --cadence '6h' --channels '171'
python -m sunerf.data.download.download_euvi --output '/glade/work/rjarolim/data/sunerf/polar/euvi' --start '2025-03-01T00:00:00' --end '2025-05-01T00:00:00' --cadence 6h --channels 171 --sources 'STEREO_B'
python -m sunerf.data.download.download_eui --output '/glade/work/rjarolim/data/sunerf/polar/eui' --start '2025-03-01T00:00:00' --end '2025-05-01T00:00:00' --cadence 6h --channels 174
