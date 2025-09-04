#!/bin/bash -l

#PBS -N sst_13392
#PBS -A P22100000
#PBS -q preempt
#PBS -l select=1:ncpus=8:ngpus=2:mem=24gb
#PBS -l walltime=24:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF



#################### Download Data ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/polar/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2025-03-01T00:00:00' --t_end '2025-05-01T00:00:00' --cadence '6h' --channel '171'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/polar/euvi' --t_start '2025-03-01T00:00:00' --t_end '2025-05-01T00:00:00' --cadence 6 --channels 171 --sources 'STEREO_B'
python -m sunerf.data.download.download_eui_174 --download_dir '/glade/work/rjarolim/data/sunerf/polar/eui' --t_start '2025-03-01T00:00:00' --t_end '2025-05-01T00:00:00' --cadence 6
