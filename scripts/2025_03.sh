#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=256gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################### 2025-03 ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2025_03/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2025-03-03T22:00:00' --t_end '2025-04-30T12:00:00' --cadence 1h --channel 171

# convert data (pre-training with center crop)
python -m sunerf.data.prep.sdo --sdo_file_path "/glade/work/rjarolim/data/sunerf/2025_03/aia/*.fits" --output_path "/glade/work/rjarolim/data/sunerf/2025_03/prep" --scale 2.2
python -m sunerf.data.prep.stereo --stereo_file_path "/glade/work/rjarolim/data/sunerf/stereo_iti_2025/171/*.fits" --output_path "/glade/work/rjarolim/data/sunerf/2025_03/prep" --scale 2.2
# training step for 1 epoch
python -m sunerf.run_emission --config "config/emission/2025_03-171.yaml"
