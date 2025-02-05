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



#################### 2012-08 ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-01T00:00:00' --t_end '2012-08-05T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-05T00:00:00' --t_end '2012-08-10T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-10T00:00:00' --t_end '2012-08-15T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-15T00:00:00' --t_end '2012-08-20T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2012-08-20T00:00:00' --t_end '2012-08-25T00:00:00'

python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-01T00:00:00' --t_end '2012-08-05T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-05T00:00:00' --t_end '2012-08-10T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-10T00:00:00' --t_end '2012-08-15T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-15T00:00:00' --t_end '2012-08-20T00:00:00'
python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2012_08/euvi' --t_start '2012-08-20T00:00:00' --t_end '2012-08-25T00:00:00'


#################### 2023-01 ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_01/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-01-01T00:00:00' --t_end '2023-01-05T00:00:00'

python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2023_01/euvi' --t_start '2023-01-01T00:00:00' --t_end '2023-01-05T00:00:00'

python -i -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_01/eui' --t_start '2023-01-01T00:00:00' --t_end '2023-01-05T00:00:00'


#################### 2010-06 ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2010_06/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2010-06-01T00:00:00' --t_end '2010-06-05T00:00:00'

python -m sunerf.data.download.download_euvi --download_dir '/glade/work/rjarolim/data/sunerf/2010_06/euvi' --t_start '2010-06-01T00:00:00' --t_end '2010-06-05T00:00:00' --source 'STEREO_B'


#################### 2023-03-06 ####################
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_03/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-03-06T00:00:00' --t_end '2023-03-11T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_03/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-03-11T00:00:00' --t_end '2023-03-16T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_03/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-03-16T00:00:00' --t_end '2023-03-21T00:00:00'

python -m sunerf.data.download.download_suvi --download_dir '/glade/work/rjarolim/data/sunerf/2023_03/suvi' --t_start '2023-03-16T00:00:00' --t_end '2023-03-21T00:00:00'


#################### 2023-04-01 ####################
# AIA
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-01T00:00:00' --t_end '2023-04-05T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-05T00:00:00' --t_end '2023-04-10T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-10T00:00:00' --t_end '2023-04-15T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-15T00:00:00' --t_end '2023-04-20T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-20T00:00:00' --t_end '2023-04-25T00:00:00'
python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/aia' --email 'robert.jarolim@uni-graz.at' --t_start '2023-04-25T00:00:00' --t_end '2023-04-30T00:00:00'
# EUI
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-01T00:00:00' --t_end '2023-04-05T00:00:00'
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-05T00:00:00' --t_end '2023-04-10T00:00:00'
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-10T00:00:00' --t_end '2023-04-15T00:00:00'
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-15T00:00:00' --t_end '2023-04-20T00:00:00'
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-20T00:00:00' --t_end '2023-04-25T00:00:00'
python -m sunerf.data.download.download_eui --download_dir '/glade/work/rjarolim/data/sunerf/2023_04/eui' --t_start '2023-04-25T00:00:00' --t_end '2023-04-30T00:00:00'

