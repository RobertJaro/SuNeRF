#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################### Download Data ####################
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2010-01-01T00:00:00' --t_end '2011-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2011-01-01T00:00:00' --t_end '2012-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2012-01-01T00:00:00' --t_end '2013-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2013-01-01T00:00:00' --t_end '2014-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2014-01-01T00:00:00' --t_end '2015-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2015-01-01T00:00:00' --t_end '2016-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2016-01-01T00:00:00' --t_end '2017-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2017-01-01T00:00:00' --t_end '2018-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2018-01-01T00:00:00' --t_end '2019-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2019-01-01T00:00:00' --t_end '2020-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2020-01-01T00:00:00' --t_end '2021-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2021-01-01T00:00:00' --t_end '2022-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2022-01-01T00:00:00' --t_end '2023-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2023-01-01T00:00:00' --t_end '2024-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2024-01-01T00:00:00' --t_end '2025-01-01T00:00:00' --cadence '24h' --channel '193'
#python -m sunerf.data.download.download_aia --download_dir '/glade/work/rjarolim/data/sunerf-conditioned/aia_193' --email 'robert.jarolim@uni-graz.at' --t_start '2025-01-01T00:00:00' --t_end '2026-01-01T00:00:00' --cadence '24h' --channel '193'

#################################################
# Data Preparation
#python -m sunerf.data.euv.prep_aia_v2 --data_path "/glade/work/rjarolim/data/sunerf-conditioned/aia_193/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-conditioned/aia_193_prep" --resolution 256

#python -m sunerf.data.conditioned.convert_aia_data \
#  --input "/glade/work/rjarolim/data/sunerf-conditioned/aia_193_prep/*.fits" \
#  --output "/glade/work/rjarolim/data/sunerf-conditioned/aia_193_npz" \
#  --nproc 32


#################################################
# Training
python -m sunerf.run_conditioned --config "config/conditioned/aia_193.yaml"
