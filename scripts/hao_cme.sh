#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# 6 Viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_220W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_160W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_100W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_040W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_6" --check_matching

# All Viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_all" --check_matching

# Background
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_005.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_006.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_007.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_008.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_ecliptic_background" --check_matching

# Prep 1view
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_060W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_1" --check_matching


# Prep 2view
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data/prep_HAO_2" --check_matching


#python -m sunerf.run_thomson --config "config/cme/hao_6view.yaml"
#python -m sunerf.run_thomson --config "config/cme/hao_2view.yaml"
python -m sunerf.run_thomson --config "config/cme/hao_all.yaml"
