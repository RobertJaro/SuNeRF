#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################################################
########### Data Selection ###########

# find coordinates for data prep (subframe)
#python -m sunerf.data.euv.find_coordinate --data_path "/glade/work/rjarolim/data/sunerf/2023_04/aia/aia.lev1_euv_12s.2023-04-13T000006Z.193.image_lev1.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep" --lat -22 --lon -258
#python -m sunerf.data.euv.find_coordinate --data_path "/glade/work/rjarolim/data/sunerf/2023_04/eui/solo_L2_eui-fsi174-image_20230424T230055162_V01.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep" --lat -22 --lon -258


#################################################################################
########### Data Preparation ###########
python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_04/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep/aia_fd" --resolution 512 --max_radius 2.0 --date_range '2023-04-06T00:00:00' '2023-04-16T00:00:00'
python -m sunerf.data.euv.prep_eui --data_path "/glade/work/rjarolim/data/sunerf/2023_04/eui/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep/eui_fd" --resolution 512 --max_radius 2.0 --date_range '2023-04-06T00:00:00' '2023-04-16T00:00:00'
python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_04/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep/aia_sub" --max_radius 2.0 --lat -22 --lon -258 --hpc_width 400 --hpc_height 400 --date_range '2023-04-06T00:00:00' '2023-04-16T00:00:00'
python -m sunerf.data.euv.prep_eui --data_path "/glade/work/rjarolim/data/sunerf/2023_04/eui/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_04/prep/eui_sub" --max_radius 2.0 --lat -22 --lon -258 --hpc_width 400 --hpc_height 400 --date_range '2023-04-06T00:00:00' '2023-04-16T00:00:00'

#################################################################################
########### Training ###########

python -m sunerf.run_plasma --config "config/combined_2023_04.yaml"
#python -m sunerf.run_plasma --config "config/aia_2023_04.yaml"

#################################################################################
########### Evaluation ###########
#python -i -m  sunerf.evaluation.slices --chk_path "/glade/work/rjarolim/sunerf/2023_04_combined_v05/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_04_combined_v05/evaluation/slices"
#python -m  sunerf.evaluation.video_observer --chk_path "/glade/work/rjarolim/sunerf/2023_04_combined_v04/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_04_combined_v04/evaluation/video"
#python -m  sunerf.evaluation.video_4pi --chk_path "/glade/work/rjarolim/sunerf/2023_04_combined_v05/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_04_combined_v05/evaluation/video_4pi"
#python -m  sunerf.evaluation.video_poles --chk_path "/glade/work/rjarolim/sunerf/2023_04_combined_v04/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_04_combined_v04/evaluation/video_poles"



#python -i -m  sunerf.evaluation.fits_planets --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/save_state.snf" --out_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/evaluation/planets"
#python -i -m  sunerf.evaluation.mars_video --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/save_state.snf" --out_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/evaluation/mars_video"
