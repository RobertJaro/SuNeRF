#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# find coordinates for data prep (subframe)
#python -i -m sunerf.data.euv.find_coordinate --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.193.image_lev1.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep" --lat -20 --lon -135

# prep subframe
#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/aia_fd" --resolution 256
#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/aia_sub" --lat 10 --lon -145 --hpc_width 700 --hpc_height 700 --date_range '2023-03-06T00:00:00' '2023-03-11T00:00:00'

# prep combined
#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/aia_fd" --resolution 256 --max_radius 2.0
#python -m sunerf.data.euv.prep_eui --data_path "/glade/work/rjarolim/data/sunerf/2023_03/eui/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/eui_fd" --resolution 256 --max_radius 2.0 --date_range '2023-03-06T00:00:00' '2023-03-21T00:00:00'
#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/aia_sub" --lat 10 --lon -145 --hpc_width 700 --hpc_height 700 --date_range '2023-03-06T00:00:00' '2023-03-11T00:00:00'
#python -m sunerf.data.euv.prep_eui --data_path "/glade/work/rjarolim/data/sunerf/2023_03/eui/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/prep/eui_sub" --lat 10 --lon -145 --hpc_width 700 --hpc_height 700 --date_range '2023-03-06T00:00:00' '2023-03-11T00:00:00'


# prep filament
#python -m sunerf.data.euv.prep_aia --data_path "/glade/work/rjarolim/data/sunerf/2023_03/aia/*.fits" --out_path "/glade/work/rjarolim/data/sunerf/2023_03/filament/aia" --lat -20 --lon -135 --hpc_width 1000 --hpc_height 1000


#python -m sunerf.run_plasma --config "config/subframe_2023_03.yaml"
#python -m sunerf.run_plasma --config "config/filament_2023_03.yaml"
python -m sunerf.run_plasma --config "config/combined_2023_03.yaml"
#python -m sunerf.run_plasma --config "config/calibration_2023_03.yaml"

########### Evaluation ###########
#python -i -m  sunerf.evaluation.slices --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v02/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_03_combined_v02/evaluation/slices"
#python -i -m  sunerf.evaluation.video_observer --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v02/save_state.snf" --video_path "/glade/work/rjarolim/sunerf/2023_03_combined_v02/evaluation/video"

#python -i -m  sunerf.evaluation.fits_planets --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/save_state.snf" --out_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/evaluation/planets"
#python -i -m  sunerf.evaluation.mars_video --chk_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/save_state.snf" --out_path "/glade/work/rjarolim/sunerf/2023_03_combined_v01/evaluation/mars_video"
