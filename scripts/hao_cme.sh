#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=32gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# 6 Viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_220W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_160W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_100W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_040W_bang_0000_*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_6" --check_matching

# All Viewpoints
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/*B/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_all" --check_matching

# Background
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_005.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_006.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_007.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_ecliptic_background" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*_008.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_ecliptic_background" --check_matching

# Prep 1view
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_060W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_1" --check_matching

# Prep 2view
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_2" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_2" --check_matching

# Prep 3view
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_3" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_3" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_080W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_3" --check_matching

# Prep Heliosphere
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_helio" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_140W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_helio" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_260W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_helio" --check_matching

# Prep Polar
python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polar" --check_matching
python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_140W_bang_040W_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polar" --check_matching
python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_260W_bang_040S_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_polar" --check_matching

# Prep Ecliptic
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_*_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_ecliptic" --check_matching

# all viewpoints
#python -m sunerf.run_thomson --config "config/cme/hao_all.yaml"
# ecliptic
#python -m sunerf.run_thomson --config "config/cme/hao_ecliptic.yaml"
# 3 viewpoints
#python -m sunerf.run_thomson --config "config/cme/hao_helio.yaml"
#python -m sunerf.run_thomson --config "config/cme/hao_3view.yaml"
python -m sunerf.run_thomson --config "config/cme/hao_polar.yaml"
# 2 viewpoints
#python -m sunerf.run_thomson --config "config/cme/hao_2view.yaml"
#python -m sunerf.run_thomson --config "config/cme_v02/hao_2view_no_physics.yaml"
#python -m sunerf.run_thomson --config "config/cme/hao_2view_background.yaml"
# 1 viewpoints
#python -m sunerf.run_thomson --config "config/cme/hao_1view.yaml"

#####################################################################
# 2 viewpoints variations (60 deg separation)


# 000 deg and 60 deg -- id: 000_060
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_0000_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_000_060" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_060W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_000_060" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "000_060" --reload

# 040 deg and 100 deg -- id: 040_100
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_040W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_040_100" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_100W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_040_100" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "040_100" --reload

# 080 deg and 140 deg -- id: 080_140
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_080W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_080_140" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_140W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_080_140" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "080_140" --reload

# 120 deg and 180 deg -- id: 120_180
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_120W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_120_180" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_180W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_120_180" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "120_180" --reload

# 160 deg and 220 deg -- id: 160_220
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_160W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_160_220" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_220W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_160_220" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "160_220" --reload

# 200 deg and 260 deg -- id: 200_260
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_200W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_200_260" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_260W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_200_260" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "200_260" --reload

# 240 deg and 300 deg -- id: 240_300
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_240W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_240_300" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_300W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_240_300" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "240_300" --reload

# 280 deg and 340 deg -- id: 280_340
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_280W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_280_340" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_340W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_280_340" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "280_340" --reload

# 320 deg and 20 deg -- id: 320_020
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_320W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_320_020" --check_matching
#python -m sunerf.data.prep.prep_hao_cme --resolution 512 --hao_path "/glade/work/rjarolim/data/sunerf-cme/hao/data_fits/dcmer_020W_bang_0000_*/*.fits" --output_path "/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_320_020" --check_matching
#python -m sunerf.run_thomson --config "config/cme/hao_2view_variations.yaml" --id "320_020" --reload


####################################################################
# Evaluation
# all viewpoints
#python -i -m sunerf.evaluation.cme.evaluate_cme_parameters --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/all_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
# Helio - 3 viewpoints
#python -m sunerf.evaluation.cme.evaluate_cme_parameters --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/helio_v02/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
# Polar - 2 viewpoints
#python -m sunerf.evaluation.cme.evaluate_cme_parameters --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polar_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
# 2 viewpoints
#python -m sunerf.evaluation.cme.evaluate_cme_parameters --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/2view_v02/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
# 2 viewpoints + Background
#python -m sunerf.evaluation.cme.evaluate_cme_parameters --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/2view_background_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"

####################################################################
# center of mass
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_000_060_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --plot_ground_truth
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_040_100_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --plot_velocity
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_080_140_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_120_180_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_160_220_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_200_260_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_240_300_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_280_340_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_320_020_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
# 3 viewpoints
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/3view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --plot_ground_truth
# no physics
python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/2view_no_physics_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"

# angle comparison
#python -i -m sunerf.evaluation.cme.angle_comparison

# com comparison
#python -i -m sunerf.evaluation.cme.com_comparison

# data overview plot
python -i -m sunerf.evaluation.visualize_cme_input


#############################################################################
# Heliospheric mapping
# all
#python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/all_v01/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-v2/all_v01/evaluation_full_v02" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks --plot_ground_truth
# ecliptic
#python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/ecliptic_v02/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-v2/ecliptic_v02/evaluation_full_v02" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
# helio
#python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/helio_v04/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-v2/helio_v04/evaluation_full_v02" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks
# polar
#python -m sunerf.evaluation.cme.center_of_mass --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/polar_v02/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-v2/polar_v02/evaluation_full_v02" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --min_longitude None --max_longitude None --min_latitude -70 --max_latitude 70 --plot_white_r_ticks

#############################################################################
# Tomography
#python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/ecliptic_v02/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" --plot_ground_truth
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/3view_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"
python -m sunerf.evaluation.cme.cme_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/variations/2view_040_100_v01/save_state.snf" --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav"


#######################################################################
# VTK
#python -m sunerf.convert.sunerf_to_vtk --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/all_v01/save_state.snf" --times 30

#######################################################################
# Visualize CME
#python -i -m sunerf.evaluation.cme.cme_visualization --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/helio_v03/save_state.snf"
