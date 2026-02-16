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

#################################################
# Prepare data for CME reconstruction

## STEREO-A/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB" --occ_min 3200 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_a_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB" --occ_min 3200 --occ_max 15000 --resize 512 512
#
## STEREO-B/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB" --occ_min 3200 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/stereo_b_cor2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB" --occ_min 3200 --occ_max 15000 --resize 512 512
#
#
## fix LASCO headers and basic prep
#python -m sunerf.data.lasco.fix_observer "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep/*" --out-dir "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep_fixed"
#
## SOHO/LASCO C2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/lasco/C2_prep_fixed/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/lasco_c2" --occ_min 2100 --occ_max 8000 --resize 512 512 --clip_max 1e+5


#################################################
# check data
# COR2
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/stereo_a_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/tB/*" --vmin 1e-12 --vmax 1e-8
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/stereo_b_cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/tB/*" --vmin 1e-12 --vmax 1e-8
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/video/lasco_c2" --tb "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/lasco_c2/*" --vmin 1e-12 --vmax 1e-8


#################################################
# clean invalid files
#python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/**/*" --dry_run

#################################################
# Train
python -m sunerf.run_thomson --config "config/cme/2010_03_cor2.yaml"
#python -m sunerf.run_thomson --config "config/cme/2010_03_cor2_stereo_a.yaml"


#################################################
# Evaluation

# ref series
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/ref_series_stereo_a"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/ref_series_stereo_b"
# tomography
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/save_state.snf" --longitudes 0 30 60 90 120 150 180 --time_range "2010-04-02T00:00" "2010-04-03T00:00"
# radius map
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/save_state.snf" --radius 5 8 12 15
# video
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v01/save_state.snf"



python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_v06/save_state.snf" --radius 5 8 12 15


# STEREO A only
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v04/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_a_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v04/ref_series_stereo_a"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v04/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2010_03/prep/stereo_b_cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2010_03_cor2_stereo_a_v04/ref_series_stereo_b"
