#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb:gpu_type=h100
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Prepare data for CME reconstruction

# Metis data
#python -m sunerf.data.metis.prep_metis --data_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_Metis/tb/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/tB" --occ_min 5700 --resize 512 512
#python -m sunerf.data.metis.prep_metis --data_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_Metis/pb/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB" --occ_min 5700 --resize 512 512

# STEREO/COR2
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB" --occ_min 3000 --occ_max 15000 --resize 512 512
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/cor/COR2_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/tB" --occ_min 3000 --occ_max 15000 --resize 512 512

#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C2_prep_fixed/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c2" --occ_min 2100 --occ_max 8000 --resize 512 512 --clip_max 1e+5

# SOHO/LASCO C3
#python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/lasco/C3_prep/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c3" --occ_min 4500 --occ_max 30000 --resize 512 512

#################################################
# check data
# COR2
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/video/cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/tB/*"
# Metis
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/video/metis" --pb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/tB/*"
# LASCO C2
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/video/lasco_c2" --tb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c2/*"
# LASCO C3
#python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/video/lasco_c3" --tb "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c3/*"

#################################################
# clean invalid files
#python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/**/*" --dry_run

#################################################
# Train
#python -m sunerf.run_thomson --config "config/cme/202409_combined.yaml"

#python -m sunerf.run_thomson --config "config/cme/202409_cor2_lasco.yaml"
python -m sunerf.run_thomson --config "config/cme/202409_cor2_metis.yaml"
#python -m sunerf.run_thomson --config "config/cme/202409_cor2_metis_nophysics.yaml"


# static
#python -m sunerf.run_thomson --config "config/cme/202409_metis_static.yaml"
#python -m sunerf.run_thomson --config "config/cme/202409_cor2_static.yaml"

# single instrument - time evolving
#python -m sunerf.run_thomson --config "config/cme/202409_cor2.yaml"

#################################################
# Evaluation
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf"
python -m sunerf.evaluation.cme.load_ref_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/20240917_110745_1P4c2A.fts"
python -m sunerf.evaluation.cme.plot_slice --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf"


python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*.fts" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/ref_series_cor2"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/ref_series_metis"

#  COR2
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_lasco_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/tB/*.fts" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_lasco_v01/ref_series_cor2"
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_lasco_v01/save_state.snf"

python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_lasco_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c2/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_lasco_v01/ref_series_lasco"


# Combined
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v09/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v09/ref_series_metis"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v09/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v09/ref_series_cor2"
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v09/save_state.snf" --radius 2 3 4 5
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v10/save_state.snf" --longitudes 0 15 30 45 60 75 90
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_combined_v10/save_state.snf"


# COR2 + Metis
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/lasco_c2/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/ref_series_lasco"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/ref_series_cor2"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*" --out_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/ref_series_metis"
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf" --longitudes 0 15 30 45 60 75 90
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v01/save_state.snf"

python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v05/save_state.snf" --longitudes 0 15 30 45 60 75 90 --radius_range 1.5 15 --time_range "2024-09-22T00:00" "2024-10-01T00:00"
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v05/save_state.snf" --occ_range 1.5 15
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_metis_v05/save_state.snf" --longitudes -10 -5 0 5 10 --radius_range 1.5 15 --time_range "2024-09-20T00:00" "2024-10-01T00:00"


# Metis static
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_metis_static_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*"
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_metis_static_v01/save_state.snf" --radius 2 3 4 5


# COR2 static
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_static_v02/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*"
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_static_v02/save_state.snf"
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_static_v02/save_state.snf" --longitudes 0 15 30 45 60 75 90


# COR2
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/cor2/pB/*"
python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_v01/save_state.snf" --longitudes 0 30 60 90 120 150 180 --time_range "2024-09-22T00:00" "2024-10-01T00:00"
python -m sunerf.evaluation.cme.plot_radius_map --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_v01/save_state.snf" --radius 6 9 12 15
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_v01/save_state.snf" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2024_10/prep/metis/pB/*"
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/202409_cor2_v01/save_state.snf"
