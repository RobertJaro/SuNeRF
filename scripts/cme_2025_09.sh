#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb:gpu_type=h100
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Train - for script use
python -m sunerf.run_thomson --config "config/cme/2025_09_cor2_ccor.yaml"
exit

#################################################
# Prepare data for CME reconstruction

# STEREO/COR2
python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/pB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB" --occ_min 4000 --occ_max 15000 --resize 512 512
python -m sunerf.data.coronagraph.prep_coronagraph --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/cor_prep/tB/*.fts" --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB" --occ_min 4000 --occ_max 15000 --resize 512 512

# CCOR
python -m sunerf.data.coronagraph.prep_ccor --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/ccor/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor" --resize 512 512

# PUNCH
python -m sunerf.data.coronagraph.prep_punch --data_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/punch/*.fits" --out_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/punch" --resize 512 512

#################################################
# check data
# COR2
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/cor2" --pb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" --vmin 1.0e-12
# CCOR
python -m sunerf.data.coronagraph.quicklook_coronagraph_video "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/video/ccor" --tb "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*" --vmin 1.0e-12

#################################################
# clean invalid files
python -m sunerf.data.coronagraph.clean_invalid --invalid_files "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/invalid_files.txt" --base_path "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/**/*" --dry_run

#################################################
# compute correction masks
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/tB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/stereo_a_cor2_tB_correction.npy"
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/cor2/pB/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/stereo_a_cor2_pB_correction.npy"
python -m sunerf.data.coronagraph.compute_correction --type full-min --input "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/ccor/*" --output "/glade/work/rjarolim/data/sunerf-cme/2025_09/prep/masks/ccor_tB_correction.npy"

#################################################
# Train
python -m sunerf.run_thomson --config "config/cme/2025_09.yaml"
python -m sunerf.run_thomson --config "config/cme/2025_09_cor.yaml"


#################################################
# Download validation data
python -m sunerf.data.download.download_aia \
  --download_dir "/glade/work/rjarolim/data/sunerf-cme/aia_validation/2025_09" \
  --email robert.jarolim@uni-graz.at \
  --t_start 2025-09-01T00:00:00 \
  --t_end 2025-09-19T00:00:00 \
  --cadence 1d \
  --channel 193

#################################################
# Evaluation

# compare to Carrington maps
python -m sunerf.evaluation.cme.carrington_map_comparison \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_cor2_ccor_v03/save_state.snf" \
  --aia_glob "/glade/work/rjarolim/data/sunerf-cme/aia_validation/2025_09/*.fits"

python -m sunerf.evaluation.cme.video_polar --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_cor2_ccor_v03/save_state.snf"
python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_cor2_ccor_v03/save_state.snf"
