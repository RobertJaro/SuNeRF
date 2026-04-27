#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Train - for script use
python -m sunerf.run_thomson --config "config/cme/2025_09_06_09.yaml"
exit


#################################################
# Evaluation

# compare to Carrington maps
python -m sunerf.evaluation.cme.video_polar --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_06_09_v01/save_state.snf" --occ_range 3 50

python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/punch" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/punch_pam/pB/*" --xlim -100000 100000 --ylim -100000 100000
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/cor2" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/cor2/pB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/ccor" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/ccor/*"

# video
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v12/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v12/video" \
  --lon_frame hci \
  --lat_range 7.3 7.3  --lon_range -90.8 -90.8  --time_range 2025-09-06T12:00 2025-09-07T00:00 --steps 20 --radius_range 3 50 \
  --lat_range 7.3 7.3  --lon_range -90.8 269.2 --time_range 2025-09-07T00:00 2025-09-07T00:00 --steps 36 --radius_range 3 50 \
  --lat_range 7.3 -45.0  --lon_range 269.2 240.2 --time_range 2025-09-07T00:00 2025-09-07T00:00 --steps 20 --radius_range 3 50 \
  --lat_range -45.0 -45.0  --lon_range 240.2 240.2 --time_range 2025-09-07T00:00 2025-09-08T00:00 --steps 20 --radius_range 3 50 \
  --lat_range -45.0 -80.0  --lon_range 240.2 240.2 --time_range 2025-09-08T00:00 2025-09-20T00:00 --steps 20 --radius_range 3 50 \
  --lat_range -80.0 80.0  --lon_range 240.2 240.2 --time_range 2025-09-20T00:00 2025-09-20T00:00 --steps 20 --radius_range 3 50 \


# video - inner FOV
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v12/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v12/video_inner" \
  --lon_frame hci \
  --lat_range 7.3 7.3  --lon_range -90.8 -90.8  --time_range 2025-09-06T12:00 2025-09-06T17:00 --steps 10 --radius_range 2.5 20 \
  --lat_range 7.3 7.3  --lon_range -90.8 269.2 --time_range 2025-09-06T17:00 2025-09-07T00:00 --steps 36 --radius_range 2.5 20 \

