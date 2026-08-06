#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=4:ngpus=1:mem=24gb
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#################################################
# Train - for script use

python -m sunerf.evaluation.cme.video_reference_animation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" \
  --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/ccor/*" \
  --resolution 256 \
  --n_ref_samples 30 \
  --n_motion_frames 100 \
  --time_advance_days 14

#python -m sunerf.run_thomson --config "config/cme/2025_09_06_09.yaml"
exit


#################################################
# Evaluation

# compare to Carrington maps
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/punch" --ref_pB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/punch_pam/pB/*" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/punch_pam/tB/*" --xlim -100000 100000 --ylim -100000 100000
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/cor2" --ref_pB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/cor2/pB/*" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/cor2/tB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/ccor" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/ccor/*"

# video
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/video" \
  --lon_frame hci \
  --lat 7.3   --lon -90.8 --time 2025-09-06T12:00 --steps 1  --radius_min 3 --radius_max 50 \
  --lat 7.3   --lon -90.8 --time 2025-09-07T00:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat 7.3   --lon 269.2 --time 2025-09-07T00:00 --steps 36 --radius_min 3 --radius_max 50 \
  --lat -45.0 --lon 240.2 --time 2025-09-07T00:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat -45.0 --lon 240.2 --time 2025-09-08T00:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat -80.0 --lon 240.2 --time 2025-09-20T00:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat 80.0  --lon 240.2 --time 2025-09-20T00:00 --steps 20 --radius_min 3 --radius_max 50 \


# video - inner FOV
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/video_inner" \
  --lon_frame hci \
  --lat 5.2 --lon -45.6 --time 2025-09-06T15:00 --steps 1  --radius_min 3 --radius_max 30 \
  --lat 5.2 --lon -45.6 --time 2025-09-06T18:000 --steps 10 --radius_min 3 --radius_max 30 \
  --lat 5.2 --lon 100 --time 2025-09-06T18:000 --steps 10 --radius_min 3 --radius_max 30 \

# video - polar
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v24/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v24/video_polar" \
  --lon_frame hci \
  --no_observer \
  --lat 89 --lon 0 --time 2025-09-20T00:00 --steps 1  --radius_min 3 --radius_max 50 \
  --lat 89 --lon 0 --time 2025-09-25T00:00 --steps 100  --radius_min 3 --radius_max 50 \


python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v24/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v24/video_20_rot" \
  --lon_frame hci \
  --lat 5.2 --lon -45.6 --time 2025-09-21T00:00 --steps 1  --radius_min 3 --radius_max 50 \
  --lat 5.2 --lon -45.6 --time 2025-09-22T11:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat 5.2 --lon -100.0 --time 2025-09-22T11:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat 60.0 --lon -100.0 --time 2025-09-22T11:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat 60.0 --lon -100.0 --time 2025-09-23T00:00 --steps 20 --radius_min 3 --radius_max 50 \


python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/video_21_inner_v3" \
  --lon_frame hci \
  --lat 7.3   --lon -90.8 --time 2025-09-21T18:00 --steps 1  --radius_min 3 --radius_max 20 \
  --lat 7.3   --lon -90.8 --time 2025-09-22T10:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat 7.3   --lon 269.2 --time 2025-09-22T10:00 --steps 36 --radius_min 3 --radius_max 20 \
  --lat -80.0   --lon 269.2 --time 2025-09-22T10:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -80.0   --lon 269.2 --time 2025-09-22T10:00 --steps 20 --radius_min 3 --radius_max 50 \
  --lat -80.0   --lon 269.2 --time 2025-09-26T00:00 --steps 50 --radius_min 3 --radius_max 50 \
  --lat 80.0   --lon 269.2 --time 2025-09-26T00:00 --steps 20 --radius_min 3 --radius_max 50 \



python -m sunerf.evaluation.cme.plot_tomography --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v16/save_state.snf" --longitudes 0 30 60 90 120 150 180 --time_range "2025-09-21T18:00" "2025-09-23T00:00" --radius_range 3 50 --latitude_range 0 360


python -m sunerf.evaluation.image_sequence_to_video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2025_09_v13/video_21_inner_v3 \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2025_09_v13/video_21_inner_v3.mp4 \
  --fps 10

python -m sunerf.evaluation.image_sequence_to_video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2010_03_v17/tomography_cme \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2010_03_v17/tomography_cme.mp4 \
  --fps 20

python -m sunerf.evaluation.image_sequence_to_video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2025_09_06/video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2025_09_06/punch_cme_v2.mp4 \
  --fps 10




python -m sunerf.evaluation.cme.video_reference_animation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v17/save_state.snf" \
  --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/ccor/*" \
  --resolution 256 \
  --n_ref_samples 20 \
  --n_motion_frames 100 \
  --time_advance_days 14
