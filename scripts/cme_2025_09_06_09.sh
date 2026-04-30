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

python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/punch" --ref_pB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/punch_pam/pB/*" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/punch_pam/tB/*" --xlim -100000 100000 --ylim -100000 100000
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/cor2" --ref_pB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/cor2/pB/*" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/cor2/tB/*"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v11/ccor" --ref_map "/glade/work/rjarolim/data/sunerf-cme/2025_09_06_09_1200/prep/ccor/*"

# video
python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2025_09_v15/video" \
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
