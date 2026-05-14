#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=2:mem=128gb:gpu_type=h100
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

python -m sunerf.run_thomson --config "config/cme/2026_04.yaml"


exit

#################################################
# Evaluation

python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/video2" \
  --lon_frame hci \
  --lat -4.660   --lon 140.137 --time 2026-04-23T00:00 --steps 1  --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 140.137 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-24T06:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-23T00:00 --steps 20 --radius_min 3 --radius_max 20 \


python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/video_polar2" \
  --lon_frame hci \
  --lat 80   --lon 0 --time 2026-04-01T00:00 --steps 1  --radius_min 3 --radius_max 20 \
  --lat 80   --lon 0 --time 2026-05-01T00:00 --steps 100 --radius_min 3 --radius_max 20


python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/cor2_clear" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2026_04/prep/cor2_clear/*" --ds_key "STEREO_A_COR2_CLEAR"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/cor2" --ref_pB_path "/glade/work/rjarolim/data/sunerf-cme/2026_04/prep/cor2/pB/*" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2026_04/prep/cor2/tB/*" --ds_key "STEREO_A_COR2"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/ccor" --ref_tB_path "/glade/work/rjarolim/data/sunerf-cme/2026_04/prep/ccor/*"


python -m sunerf.convert.sunerf_to_vtk --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" --times "2026-04-23T17:00" "2026-04-23T18:00" "2026-04-23T19:00" "2026-04-23T20:00" "2026-04-23T21:00" --radius_range 3 20 --pixel_per_Rs 4

python -m sunerf.evaluation.image_sequence_to_video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2026_04/video2 \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2026_04/video2.mp4 \
  --fps 10
