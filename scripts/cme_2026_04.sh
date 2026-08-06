#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=128gb
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

#python -m sunerf.run_thomson --config "config/cme/2026_04.yaml"
python -m sunerf.run_thomson --config "config/cme/2026_04_cme.yaml"


exit

#################################################
# Evaluation

python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v05/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v05/video" \
  --lon_frame hci \
  --lat -4.660   --lon 140.137 --time 2026-04-23T00:00 --steps 1  --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 140.137 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-23T21:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-24T06:00 --steps 20 --radius_min 3 --radius_max 20 \
  --lat -4.660   --lon 200 --time 2026-04-23T00:00 --steps 20 --radius_min 3 --radius_max 20 \


python -m sunerf.evaluation.cme.video \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v05/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v05/video_polar" \
  --lon_frame hci \
  --lat 80   --lon 0 --time 2026-04-01T00:00 --steps 1  --radius_min 3 --radius_max 20 \
  --lat 80   --lon 0 --time 2026-05-01T00:00 --steps 100 --radius_min 3 --radius_max 20

python -m sunerf.evaluation.cme.density_los_profile \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/save_state.snf" \
  --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/density_los_1948" \
  --time 2026-04-23T19:48 \
  --point A 3000 5000 \
  --point B 2000 5000 \
  --observer earth \
  --occ_min 3 \
  --occ_max 20


python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/cor2_clear" --ref_tB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04/prep/cor2_clear/*" --instrument_key "STEREO_A_COR2_CLEAR" --time_range "2026-04-23T14:00:00" "2026-04-24T00:00:00"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/cor2" --ref_pB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04/prep/cor2/pB/*" --ref_tB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04/prep/cor2/tB/*" --instrument_key "STEREO_A_COR2" --time_range "2026-04-23T14:00:00" "2026-04-24T00:00:00"
python -m sunerf.evaluation.cme.plot_ref_series --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/save_state.snf" --out_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v07/ccor" --ref_tB_path "/glade/campaign/hao/radmhd/rjarolim/SuNeRF_CME_OBS/2026_04/prep/ccor/*" --instrument_key "CCOR" --time_range "2026-04-23T14:00:00" "2026-04-24T00:00:00"


python -m sunerf.convert.sunerf_to_vtk --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/2026_04_v03/save_state.snf" --times "2026-04-23T17:00" "2026-04-23T18:00" "2026-04-23T19:00" "2026-04-23T20:00" "2026-04-23T21:00" --radius_range 3 20 --pixel_per_Rs 4

python -m sunerf.evaluation.image_sequence_to_video \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2026_04/video2 \
  /Users/rjarolim/PycharmProjects/SuNeRF/results/2026_04/video2.mp4 \
  --fps 10
