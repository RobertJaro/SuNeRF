#!/bin/bash -l

#PBS -N SuNeRF-mass-transport
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=04:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

# Mass-distance-time diagram, three-part front kinematics and drag of the
# synthetic reconstructions, compared with the ground-truth density cubes.
# No pre-event background is subtracted (see sunerf/evaluation/cme/mass_transport.py).

########### HAO 2010-04-03 CME (frames 10-40 of the density series) ###########

for run in 2view_v02 3view_v01 all_v01; do
python -m sunerf.evaluation.cme.mass_transport_evaluation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/${run}/save_state.snf" \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/hao/density_cube_v2/*.sav" \
  --dataset hao --date0 "2010-04-03T09:04:00.000" --frames 10:40 \
  --min_radius 30 --max_radius 120 --min_latitude -60 --max_latitude 60 \
  --min_longitude 70 --max_longitude 200 \
  --target_latitude 0 --target_longitude 135 --cone_half_angle 30
done

########### HAO CME1 / CME2 (2010-05-27) ###########

for run in cme1_2view_v01 cme1_3view_v01 cme1_4view_v01; do
python -m sunerf.evaluation.cme.mass_transport_evaluation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/${run}/save_state.snf" \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/cme1/density_cube/*.sav" \
  --dataset hao --date0 "2010-05-27T07:04:39.000" \
  --min_radius 30 --max_radius 120 --min_latitude -70 --max_latitude 30 \
  --min_longitude 0 --max_longitude 180 \
  --target_latitude -25 --target_longitude 60 --cone_half_angle 30
done

for run in cme2_2view_v01 cme2_3view_v01 cme2_4view_v01; do
python -m sunerf.evaluation.cme.mass_transport_evaluation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-v2/${run}/save_state.snf" \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/cme2/density_cube/*.sav" \
  --dataset hao --date0 "2010-05-27T07:04:39.000" \
  --min_radius 30 --max_radius 120 --min_latitude -20 --max_latitude 80 \
  --min_longitude 0 --max_longitude 180 \
  --target_latitude 35 --target_longitude 60 --cone_half_angle 30
done

########### PSI 2021-10-28 CME (145 dumps at 5 min; cone around lon 245, lat -20) ###########

for run in clean_2view_v01 clean_3view_v01 degraded_2view_v01 degraded_3view_v01; do
python -m sunerf.evaluation.cme.mass_transport_evaluation \
  --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/psi_obs/${run}/save_state.snf" \
  --data_path "/glade/work/rjarolim/data/sunerf-cme/psi_obs/rho/rho*.hdf" \
  --dataset psi --dump_times "/glade/work/rjarolim/data/sunerf-cme/psi_obs/dump_times.json" \
  --frames 12::3 --stride 2 \
  --min_radius 2 --max_radius 30 --min_latitude -70 --max_latitude 30 \
  --min_longitude 180 --max_longitude 310 \
  --target_latitude -20 --target_longitude 245 --cone_half_angle 30 \
  --dr 0.25 --smooth 0.5 --r_search_min 3
done
