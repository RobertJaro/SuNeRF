#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=32:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda
module load cuda
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

########### Download ###########

#python -m sunerf.data.psi.download_psi_cme --observers L1 L4 L5 --out-dir /glade/work/rjarolim/data/psi_cme
#python -m sunerf.data.psi.download_psi_cme --observers P1 --out-dir /glade/work/rjarolim/data/psi_cme
#python -m sunerf.data.psi.download_psi_cme --observers R2 --out-dir /glade/work/rjarolim/data/psi_cme


########### Fix data ###########
#python -m sunerf.data.psi.prep_psi_cme --data_path "/glade/work/rjarolim/data/psi_cme/**/*.fts" --output_path "/glade/work/rjarolim/data/psi_cme_prep" --resolution 512 512


########### Fit scaling masks ###########
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/psi_cme_prep/L1/pb/*" \
#  --output "/glade/work/rjarolim/data/psi_cme_prep/masks/L1_pb_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/psi_cme_prep/masks/L1_pb.png" \
#  --degree 3
#
#python -m sunerf.data.coronagraph.fit_radial_profile \
#  --input "/glade/work/rjarolim/data/psi_cme_prep/L1/tb/*" \
#  --output "/glade/work/rjarolim/data/psi_cme_prep/masks/L1_tb_fit.npy" \
#  --plot-output "/glade/work/rjarolim/data/psi_cme_prep/masks/L1_tb.png" \
#  --degree 3

############ Train ###########
#python -m sunerf.run_thomson --config "config/cme/psi_cme_1view.yaml"
#python -m sunerf.run_thomson --config "config/cme/psi_cme_3view.yaml"
python -m sunerf.run_thomson --config "config/cme/psi_cme_3view_physics.yaml"
#python -m sunerf.run_thomson --config "config/cme/psi_cme_4view.yaml"

exit
######## Evaluation #####


python -m sunerf.evaluation.cme.video --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/psi_cme_3view_physics_v01/save_state.snf"

python -m sunerf.evaluation.cme.video_fixed_rotation --sunerf_path "/glade/work/rjarolim/sunerf-cme-obs/psi_cme_3view_physics_v01/save_state.snf" --latitude 20 --longitude -20 --occ_range 2 25
