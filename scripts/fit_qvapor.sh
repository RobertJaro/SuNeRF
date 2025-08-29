#!/bin/bash -l

#PBS -N SuNeRF
#PBS -A P22100000
#PBS -q main
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
#PBS -l job_priority=economy
#PBS -l walltime=12:00:00

module load conda/latest
module load cuda/11.7.1
conda activate lightning

cd /glade/u/home/rjarolim/projects/SuNeRF

python -m sunerf.train.fit_qvapor --data_path "/glade/work/rjarolim/data/qvapor_tomography" --work_directory "/glade/derecho/scratch/rjarolim/nearthfs/qvapor" --out_path "/glade/work/rjarolim/nearthfs/qvapor"

python -m sunerf.evaluation.nearthf_loader
python -m sunerf.run_water --config "config/water/qvapor.yaml" --reload



