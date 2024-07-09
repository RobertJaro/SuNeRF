# Training from scratch
# init workspace
conda activate sunerf
cd /home/rjarolim/projects/SuNeRF
# convert data (pre-training with center crop)
python -m sunerf.data.prep.psi --psi_path "/mnt/data/nerf_data/psi_data/193/*.fits" --output_path "/mnt/data/workspace/prep_psi/193"
# training step for 1 epoch
python -m sunerf.run_emission --config "config/psi-193.yaml"