# Training from scratch
# init workspace
conda activate sunerf
cd /home/rjarolim/projects/SuNeRF
# convert data
python -m sunerf.data.prep.psi --psi_path "/mnt/data/nerf_data/psi_data/193/*.fits" --output_path "/mnt/data/workspace/prep_psi/193"
python -m sunerf.run_emission --config "config/psi_193.yaml"