# Training from scratch
# init workspace
conda activate sunerf
cd /home/rjarolim/projects/SuNeRF
# convert data (pre-training with center crop)

python -i -m  sunerf.evaluation.video --chk_path "/mnt/data/runs/psi_193_v01/save_state.snf" --video_path "/mnt/data/runs/psi_193_v01/video"
