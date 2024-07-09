# Training from scratch
# init workspace
conda activate sunerf
cd /home/rjarolim/projects/SuNeRF
# convert data (pre-training with center crop)
python -m sunerf.data.prep.sdo --sdo_file_path "/mnt/data/nerf_data/sdo_2012_08/1h_193/*.fits" --output_path "/mnt/data/workspace/prep_2012_08/193" --center_crop
python -m sunerf.data.prep.stereo --stereo_file_path "/mnt/data/nerf_data/stereo_2012_08_converted_fov/195/*.fits" --output_path "/mnt/data/workspace/prep_2012_08/193" --center_crop
# training step for 1 epoch
python -m sunerf.run_emission --config "config/emission_2012_08-193.yaml"
# convert data (full training without center crop)
# clear previous data
rm -r "/mnt/data/workspace/prep_2012_08/193"
python -m sunerf.prep.prep_sdo --sdo_file_path "/mnt/data/nerf_data/sdo_2012_08/1h_193/*.fits" --output_path "/mnt/data/workspace/prep_2012_08/193"
python -m sunerf.prep.prep_stereo --stereo_file_path "/mnt/data/nerf_data/stereo_2012_08_converted_fov/195/*.fits" --output_path "/mnt/data/workspace/prep_2012_08/193"
# full training
python -m sunerf.sunerf --wandb_project "sunerf" --wandb_name "193" --resume_from_checkpoint "/mnt/data/runs/193_crop/last.ckpt" --data_path "/mnt/data/workspace/prep_2012_08/193/*.fits" --path_to_save "/mnt/data/runs/193_v01" --train "config/dgx_train.yaml" --hyperparameters "config/hyperparameters.yaml"

