import argparse
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from sunerf.data.loader.single_channel import SingleChannelDataModule
from sunerf.model.sunerf import save_state, EmissionSuNeRFModule
from sunerf.train.callback import TestImageCallback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    # setup paths
    path_to_save = config['path_to_save']
    os.makedirs(path_to_save, exist_ok=True)
    working_dir = config['working_directory'] if 'working_directory' in config else path_to_save
    os.makedirs(working_dir, exist_ok=True)

    # setup default configs
    data_config = config['data']
    model_config = config['model'] if 'model' in config else {}
    training_config = config['training'] if 'training' in config else {}
    image_scaling_config = config['image_scaling'] if 'image_scaling' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 100
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=working_dir)

    # initialize data module and model
    data_module = SingleChannelDataModule(**data_config, working_dir=working_dir)

    # initialize SuNeRF model
    sunerf = EmissionSuNeRFModule(Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                  image_scaling_config=image_scaling_config,
                                  validation_dataset_mapping=data_module.validation_dataset_mapping,
                                  **model_config)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=path_to_save,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(path_to_save, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    test_image_callback = TestImageCallback(data_module.validation_dataset_mapping[0],
                                            data_module.config['resolution'],
                                            cmap=data_module.config['cmap'])
    callbacks = [checkpoint_callback, save_callback, test_image_callback]

    N_GPUS = torch.cuda.device_count()
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(path_to_save, 'final.ckpt'))
