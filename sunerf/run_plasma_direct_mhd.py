import argparse
import os
import warnings

import torch
torch.set_float32_matmul_precision('medium')
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from sunerf.data.loader.mhd import PSIMHDDataModule
from sunerf.model.sunerf import save_state
from sunerf.model.sunerf_mhd import PlasmaSuNeRFModuleMHD
from sunerf.train.callback import PlasmaImageCallback, AbsorptionCallback
from sunerf.train.util import load_yaml_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args, overwrite_args = parser.parse_known_args()

    yaml_config_file = args.config
    config = load_yaml_config(yaml_config_file, overwrite_args)

    # setup paths
    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    work_directory = config['work_directory'] if 'work_directory' in config else base_path
    os.makedirs(work_directory, exist_ok=True)

    # setup default configs
    data_config = config['data']
    model_config = config['model'] if 'model' in config else {}
    training_config = config['training'] if 'training' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 1000
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize data module and model
    warnings.filterwarnings("ignore")  # ignore warnings from sunpy
    data_module = PSIMHDDataModule(**data_config, working_dir=work_directory) 

    # initialize SuNeRF model
    sunerf = PlasmaSuNeRFModuleMHD(Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                validation_dataset_mapping=data_module.validation_dataset_mapping,
                                **model_config)
    
    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    callbacks = [checkpoint_callback, save_callback]

    # for k in data_module.validation_dataset_mapping.values():
    #     if k == 'absorption':
    #         continue
    #     test_image_callback = PlasmaImageCallback(k, data_module.config[k]['image_shape'],
    #                                               cmaps=data_module.config[k]['cmaps'])
    #     callbacks.append(test_image_callback)

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
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
