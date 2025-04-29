import argparse
import os
import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from sunerf.data.loader.thomson_instrument import ThomsonDataModule
from sunerf.model.sunerf import save_state
from sunerf.model.thomson import ThomsonSuNeRFModule
from sunerf.train.callback import ThomsonImageCallback, LatitudeSliceCallback, LongitudeSliceCallback, CubeCallback, \
    VelocitySliceCallback
from sunerf.train.util import load_yaml_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--reload', action='store_true')
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
    instruments = config['instruments']
    model_config = config['model'] if 'model' in config else {}
    sampling_config = config['sampling'] if 'sampling' in config else {}
    lambda_config = config['lambda'] if 'lambda' in config else {}
    module_config = config['module'] if 'module' in config else {}
    training_config = config['training'] if 'training' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 50
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    check_val_every_n_epoch = training_config[
        'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training_config else 1
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize data module and model
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')
    if os.path.exists(data_module_save_path) and not args.reload:
        print('Loaded data module from file. If you want to reload the data, use --reload')
        data_module = torch.load(data_module_save_path)
    else:
        warnings.filterwarnings("ignore")  # ignore warnings from sunpy
        data_module = ThomsonDataModule(**data_config, work_directory=work_directory)
        torch.save(data_module, data_module_save_path)

    image_scaling = list(data_module.config.values())[0]['image_scaling']
    rho_normalization = image_scaling / (8.69 * 1e-7)

    # initialize SuNeRF model
    sunerf = ThomsonSuNeRFModule(instruments=instruments,
                                 Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                 validation_dataset_mapping=data_module.validation_dataset_mapping,
                                 model_config=model_config,
                                 sampling_config=sampling_config, **module_config,
                                 lambda_config=lambda_config)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_state(sunerf, data_module, save_path))

    callbacks = [checkpoint_callback, save_callback]

    for callback_config in config.get('callbacks', []):
        ds_key = callback_config.pop('ds_key', None)
        callback_type = callback_config.pop('type')
        if callback_type.lower() == 'thomson_image':
            callback = ThomsonImageCallback(ds_key=ds_key,
                                            image_shape=data_module.validation_datasets[ds_key].image_shape)
        elif callback_type.lower() == 'latitude_slice':
            latitude = callback_config.get('latitude', 0)
            callback = LatitudeSliceCallback(ds_key=ds_key, latitude=latitude,
                                             cube_shape=data_module.validation_datasets[ds_key].cube_shape,
                                             rho_normalization=rho_normalization,
                                             Rs_per_ds=data_module.Rs_per_ds,
                                             seconds_per_dt=data_module.seconds_per_dt)
        elif callback_type.lower() == 'longitude_slice':
            longitude = callback_config.get('longitude', 0)
            callback = LongitudeSliceCallback(ds_key=ds_key, longitude=longitude,
                                              cube_shape=data_module.validation_datasets[ds_key].cube_shape,
                                              rho_normalization=rho_normalization,
                                              Rs_per_ds=data_module.Rs_per_ds,
                                              seconds_per_dt=data_module.seconds_per_dt)
        elif callback_type.lower() == 'cube':
            callback = CubeCallback(ds_key=ds_key,
                                    cube_shape=data_module.validation_datasets[ds_key].cube_shape,
                                    Rs_per_ds=data_module.Rs_per_ds,
                                    seconds_per_dt=data_module.seconds_per_dt)
        elif callback_type.lower() == 'velocity_slice':
            latitude = callback_config.get('latitude', 0)
            callback = VelocitySliceCallback(ds_key=ds_key, latitude=latitude,
                                             cube_shape=data_module.validation_datasets[ds_key].cube_shape,
                                             rho_normalization=rho_normalization,
                                             Rs_per_ds=data_module.Rs_per_ds,
                                             seconds_per_dt=data_module.seconds_per_dt)
        else:
            raise ValueError(f'Unknown callback type {callback_type}')
        callbacks.append(callback)

    N_GPUS = torch.cuda.device_count()
    torch.set_float32_matmul_precision('high')

    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy='dp' if N_GPUS > 1 else None,  # ddp breaks memory and wandb
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
