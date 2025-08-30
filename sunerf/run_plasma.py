import argparse
import os
import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger

from sunerf.data.loader.multi_instrument import MultiInstrumentDataModule
from sunerf.model.plasma import PlasmaSuNeRFModule, save_plasma_sunerf
from sunerf.train.callback import PlasmaImageCallback, AbsorptionCallback
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
    instruments_config = config['instruments']
    data_config = config['data']
    model_config = config['model'] if 'model' in config else {'encoding': 'gaussian'}
    sampling_config = config['sampling'] if 'sampling' in config else {}
    training_config = config['training'] if 'training' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}
    shuffle_config = config['shuffle'] if 'shuffle' in config else {}
    lambda_config = config['lambda'] if 'lambda' in config else {}
    absorption_config = config['absorption'] if 'absorption' in config else {'type': 'learned'}

    # absorption config
    use_absorption = 'type' in absorption_config and absorption_config['type'] is not None

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 1000
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)
    logger.experiment.config.update(config, allow_val_change=True)

    # initialize data module and model
    # initialize data module and model
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')
    if os.path.exists(data_module_save_path) and not args.reload:
        print('Loaded data module from file. If you want to reload the data, use --reload')
        data_module = torch.load(data_module_save_path)
        # update batch size
        default_batch_size = data_config['batch_size']
        train_ds_config = data_config['train_datasets']
        ds_batch_size = {config['key']: config.get('batch_size', default_batch_size) for config in train_ds_config}
        for ds_key, ds in data_module.training_datasets.items():
           ds.batch_size = ds_batch_size[ds_key]
    else:
        warnings.filterwarnings("ignore")  # ignore warnings from sunpy
        data_module = MultiInstrumentDataModule(**data_config, work_directory=work_directory, use_absorption=use_absorption)
        torch.save(data_module, data_module_save_path)

    # initialize SuNeRF model
    sunerf = PlasmaSuNeRFModule(Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                validation_dataset_mapping=data_module.validation_dataset_mapping,
                                instruments_config=instruments_config, model_config=model_config,
                                sampling_config=sampling_config, shuffle_config=shuffle_config,
                                lambda_config=lambda_config, absorption_config=absorption_config)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_plasma_sunerf(sunerf, data_module, save_path))

    callbacks = [checkpoint_callback, save_callback]
    if use_absorption:
        absorption_callback = AbsorptionCallback('absorption', data_module.validation_datasets['absorption'].image_shape)
        callbacks.append(absorption_callback)


    for k in data_module.validation_dataset_mapping.values():
        if k == 'absorption':
            continue
        test_image_callback = PlasmaImageCallback(k, data_module.config[k]['image_shape'],
                                                  cmaps=data_module.config[k]['cmaps'])
        callbacks.append(test_image_callback)

    N_GPUS = torch.cuda.device_count()
    torch.set_float32_matmul_precision('medium')  # set precision for matmul
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
