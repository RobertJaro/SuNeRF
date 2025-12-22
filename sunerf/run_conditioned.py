import argparse
import os

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from sunerf.data.loader.conditioned import ConditionedDataModule
from sunerf.model.conditioned import ConditionedSuNeRFModule
from sunerf.train.callback import ConditionedImageCallback


def save_conditioned_sunerf(sunerf, data_module, save_path):
    r"""Save the SuNeRF model along with the data module.

    Args:
        sunerf_module: The SuNeRF model to be saved.
        data_module: The data module associated with the model.
        save_path: Path to save the model and data module.
    """
    output_path = os.path.dirname(save_path)
    os.makedirs(output_path, exist_ok=True)
    state = {
        # sunerf  rendering module
        'rendering': sunerf.rendering,
        # data infor
        'data_config': data_module.config,
        # data scaling
        'Rs_per_ds': data_module.Rs_per_ds,
    }
    torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    # setup paths
    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    work_directory = config['work_directory'] if 'work_directory' in config else base_path
    os.makedirs(work_directory, exist_ok=True)

    # setup default configs
    data_config = config['data']
    model_config = config['model'] if 'model' in config else {}
    sampling_config = config['sampling'] if 'sampling' in config else {}
    training_config = config['training'] if 'training' in config else {}
    image_scaling_config = config['image_scaling'] if 'image_scaling' in config else {}
    lambda_config = config['lambda'] if 'lambda' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else int(1e8)
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    check_val_every_n_epoch = training_config['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training_config else None
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)

    # initialize data module
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')


    @rank_zero_only
    def _init_data_module():
        data_module = ConditionedDataModule(**data_config, work_directory=work_directory)
        torch.save(data_module, data_module_save_path)


    _init_data_module()
    # load data module for all ranks
    data_module = torch.load(data_module_save_path, weights_only=False)

    # initialize SuNeRF model
    sunerf = ConditionedSuNeRFModule(Rs_per_ds=data_module.Rs_per_ds,
                                     image_scaling_config=image_scaling_config,
                                     validation_dataset_mapping=data_module.validation_dataset_mapping,
                                     in_channels=data_module.config['channels'], sampling_config=sampling_config,
                                     model_config=model_config, lambda_config=lambda_config,)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps,
                                          every_n_epochs=check_val_every_n_epoch)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(
        on_validation_end=lambda *args: save_conditioned_sunerf(sunerf, data_module, save_path))

    callbacks = [checkpoint_callback, save_callback]
    for k, v in data_module.validation_dataset_mapping.items():
        test_image_callback = ConditionedImageCallback(v, data_module.config['resolution'],
                                                       cmap=data_module.config['cmap'])
        callbacks.append(test_image_callback)

    torch.set_float32_matmul_precision('medium')  # for A100 GPUs
    n_gpus = torch.cuda.device_count()
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=n_gpus,
                      accelerator='gpu' if n_gpus >= 1 else None,
                      strategy=DDPStrategy(find_unused_parameters=False) if n_gpus > 1 else 'auto',
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      val_check_interval=log_every_n_steps,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
