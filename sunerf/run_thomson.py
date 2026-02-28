import argparse
import os
import warnings

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LambdaCallback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from sunerf.data.loader.thomson_instrument import ThomsonDataModule
from sunerf.model.thomson import ThomsonSuNeRFModule, save_thomson_sunerf
from sunerf.train.callback import ThomsonImageCallback, LatitudeSliceCallback, LongitudeSliceCallback, CubeCallback, \
    VelocitySliceCallback, CorrectionImageCallback, FullStarBackgroundCallback, RadialSlicesCallback, \
    LongitudeTimeVelocityMagCallback, FixedViewpointSeriesCallback, \
    LongitudeSlicesCallback
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
    shuffle_config = config['shuffle'] if 'shuffle' in config else None
    sampling_config = config['sampling'] if 'sampling' in config else {}
    lambda_config = config['lambda'] if 'lambda' in config else {}
    module_config = config['module'] if 'module' in config else {}
    training_config = config['training'] if 'training' in config else {}
    logging_config = config['logging'] if 'logging' in config else {'project': 'sunerf'}

    # setup training config
    epochs = training_config['epochs'] if 'epochs' in training_config else 200
    log_every_n_steps = training_config['log_every_n_steps'] if 'log_every_n_steps' in training_config else None
    check_val_every_n_epoch = training_config[
        'check_val_every_n_epoch'] if 'check_val_every_n_epoch' in training_config else 1
    ckpt_path = training_config['meta_path'] if 'meta_path' in training_config else 'last'

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)


    @rank_zero_only
    def _log_hparams(cfg):
        logger.log_hyperparams(cfg)


    _log_hparams(config)

    # initialize data module and model
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')


    @rank_zero_only
    def _load_data_module():
        if os.path.exists(data_module_save_path) and not args.reload:
            print('Loaded data module from file. If you want to reload the data, use --reload')
            return True
        warnings.filterwarnings("ignore")  # ignore warnings from sunpy
        data_module = ThomsonDataModule(**data_config, work_directory=work_directory)
        torch.save(data_module, data_module_save_path)


    _load_data_module()  # ensure only rank 0 loads/saves the data module
    data_module = torch.load(data_module_save_path)  # all ranks load the data module

    image_scaling = list(data_module.config.values())[0]['image_scaling']
    rho_normalization = image_scaling / (8.69 * 1e-7)

    # initialize SuNeRF model
    sunerf = ThomsonSuNeRFModule(instruments=instruments,
                                 Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                 validation_dataset_mapping=data_module.validation_dataset_mapping,
                                 model_config=model_config,
                                 sampling_config=sampling_config, **module_config,
                                 lambda_config=lambda_config, shuffle_config=shuffle_config)

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(on_validation_end=lambda *args: save_thomson_sunerf(sunerf, data_module, save_path))

    callbacks = [checkpoint_callback, save_callback]

    for cb_cfg in config.get('callbacks', []):
        cb_cfg = dict(cb_cfg)  # avoid mutating config
        ds_key = cb_cfg.pop('ds_key', None)
        cb_type = cb_cfg.pop("type").lower()

        if ds_key is None:
            raise ValueError(f"Callback '{cb_type}' is missing 'ds_key'")

        ds = data_module.validation_datasets[ds_key]
        # NOTE: ds is wrapped. We need to access the base dataset for shapes/meta.
        base = getattr(ds, "dataset", ds)  # RenderModeDataset stores base in `.dataset`

        if cb_type == "thomson_image":
            callback = ThomsonImageCallback(ds_key=ds_key, image_shape=base.image_shape)

        elif cb_type == "latitude_slice":
            callback = LatitudeSliceCallback(
                ds_key=ds_key,
                latitude=cb_cfg.get("latitude", 0),
                cube_shape=base.cube_shape,
                rho_normalization=rho_normalization,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
            )

        elif cb_type == "longitude_slice":
            callback = LongitudeSliceCallback(
                ds_key=ds_key,
                longitude=cb_cfg.get("longitude", 0),
                cube_shape=base.cube_shape,
                rho_normalization=rho_normalization,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
            )

        elif cb_type == "cube":
            callback = CubeCallback(
                ds_key=ds_key,
                cube_shape=base.cube_shape,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
            )

        elif cb_type == "velocity_slice":
            callback = VelocitySliceCallback(
                ds_key=ds_key,
                latitude=cb_cfg.get("latitude", 0),
                cube_shape=base.cube_shape,
                rho_normalization=rho_normalization,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
            )

        elif cb_type == "correction_image":
            callback = CorrectionImageCallback(ds_key=ds_key, image_shape=base.image_shape)

        # -----------------------------
        # NEW: radial slices (configurable radii)
        # -----------------------------
        elif cb_type == "radial_slices":
            # expects base.cube_shape = (Nr, Ntheta, Nphi, Nt)
            callback = RadialSlicesCallback(
                ds_key=ds_key,
                cube_shape=base.cube_shape,
                radii=base.radii,
                name=cb_cfg.get("name", ds_key),
            )

        # -----------------------------
        # NEW: longitude slices over time (density)
        # shared dataset with velocity callback
        # expects base.cube_shape = (Nlon, Nt, Nr, Ntheta)
        # -----------------------------
        elif cb_type == "longitude_density":
            callback = LongitudeSlicesCallback(
                ds_key=ds_key,
                cube_shape=base.cube_shape,
                longitude_deg=base.longitude_deg,
                name=cb_cfg.get("name", ds_key),
            )

        # -----------------------------
        # NEW: longitude slices over time (velocity magnitude)
        # shared dataset with density callback
        # -----------------------------
        elif cb_type == "longitude_time_velocitymag":
            callback = LongitudeTimeVelocityMagCallback(
                ds_key=ds_key,
                cube_shape=base.cube_shape,
                name=cb_cfg.get("name", ds_key),
            )

        # -----------------------------
        # NEW: fixed viewpoint series (3 rows: tB, pB, density; 6 cols over time)
        # expects base.image_shape and base.n_times
        # -----------------------------
        elif cb_type == "fixed_viewpoint_series":
            n_times = getattr(base, "n_times", cb_cfg.get("n_times", 6))
            callback = FixedViewpointSeriesCallback(
                ds_key=ds_key,
                image_shape=base.image_shape,
                n_times=n_times,
                name=cb_cfg.get("name", ds_key),
            )

        # -----------------------------
        # Existing: star background (log scale)
        # NOTE: you called it star_background_full; keep name for backward compat
        # expects base.image_shape (recommended) OR base.sky_shape (if you kept that naming)
        # -----------------------------
        elif cb_type in ("star_background_full", "full_star_background"):
            # Prefer image_shape (H,W) since your FullStarBackgroundDataset sets that.
            if hasattr(base, "image_shape"):
                image_shape = base.image_shape
                callback = FullStarBackgroundCallback(
                    ds_key=ds_key,
                    image_shape=image_shape,
                    eps=cb_cfg.get("eps", 1e-12),
                    name=cb_cfg.get("name", ds_key),
                )
            else:
                # fallback if you used sky_shape naming previously
                callback = FullStarBackgroundCallback(
                    ds_key=ds_key,
                    image_shape=base.sky_shape,
                    eps=cb_cfg.get("eps", 1e-12),
                    name=cb_cfg.get("name", ds_key),
                )

        else:
            raise ValueError(f"Unknown callback type '{cb_type}'")

        callbacks.append(callback)

    N_GPUS = torch.cuda.device_count()
    torch.set_float32_matmul_precision('high')

    n_gpus = torch.cuda.device_count()
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=N_GPUS,
                      accelerator='gpu' if N_GPUS >= 1 else None,
                      strategy=DDPStrategy(find_unused_parameters=False) if n_gpus > 1 else 'auto',
                      num_sanity_val_steps=-1,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
