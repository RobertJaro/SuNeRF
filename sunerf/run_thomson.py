import argparse
import os
import warnings

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LambdaCallback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from sunerf.data.loader.thomson_instrument import ThomsonDataModule
from sunerf.model.thomson import ThomsonSuNeRFModule, save_thomson_sunerf
from sunerf.physics.thomson import MSB, SIGMA_NE, electron_density_normalization_cm3
from sunerf.train.callback import ThomsonImageCallback, LatitudeSliceCallback, LongitudeSliceCallback, CubeCallback, \
    VelocitySliceCallback, CorrectionImageCallback, FullStarBackgroundCallback, RadialSlicesCallback, \
    LongitudeTimeVelocityMagCallback, FixedViewpointSeriesCallback, \
    LongitudeSlicesCallback, InSituTimeSeriesCallback
from sunerf.train.runtime import (
    _data_cache_is_usable,
    _data_module_cache_files,
    _is_cache_generation,
    build_data_cache_fingerprint,
    build_trainer,
    cache_reload_token,
    prepare_data_module,
    start_wandb_logger,
    trainer_device_config,
)
from sunerf.train.util import load_yaml_config


__all__ = [
    "_data_cache_is_usable",
    "_data_module_cache_files",
    "_is_cache_generation",
    "build_data_cache_fingerprint",
    "cache_reload_token",
    "trainer_device_config",
]


def _load_stage_initial_weights(sunerf, state_dict):
    """Load model weights while retaining loss schedules from the new stage config."""
    lambda_schedule_state = {
        key: value.detach().clone()
        for key, value in sunerf.lambdas.state_dict().items()
    }
    # Schedule layouts can change between stages (for example, exponential to
    # step), so exclude prior lambda buffers from the strict model-weight load.
    stage_state_dict = state_dict.copy()
    if hasattr(state_dict, '_metadata'):
        stage_state_dict._metadata = state_dict._metadata
    for key in tuple(stage_state_dict):
        if key.startswith('lambdas.'):
            stage_state_dict.pop(key)
    try:
        return sunerf.load_state_dict(stage_state_dict, strict=True)
    finally:
        sunerf.lambdas.load_state_dict(lambda_schedule_state, strict=True)
        sunerf._set_lambda_schedule_step(0)


@rank_zero_only
def _save_thomson_sunerf_rank_zero(*args, **kwargs):
    """Write the custom inference artifact from one distributed rank only."""
    return save_thomson_sunerf(*args, **kwargs)


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
    init_path = training_config.get('init_path')
    ignore_unexpected_state_prefixes = tuple(training_config.get('ignore_unexpected_state_prefixes', ()))
    ckpt_path = training_config.get('meta_path', None if init_path else 'last')
    if init_path is not None and 'meta_path' in training_config:
        raise ValueError("training.init_path and training.meta_path are mutually exclusive.")
    physics_update_interval = training_config.get('physics_update_interval', 1)

    # Start the W&B run on rank zero first; data preparation logs overviews.
    logger = start_wandb_logger(logging_config, work_directory, hparams=config)

    # Both physics pipelines use the same fingerprinted, atomically published
    # cache manager: rank zero builds, every rank loads the published module.
    data_module = prepare_data_module(
        work_directory,
        data_config,
        lambda generation: ThomsonDataModule(
            **data_config,
            work_directory=generation,
        ),
        reload=args.reload,
    )

    stale_scaling_datasets = []
    for config_group, loaded_datasets in (
        (data_config.get('train_datasets', []), data_module.training_datasets),
        (data_config.get('valid_datasets', []), data_module.validation_datasets),
    ):
        for dataset_config in config_group:
            expected_scaling_config = dataset_config.get('scaling_mask_config')
            if expected_scaling_config is None:
                continue
            dataset_key = dataset_config.get('key', dataset_config.get('type'))
            loaded_dataset = loaded_datasets.get(dataset_key)
            loaded_dataset = getattr(loaded_dataset, 'dataset', loaded_dataset)
            actual_scaling_config = getattr(loaded_dataset, 'data_config', {}).get('scaling_mask_config')
            has_scaling_mask = 'scaling_mask' in getattr(loaded_dataset, 'batches_file_paths', {})
            if actual_scaling_config != expected_scaling_config or not has_scaling_mask:
                stale_scaling_datasets.append(dataset_key)
    if stale_scaling_datasets:
        stale_keys = ', '.join(sorted(set(stale_scaling_datasets)))
        raise RuntimeError(
            f"Loaded data_module.pkl has stale or missing radial scaling for: {stale_keys}. "
            "Rerun with --reload to rebuild the normalized images and overview plots."
        )

    if not hasattr(data_module, "drho_cm3") or data_module.drho_cm3 is None:
        raise RuntimeError(
            "Loaded data module does not define drho_cm3. "
            "Remove the cached data_module.pkl or rerun with --reload."
        )

    # initialize SuNeRF model
    sunerf = ThomsonSuNeRFModule(instruments=instruments,
                                 Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
                                 validation_dataset_mapping=data_module.validation_dataset_mapping,
                                 model_config=model_config,
                                 sampling_config=sampling_config, **module_config,
                                 lambda_config=lambda_config, shuffle_config=shuffle_config,
                                 physics_update_interval=physics_update_interval)

    # Initialize a new training stage from model weights without restoring the
    # previous optimizer, scheduler, epoch, or global-step state.
    if init_path is not None:
        schedule_attributes = ('alpha_max', 'cold_steps', 'warm_steps')
        temporal_model = getattr(sunerf.model, 'model', None)
        schedule_config = {
            name: getattr(temporal_model, name).detach().clone()
            for name in schedule_attributes
            if temporal_model is not None and hasattr(temporal_model, name)
        }
        checkpoint = torch.load(init_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if ignore_unexpected_state_prefixes:
            ignored_keys = [
                key for key in state_dict
                if key.startswith(ignore_unexpected_state_prefixes)
            ]
            for key in ignored_keys:
                state_dict.pop(key)
            if ignored_keys:
                warnings.warn(
                    "Ignoring configured legacy checkpoint keys: " + ", ".join(sorted(ignored_keys))
                )
        _load_stage_initial_weights(sunerf, state_dict)
        # Schedule buffers are stage configuration, not learned weights. Keep
        # the values from this YAML and initialize their derived state at step 0.
        for name, value in schedule_config.items():
            getattr(temporal_model, name).copy_(value)
        sunerf.model.step(0)

    image_scaling = list(data_module.config.values())[0]['image_scaling']
    drho_cm3 = data_module.drho_cm3
    expected_drho_cm3 = electron_density_normalization_cm3(image_scaling, data_module.Rs_per_ds)
    if abs(float(drho_cm3) - expected_drho_cm3) > max(1e-6 * expected_drho_cm3, 1e-12):
        raise RuntimeError(
            "Loaded data_module.pkl has a stale Thomson density normalization. "
            "Rerun with --reload to rebuild it with the MSB-based normalization."
        )

    # initialize callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=base_path,
                                          save_last=True,
                                          every_n_train_steps=log_every_n_steps)
    save_path = os.path.join(base_path, 'save_state.snf')
    save_callback = LambdaCallback(
        on_validation_end=lambda *args: _save_thomson_sunerf_rank_zero(
            sunerf, data_module, save_path, msb_norm=image_scaling, msb=MSB, sigma_ne=SIGMA_NE
        )
    )

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
                drho_cm3=drho_cm3,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
            )

        elif cb_type == "longitude_slice":
            callback = LongitudeSliceCallback(
                ds_key=ds_key,
                longitude=cb_cfg.get("longitude", 0),
                cube_shape=base.cube_shape,
                drho_cm3=drho_cm3,
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
                drho_cm3=drho_cm3,
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
                drho_cm3=drho_cm3,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
                name=cb_cfg.get("name"),
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
                drho_cm3=drho_cm3,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
                longitude_deg=base.longitude_deg,
                name=cb_cfg.get("name"),
            )

        # -----------------------------
        # NEW: longitude slices over time (velocity magnitude)
        # shared dataset with density callback
        # -----------------------------
        elif cb_type == "longitude_time_velocitymag":
            callback = LongitudeTimeVelocityMagCallback(
                ds_key=ds_key,
                cube_shape=base.cube_shape,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
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

        elif cb_type == "insitu_timeseries":
            callback = InSituTimeSeriesCallback(
                ds_key=ds_key,
                drho_cm3=base.drho_cm3,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
                name=cb_cfg.get("name", ds_key),
            )

        elif cb_type == "full_star_background":
            callback = FullStarBackgroundCallback(
                ds_key=ds_key,
                image_shape=base.image_shape,
                name=cb_cfg.get("name", ds_key),
            )

        else:
            raise ValueError(f"Unknown callback type '{cb_type}'")

        callbacks.append(callback)

    trainer = build_trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=epochs,
        val_check_interval=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
    # Validation cadence need not coincide with the final optimizer step. Publish
    # one final inference artifact from rank zero so it cannot lag final.ckpt.
    _save_thomson_sunerf_rank_zero(
        sunerf, data_module, save_path,
        msb_norm=image_scaling, msb=MSB, sigma_ne=SIGMA_NE,
    )
