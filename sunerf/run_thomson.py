import argparse
import glob
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LambdaCallback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from sunerf.data.loader.thomson_instrument import ThomsonDataModule
from sunerf.model.thomson import ThomsonSuNeRFModule, save_thomson_sunerf
from sunerf.physics.thomson import MSB, SIGMA_NE, electron_density_normalization_cm3
from sunerf.train.callback import ThomsonImageCallback, LatitudeSliceCallback, LongitudeSliceCallback, CubeCallback, \
    VelocitySliceCallback, CorrectionImageCallback, FullStarBackgroundCallback, RadialSlicesCallback, \
    LongitudeTimeVelocityMagCallback, FixedViewpointSeriesCallback, \
    LongitudeSlicesCallback, InSituTimeSeriesCallback
from sunerf.train.util import load_yaml_config


DATA_CACHE_FORMAT_VERSION = 3


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


def _cache_json_default(value):
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _cache_relevant_config(value):
    """Drop runtime-only options which do not change cached array contents."""
    if isinstance(value, dict):
        return {
            key: _cache_relevant_config(item)
            for key, item in value.items()
            if key not in {'num_workers', 'preprocess_workers'}
        }
    if isinstance(value, list):
        return [_cache_relevant_config(item) for item in value]
    return value


def _configured_source_files(value, key=''):
    """Collect input files referenced by path/file entries in the data config."""
    if isinstance(value, dict):
        files = []
        for child_key, child_value in value.items():
            files.extend(_configured_source_files(child_value, str(child_key)))
        return files
    if isinstance(value, (list, tuple)):
        files = []
        for child_value in value:
            files.extend(_configured_source_files(child_value, key))
        return files
    if not isinstance(value, str):
        return []

    expanded_value = os.path.expanduser(value)
    is_path_key = any(token in key.lower() for token in ('path', 'file'))
    if not is_path_key and not glob.has_magic(expanded_value) and not os.path.isfile(expanded_value):
        return []
    matches = glob.glob(expanded_value)
    if not matches and os.path.isfile(expanded_value):
        matches = [expanded_value]
    return [os.path.abspath(path) for path in matches if os.path.isfile(path)]


def build_data_cache_fingerprint(data_config):
    """Fingerprint cache semantics plus source path, size, and modification time."""
    source_records = []
    for path in sorted(set(_configured_source_files(data_config))):
        stat = os.stat(path)
        source_records.append({
            'path': path,
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    payload = {
        'format_version': DATA_CACHE_FORMAT_VERSION,
        'config': _cache_relevant_config(data_config),
        'sources': source_records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=_cache_json_default).encode()
    return hashlib.sha256(encoded).hexdigest(), source_records


def _atomic_torch_save(value, destination):
    temporary_path = f'{destination}.tmp-{uuid.uuid4().hex}'
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def _atomic_json_save(value, destination):
    temporary_path = f'{destination}.tmp-{uuid.uuid4().hex}'
    try:
        with open(temporary_path, 'w') as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def _is_cache_generation(path, cache_root):
    if not path:
        return False
    real_path = os.path.realpath(path)
    real_root = os.path.realpath(cache_root)
    return (
        os.path.dirname(real_path) == real_root
        and os.path.basename(real_path).startswith('generation-')
    )


def _data_module_cache_files(data_module):
    files = set()
    datasets = (
        *getattr(data_module, 'training_datasets', {}).values(),
        *getattr(data_module, 'validation_datasets', {}).values(),
    )
    for dataset in datasets:
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        files.update(os.path.abspath(path) for path in getattr(dataset, 'batches_file_paths', {}).values())
    return sorted(files)


def _data_cache_is_usable(data_module, fingerprint, reload_token=None):
    """Validate provenance and every mmap file before accepting a cache."""
    if data_module is None or getattr(data_module, 'cache_fingerprint', None) != fingerprint:
        return False
    if reload_token is not None and getattr(data_module, 'cache_reload_token', None) != reload_token:
        return False
    cache_files = getattr(data_module, 'cache_files', None)
    if cache_files is None:
        cache_files = _data_module_cache_files(data_module)
    return all(os.path.isfile(path) for path in cache_files)


def _remove_cache_generation(path, cache_root):
    """Delete only a UUID generation directory created by this cache manager."""
    if _is_cache_generation(path, cache_root):
        shutil.rmtree(path, ignore_errors=True)


def _wait_for_data_module(path, fingerprint, reload_token=None):
    """Wait for rank zero's atomic cache publication and validate its manifest."""
    timeout = float(os.environ.get('SUNERF_CACHE_WAIT_SECONDS', 6 * 60 * 60))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            data_module = torch.load(path, weights_only=False)
            if _data_cache_is_usable(data_module, fingerprint, reload_token=reload_token):
                return data_module
            last_error = RuntimeError(
                'cache fingerprint does not match or one of its generated files is missing'
            )
        except (FileNotFoundError, EOFError, OSError, RuntimeError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f'Timed out waiting for data cache {path}: {last_error}')


def cache_reload_token(enabled):
    """Return one reload publication token shared by all ranks in this launch."""
    if not enabled:
        return None
    existing = os.environ.get('SUNERF_CACHE_RELOAD_TOKEN')
    if existing:
        return existing

    # torchrun provides TORCHELASTIC_RUN_ID. Lightning's subprocess launcher
    # instead inherits the UUID placed in the parent environment below.
    shared_launch_id = os.environ.get('TORCHELASTIC_RUN_ID')
    if not shared_launch_id and ('RANK' in os.environ or 'LOCAL_RANK' in os.environ):
        shared_launch_id = ':'.join([
            os.environ.get('SLURM_JOB_ID', os.environ.get('PBS_JOBID', 'external')),
            os.environ.get('MASTER_ADDR', 'localhost'),
            os.environ.get('MASTER_PORT', 'unknown'),
            os.environ.get('WORLD_SIZE', 'unknown'),
        ])
    token = shared_launch_id or uuid.uuid4().hex
    os.environ['SUNERF_CACHE_RELOAD_TOKEN'] = token
    return token


@rank_zero_only
def _save_thomson_sunerf_rank_zero(*args, **kwargs):
    """Write the custom inference artifact from one distributed rank only."""
    return save_thomson_sunerf(*args, **kwargs)


def trainer_device_config(cuda_device_count):
    """Return a valid Lightning accelerator/device pair for GPU or CPU hosts."""
    cuda_device_count = int(cuda_device_count)
    if cuda_device_count > 0:
        return 'gpu', cuda_device_count
    return 'cpu', 1


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

    # initialize logger
    logger = WandbLogger(**logging_config, save_dir=work_directory)


    @rank_zero_only
    def _log_hparams(cfg):
        logger.log_hyperparams(cfg)


    _log_hparams(config)

    # initialize data module and model
    data_module_save_path = os.path.join(work_directory, 'data_module.pkl')
    data_manifest_path = os.path.join(work_directory, 'data_cache_manifest.json')
    data_cache_root = os.path.join(work_directory, '.sunerf_cache')
    cache_fingerprint, source_records = build_data_cache_fingerprint(data_config)
    reload_token = cache_reload_token(args.reload)


    @rank_zero_only
    def _load_data_module():
        old_data_module = None
        if os.path.exists(data_module_save_path):
            try:
                old_data_module = torch.load(data_module_save_path, weights_only=False)
            except (EOFError, OSError, RuntimeError) as error:
                warnings.warn(f'Ignoring unreadable data cache: {error}')
            if (
                not args.reload
                and old_data_module is not None
                and _data_cache_is_usable(old_data_module, cache_fingerprint)
            ):
                print('Loaded validated data cache. Use --reload to rebuild it explicitly.')
                return

        os.makedirs(data_cache_root, exist_ok=True)
        generation_directory = os.path.join(data_cache_root, f'generation-{uuid.uuid4().hex}')
        os.makedirs(generation_directory)
        warnings.filterwarnings("ignore")  # ignore warnings from sunpy
        data_module = None
        try:
            data_module = ThomsonDataModule(**data_config, work_directory=generation_directory)
            data_module.cache_fingerprint = cache_fingerprint
            data_module.cache_format_version = DATA_CACHE_FORMAT_VERSION
            data_module.cache_generation_directory = generation_directory
            data_module.cache_files = _data_module_cache_files(data_module)
            data_module.cache_reload_token = reload_token
            _atomic_torch_save(data_module, data_module_save_path)
            try:
                _atomic_json_save({
                    'format_version': DATA_CACHE_FORMAT_VERSION,
                    'fingerprint': cache_fingerprint,
                    'generation_directory': generation_directory,
                    'cache_files': data_module.cache_files,
                    'reload_token': reload_token,
                    'sources': source_records,
                }, data_manifest_path)
            except OSError as error:
                # The pickle contains the same provenance and is already
                # atomically published, so a sidecar failure is non-fatal.
                warnings.warn(f'Could not write data-cache manifest: {error}')
        except Exception:
            if data_module is not None:
                data_module.clear()
            _remove_cache_generation(generation_directory, data_cache_root)
            raise

        # Publish the new pickle first. Only after it is durable is it safe to
        # remove the prior generation recorded by the old data module.
        if old_data_module is not None:
            old_generation = getattr(old_data_module, 'cache_generation_directory', None)
            # Legacy caches outside our generation root are deliberately left
            # untouched; only directories created and recorded by this manager
            # are eligible for recursive cleanup.
            _remove_cache_generation(old_generation, data_cache_root)


    _load_data_module()  # ensure only rank 0 loads/saves the data module
    data_module = _wait_for_data_module(
        data_module_save_path,
        cache_fingerprint,
        reload_token=reload_token,
    )
    # Worker count is runtime state and deliberately excluded from the expensive
    # array-cache fingerprint.
    if 'num_workers' in data_config:
        data_module.num_workers = data_config['num_workers']

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
                drho_cm3=drho_cm3,
                Rs_per_ds=data_module.Rs_per_ds,
                seconds_per_dt=data_module.seconds_per_dt,
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

    torch.set_float32_matmul_precision('high')

    n_gpus = torch.cuda.device_count()
    accelerator, devices = trainer_device_config(n_gpus)
    trainer = Trainer(max_epochs=epochs,
                      logger=logger,
                      devices=devices,
                      accelerator=accelerator,
                      strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else 'auto',
                      num_sanity_val_steps=0,  # validate all points to check the first image
                      val_check_interval=log_every_n_steps,
                      check_val_every_n_epoch=check_val_every_n_epoch,
                      gradient_clip_val=0.5,
                      callbacks=callbacks)

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
    # Validation cadence need not coincide with the final optimizer step. Publish
    # one final inference artifact from rank zero so it cannot lag final.ckpt.
    _save_thomson_sunerf_rank_zero(
        sunerf, data_module, save_path,
        msb_norm=image_scaling, msb=MSB, sigma_ne=SIGMA_NE,
    )
