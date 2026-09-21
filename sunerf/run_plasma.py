import argparse
import copy
import os
from collections.abc import Mapping, Sequence

import numpy as np
from astropy import units as u
from lightning.pytorch.callbacks import LambdaCallback, ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from sunerf.configuration import canonical_channel_id, validate_plasma_config
from sunerf.data.euv.estimate_scaling import resolve_instrument_scaling
from sunerf.data.loader.multi_instrument import MultiInstrumentDataModule
from sunerf.model.plasma import PlasmaSuNeRFModule, save_plasma_sunerf
from sunerf.response import SENSITIVITY_CONVENTIONS, load_response_artifact
from sunerf.train.callback import AbsorptionCallback
from sunerf.train.euv_callback import EUVTomographyCallback
from sunerf.train.runtime import (
    build_data_cache_fingerprint,
    build_trainer,
    prepare_data_module,
    start_wandb_logger,
)
from sunerf.train.util import load_yaml_config


@rank_zero_only
def _save_plasma_sunerf_rank_zero(*args, **kwargs):
    return save_plasma_sunerf(*args, **kwargs)


def _configured_dataset_instruments(data_config):
    mapping = {}
    for group_name in ('train_datasets', 'valid_datasets'):
        for dataset in data_config.get(group_name, ()):  # schema validation guarantees mappings
            ds_key = dataset.get('key', dataset.get('type'))
            instrument_key = dataset.get('instrument_key', ds_key)
            previous = mapping.setdefault(ds_key, instrument_key)
            if previous != instrument_key:
                raise ValueError(
                    f"Dataset '{ds_key}' maps to both '{previous}' and '{instrument_key}'."
                )
    return mapping


def _expected_measurement_units(renderer, response_config, n_channels):
    explicit = response_config.get('measurement_unit')
    if explicit is None:
        provenance = getattr(renderer, 'response_provenance', {})
        explicit = provenance.get('measurement_unit', provenance.get('output_unit'))
    if explicit is not None:
        if isinstance(explicit, (list, tuple)):
            if len(explicit) != n_channels:
                raise ValueError(
                    f"Response defines {len(explicit)} measurement units for {n_channels} channels."
                )
            return [(u.Unit(value),) for value in explicit]
        return [(u.Unit(explicit),)] * n_channels

    # A temperature response multiplies an emission measure n_e^2 dl [cm^-5].
    # Instrument response tables often include pix^-1 even though FITS BUNIT
    # records the value stored in each pixel without that index unit. Accept both
    # explicit-pixel and implicit-pixel representations of the same output.
    response_output_unit = u.Unit(renderer.response_unit) * u.cm ** -5
    return [
        (response_output_unit, response_output_unit * u.pix)
        for _ in range(n_channels)
    ]


def _ordered_response_ids(value, channels, source):
    if value is None:
        return None
    if isinstance(value, Mapping):
        canonical_values = {
            canonical_channel_id(channel): response_id
            for channel, response_id in value.items()
        }
        expected_channels = [canonical_channel_id(channel) for channel in channels]
        if set(canonical_values) != set(expected_channels):
            raise ValueError(
                f'{source} response_id mapping must contain exactly channels {tuple(channels)}.'
            )
        ordered = [canonical_values[channel] for channel in expected_channels]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(channels):
            raise ValueError(
                f'{source} defines {len(value)} response IDs for {len(channels)} channels.'
            )
        ordered = list(value)
    else:
        ordered = [value] * len(channels)
    ordered = tuple(str(response_id).strip() for response_id in ordered)
    if any(not response_id for response_id in ordered):
        raise ValueError(f'{source} response_id values must be non-empty strings.')
    return ordered


def _expected_response_ids(renderer, response_config, response_channels):
    configured = _ordered_response_ids(
        response_config.get('response_id'), response_channels, 'temperature_response'
    )
    provenance = getattr(renderer, 'response_provenance', {})
    artifact = _ordered_response_ids(
        provenance.get('source_response_id', provenance.get('response_id')),
        response_channels,
        'response artifact provenance',
    )
    if configured is not None and artifact is not None and configured != artifact:
        raise ValueError(
            f'Configured response_id {configured} disagrees with response artifact '
            f'provenance {artifact}.'
        )
    expected = configured if configured is not None else artifact
    if expected is None:
        raise ValueError(
            'Every temperature response must define response_id explicitly or carry '
            'provenance.response_id so the renderer identity is reproducible.'
        )
    return expected


def validate_configured_response_artifacts(instruments_config):
    """Fail on response identity/calibration errors before loading observations."""
    validated = {}
    for instrument in instruments_config:
        instrument_key = instrument['key']
        response_config = instrument['temperature_response']
        artifact = load_response_artifact(response_config['artifact'])
        artifact.verify_response_id()
        selected = artifact.select_channels(response_config['channels'])
        provenance = artifact.provenance
        for field in (
            'sensitivity_convention', 'calibration_epoch', 'measurement_semantics'
        ):
            if not str(provenance.get(field, '')).strip():
                raise ValueError(
                    f"Response artifact for '{instrument_key}' is missing provenance.{field}."
                )
        if provenance['sensitivity_convention'] not in SENSITIVITY_CONVENTIONS:
            raise ValueError(
                f"Response artifact for '{instrument_key}' has invalid "
                'sensitivity_convention.'
            )
        if provenance['measurement_semantics'] not in {
            'surface_brightness', 'per_native_pixel'
        }:
            raise ValueError(
                f"Response artifact for '{instrument_key}' has invalid "
                'measurement_semantics.'
            )
        if provenance['measurement_semantics'] == 'per_native_pixel':
            solid_angle = provenance.get('native_pixel_solid_angle_sr')
            tolerance = provenance.get(
                'native_pixel_solid_angle_relative_tolerance'
            )
            if (
                not isinstance(solid_angle, (int, float))
                or not np.isfinite(solid_angle)
                or solid_angle <= 0
                or not isinstance(tolerance, (int, float))
                or not np.isfinite(tolerance)
                or not 0 <= tolerance < 1
            ):
                raise ValueError(
                    f"Response artifact for '{instrument_key}' has an invalid "
                    'native-pixel solid-angle contract.'
                )
        configured_ids = _ordered_response_ids(
            response_config.get('response_id'),
            selected.channels,
            f"instrument {instrument_key!r} temperature_response",
        )
        expected_ids = (artifact.response_id,) * len(selected.channels)
        if configured_ids is not None and configured_ids != expected_ids:
            raise ValueError(
                f"Configured response IDs for '{instrument_key}' do not match "
                f'the immutable source artifact ID {artifact.response_id!r}.'
            )
        validated[instrument_key] = {
            'response_id': artifact.response_id,
            'channels': selected.channels,
        }
    return validated


def validate_plasma_runtime_contract(sunerf, data_module, data_config, instruments_config):
    """Cross-check prepared observations against the instantiated renderers."""
    dataset_instruments = _configured_dataset_instruments(data_config)
    instrument_configs = {config['key']: config for config in instruments_config}

    resolved_instruments = {}
    for ds_key, dataset_metadata in data_module.config.items():
        instrument_key = dataset_instruments.get(ds_key, dataset_metadata.get('instrument_key', ds_key))
        if instrument_key not in sunerf.rendering.rendering_modules:
            raise ValueError(
                f"Prepared dataset '{ds_key}' maps to unknown instrument '{instrument_key}'."
            )
        renderer = sunerf.rendering.rendering_modules[instrument_key]
        prepared_channels = tuple(str(channel) for channel in dataset_metadata.get('channel_ids', ()))
        response_channels = tuple(str(channel) for channel in renderer.channels)
        if not prepared_channels:
            raise ValueError(f"Prepared dataset '{ds_key}' is missing ordered channel_ids metadata.")
        if [canonical_channel_id(value) for value in prepared_channels] != [
            canonical_channel_id(value) for value in response_channels
        ]:
            raise ValueError(
                f"Prepared dataset '{ds_key}' channel order {prepared_channels} does not match "
                f"renderer '{instrument_key}' response order {response_channels}."
            )

        measured_units = tuple(dataset_metadata.get('measurement_units', ()))
        if len(measured_units) != len(prepared_channels):
            raise ValueError(
                f"Prepared dataset '{ds_key}' must define one measurement unit per channel."
            )
        response_config = instrument_configs[instrument_key]['temperature_response']
        prepared_conventions = tuple(
            str(value).strip()
            for value in dataset_metadata.get('sensitivity_conventions', ())
        )
        if len(prepared_conventions) != len(prepared_channels):
            raise ValueError(
                f"Prepared dataset '{ds_key}' must define one sensitivity convention "
                'per channel.'
            )
        artifact_convention = renderer.response_provenance.get(
            'sensitivity_convention'
        )
        if artifact_convention not in SENSITIVITY_CONVENTIONS:
            raise ValueError(
                f"Renderer '{instrument_key}' response artifact is missing a valid "
                'sensitivity_convention.'
            )
        if prepared_conventions != (artifact_convention,) * len(prepared_channels):
            raise ValueError(
                f"Prepared dataset '{ds_key}' sensitivity conventions "
                f'{prepared_conventions} do not match renderer {instrument_key!r} '
                f"convention {artifact_convention!r}."
            )
        artifact_measurement_semantics = renderer.response_provenance.get(
            'measurement_semantics'
        )
        if artifact_measurement_semantics not in {
            'surface_brightness', 'per_native_pixel'
        }:
            raise ValueError(
                f"Renderer '{instrument_key}' response artifact is missing valid "
                'measurement_semantics.'
            )
        prepared_measurement_semantics = tuple(
            str(value).strip()
            for value in dataset_metadata.get('measurement_semantics', ())
        )
        if prepared_measurement_semantics != (
            artifact_measurement_semantics,
        ) * len(prepared_channels):
            raise ValueError(
                f"Prepared dataset '{ds_key}' measurement semantics "
                f'{prepared_measurement_semantics} do not match renderer '
                f"{instrument_key!r} semantics {artifact_measurement_semantics!r}."
            )
        if artifact_measurement_semantics == 'per_native_pixel':
            expected_solid_angle = float(
                renderer.response_provenance.get('native_pixel_solid_angle_sr', np.nan)
            )
            relative_tolerance = float(
                renderer.response_provenance.get(
                    'native_pixel_solid_angle_relative_tolerance', np.nan
                )
            )
            prepared_solid_angles = np.asarray(
                dataset_metadata.get(
                    'native_pixel_solid_angle_sr_by_observation',
                    dataset_metadata.get('native_pixel_solid_angle_sr', ()),
                ),
                dtype=np.float64,
            )
            if (
                not np.isfinite(expected_solid_angle)
                or expected_solid_angle <= 0
                or not np.isfinite(relative_tolerance)
                or not 0 <= relative_tolerance < 1
                or prepared_solid_angles.ndim not in {1, 2}
                or prepared_solid_angles.shape[-1] != len(prepared_channels)
                or not np.all(np.isfinite(prepared_solid_angles))
                or np.any(prepared_solid_angles <= 0)
            ):
                raise ValueError(
                    f"Dataset '{ds_key}' or renderer '{instrument_key}' has an invalid "
                    'native-pixel solid-angle contract.'
                )
            relative_errors = np.abs(
                prepared_solid_angles - expected_solid_angle
            ) / expected_solid_angle
            if np.any(relative_errors > relative_tolerance):
                raise ValueError(
                    f"Prepared dataset '{ds_key}' native pixel solid angles "
                    f'{prepared_solid_angles.tolist()} do not match response value '
                    f'{expected_solid_angle} sr within relative tolerance '
                    f'{relative_tolerance}.'
                )
        expected_units = _expected_measurement_units(
            renderer, response_config, len(prepared_channels)
        )
        for channel, measured, expected_candidates in zip(
            prepared_channels, measured_units, expected_units
        ):
            if str(measured).strip().lower() in {'', 'unknown', 'none'}:
                raise ValueError(
                    f"Prepared dataset '{ds_key}' channel '{channel}' has no physical BUNIT."
                )
            measured_unit = u.Unit(measured)
            if not any(measured_unit.is_equivalent(expected) for expected in expected_candidates):
                expected_text = ' or '.join(str(value) for value in expected_candidates)
                raise ValueError(
                    f"Prepared dataset '{ds_key}' channel '{channel}' unit {measured_unit} is "
                    f"incompatible with response output {expected_text} after n_e^2 dl."
                )

        resolved_instruments[ds_key] = instrument_key

    # The cache stores metadata derived from the prepared observations.  Persist
    # the YAML dataset-to-instrument mapping alongside it only after every
    # scientific compatibility check succeeds, so inference artifacts can map a
    # dataset back to the correct renderer even when their keys differ.
    for ds_key, instrument_key in resolved_instruments.items():
        data_module.config[ds_key]['instrument_key'] = instrument_key

    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--reload', action='store_true')
    args, overwrite_args = parser.parse_known_args(argv)

    # Fail on schema, unit, and channel-order mistakes before creating paths,
    # caches, a logger, or a model.
    config = validate_plasma_config(load_yaml_config(args.config, overwrite_args))
    validate_configured_response_artifacts(config['instruments'])
    # Image-scaling table references become explicit per-channel constants before
    # they are logged as hyperparameters or reach the model.
    config['instruments'] = resolve_instrument_scaling(config['instruments'])
    config_fingerprint, config_source_records = build_data_cache_fingerprint(config)

    base_path = config['base_path']
    os.makedirs(base_path, exist_ok=True)
    work_directory = config.get('work_directory', base_path)
    os.makedirs(work_directory, exist_ok=True)

    instruments_config = config['instruments']
    data_config = config['data']
    model_config = config.get('model', {})
    sampling_config = config.get('sampling', {})
    hierarchical_sampling_config = config.get('hierarchical_sampling', {})
    training_config = config.get('training', {})
    logging_config = config.get('logging', {'project': 'sunerf'})
    shuffle_config = config.get('shuffle')
    lambda_config = config.get('lambda', {})
    absorption_config = config.get('absorption', {'type': None})
    module_config = config.get('module', {})
    euv_callback_config = config['callbacks']['euv_tomography']
    use_absorption_validation = absorption_config.get('type') in {'learned', 'constant'}

    epochs = training_config.get('epochs', 1000)
    log_every_n_steps = training_config.get('log_every_n_steps')
    check_val_every_n_epoch = training_config.get('check_val_every_n_epoch', 1)
    ckpt_path = training_config.get('meta_path', 'last')

    # Schema and response-artifact validation above needs no external state.
    # Data preparation logs overview figures, so the W&B run must be started
    # (on rank zero) before the data module is built, as in the Thomson runner.
    logger = start_wandb_logger(logging_config, work_directory, hparams=config)

    data_module = prepare_data_module(
        work_directory,
        data_config,
        lambda generation: MultiInstrumentDataModule(
            **data_config,
            work_directory=generation,
            use_absorption=use_absorption_validation,
        ),
        reload=args.reload,
    )

    sunerf = PlasmaSuNeRFModule(
        Rs_per_ds=data_module.Rs_per_ds,
        seconds_per_dt=data_module.seconds_per_dt,
        validation_dataset_mapping=data_module.validation_dataset_mapping,
        instruments_config=instruments_config,
        model_config=model_config,
        sampling_config=sampling_config,
        hierarchical_sampling_config=hierarchical_sampling_config,
        shuffle_config=shuffle_config,
        lambda_config=lambda_config,
        absorption_config=absorption_config,
        **module_config,
    )
    validate_plasma_runtime_contract(
        sunerf,
        data_module,
        data_config,
        instruments_config,
    )
    sunerf.config_schema_version = config['schema_version']
    sunerf.config_fingerprint = config_fingerprint
    sunerf.config_source_records = config_source_records
    # Callback setup registers the exact tensors required by each plot. Enabling
    # the filter explicitly also prevents retaining diagnostics when every EUV
    # plot is disabled in YAML.
    sunerf.enable_validation_output_filter()

    checkpoint_callback = ModelCheckpoint(
        dirpath=base_path,
        save_last=True,
        every_n_train_steps=log_every_n_steps,
    )
    save_path = os.path.join(base_path, 'save_state.safe.pt')
    save_callback = LambdaCallback(
        on_validation_end=lambda *unused: _save_plasma_sunerf_rank_zero(
            sunerf, data_module, save_path
        )
    )
    callbacks = [checkpoint_callback, save_callback]

    if use_absorption_validation:
        absorption_dataset = getattr(
            data_module.validation_datasets['absorption'],
            'dataset',
            data_module.validation_datasets['absorption'],
        )
        callbacks.append(AbsorptionCallback('absorption', absorption_dataset.image_shape))

    if euv_callback_config['enabled']:
        available_validation_datasets = set(
            data_module.validation_dataset_mapping.values()
        )
        for ds_key, dataset_plot_config in euv_callback_config['datasets'].items():
            if ds_key not in available_validation_datasets:
                raise ValueError(
                    f"Configured EUV callback dataset {ds_key!r} is not present in "
                    "the prepared validation data module."
                )
            dataset = getattr(
                data_module.validation_datasets[ds_key],
                'dataset',
                data_module.validation_datasets[ds_key],
            )
            dataset_config = data_module.config[ds_key]
            image_shape = dataset_config.get('image_shape')
            if image_shape is None:
                image_shape = dataset.image_shape
            instrument_key = dataset_config['instrument_key']
            channel_metadata = copy.deepcopy(
                sunerf.instrument_metadata[instrument_key]['channels']
            )
            cmaps = dataset_config.get('cmaps')
            if cmaps is not None:
                if len(cmaps) != len(channel_metadata):
                    raise ValueError(
                        f"Dataset {ds_key!r} has {len(cmaps)} colormaps for "
                        f"{len(channel_metadata)} channels."
                    )
                for channel, cmap in zip(channel_metadata, cmaps):
                    channel['cmap'] = str(cmap)
            callbacks.append(EUVTomographyCallback(
                ds_key=ds_key,
                instrument_key=instrument_key,
                image_shape=image_shape,
                channel_metadata=channel_metadata,
                selected_channels=dataset_plot_config['channels'],
                products=euv_callback_config['products'],
                every_n_validations=euv_callback_config['every_n_validations'],
                figure_dpi=euv_callback_config['figure_dpi'],
            ))

    trainer = build_trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=epochs,
        val_check_interval=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    trainer.fit(sunerf, data_module, ckpt_path=ckpt_path)
    trainer.save_checkpoint(os.path.join(base_path, 'final.ckpt'))
    # Validation cadence need not coincide with the final optimizer step.
    _save_plasma_sunerf_rank_zero(sunerf, data_module, save_path)


if __name__ == '__main__':
    main()
