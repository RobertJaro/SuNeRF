import copy
import hashlib
import math
import os
import re
from datetime import date, datetime

import numpy as np
import torch
from astropy import units as u
from astropy.io.fits import Header
from astropy.wcs import WCS
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import nn

from sunerf.data.euv.observation import normalize_image_scaling
from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.model.model import PlasmaModel
from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.util import jacobian
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.plasma import PlasmaRadiativeTransfer, init_absorption_model
from sunerf.resources import normalize_artifact_reference, resolve_artifact_path
from sunerf.train.runtime import atomic_torch_save
from sunerf.train.scaling import ImageAsinhScaling, ImageLinearScaling, ImageLogScaling


PLASMA_SAFE_ARTIFACT_FORMAT_VERSION = 2
PLASMA_SAFE_ARTIFACT_SUFFIX = '.safe.pt'
PLASMA_GRID_ARTIFACT_TYPE = 'sunerf.plasma.grid'
PLASMA_GRID_ARTIFACT_FORMAT_VERSION = 1
_SAFE_TYPE_KEY = '__sunerf_safe_type__'


def build_log_temperature_grid(temperature_grid_config):
    """Build an inclusive, exactly spaced log10(K) grid from explicit metadata."""
    if not isinstance(temperature_grid_config, dict):
        raise ValueError(
            "model.temperature_grid must define log10_K_min, log10_K_max, and step_dex."
        )
    required = {'log10_K_min', 'log10_K_max', 'step_dex'}
    missing = sorted(required.difference(temperature_grid_config))
    unknown = sorted(set(temperature_grid_config).difference(required))
    if missing or unknown:
        raise ValueError(
            f"model.temperature_grid has missing fields {missing} and unsupported fields {unknown}."
        )
    minimum = float(temperature_grid_config['log10_K_min'])
    maximum = float(temperature_grid_config['log10_K_max'])
    step = float(temperature_grid_config['step_dex'])
    if not all(math.isfinite(value) for value in (minimum, maximum, step)):
        raise ValueError('model.temperature_grid values must be finite.')
    if maximum <= minimum or step <= 0:
        raise ValueError(
            'model.temperature_grid requires log10_K_max > log10_K_min and step_dex > 0.'
        )
    interval_count = (maximum - minimum) / step
    rounded_count = round(interval_count)
    if rounded_count < 1 or not math.isclose(
        interval_count, rounded_count, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            'model.temperature_grid range must contain an integer number of step_dex intervals.'
        )
    return np.linspace(minimum, maximum, rounded_count + 1, dtype=np.float32)


def _to_safe_primitive(value):
    """Encode runtime metadata using only weights-only-safe types and tensors."""
    # NumPy 2 scalar floats can satisfy ``isinstance(value, float)``; normalize
    # them before the Python primitive branches so no NumPy reconstruction
    # globals enter the pickle stream.
    if isinstance(value, np.generic):
        return _to_safe_primitive(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Safe artifact metadata cannot contain non-finite floats.')
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, u.Quantity):
        return {
            _SAFE_TYPE_KEY: 'quantity',
            'unit': str(value.unit),
            'value': _to_safe_primitive(np.asarray(value.value)),
        }
    if isinstance(value, (datetime, date)):
        return {_SAFE_TYPE_KEY: 'datetime', 'value': value.isoformat()}
    if isinstance(value, Header):
        return {
            _SAFE_TYPE_KEY: 'fits_header',
            'value': value.tostring(sep='\n', endcard=False, padding=False),
        }
    if isinstance(value, WCS):
        header = value.to_header(relax=True)
        return {
            _SAFE_TYPE_KEY: 'fits_wcs',
            'value': header.tostring(sep='\n', endcard=False, padding=False),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return [_to_safe_primitive(item) for item in value.tolist()]
        return {
            _SAFE_TYPE_KEY: 'ndarray',
            'dtype': str(value.dtype),
            'value': _to_safe_primitive(value.tolist()),
        }
    if isinstance(value, dict):
        return {
            _to_safe_primitive(key): _to_safe_primitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_safe_primitive(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    raise TypeError(
        f"Cannot encode {type(value).__name__} in a weights-only-safe plasma artifact."
    )


def to_safe_artifact_primitive(value):
    """Public encoder shared by safe plasma and grid-backed artifacts."""
    return _to_safe_primitive(value)


def _channel_wavelength(channel):
    """Return a numeric wavelength when a channel ID ends in one."""
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)$", str(channel).strip())
    if match is None:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(resolve_artifact_path(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _normalize_instrument_response_paths(instruments_config):
    """Copy an instrument config and freeze response paths at construction time."""
    normalized = copy.deepcopy(instruments_config)
    for instrument in normalized:
        response = instrument.get('temperature_response')
        if not isinstance(response, dict):
            continue
        path = response.get('artifact')
        if path is not None:
            response['artifact'] = normalize_artifact_reference(path)
    return normalized


def _validate_instrument_calibration_gauge(instruments_config):
    """Enforce one identifiable common-gain gauge for direct module users."""
    responses = {
        instrument['key']: instrument.get('temperature_response', {})
        for instrument in instruments_config
    }
    learnable = {
        key for key, response in responses.items()
        if response.get('learnable', False)
    }
    references = {
        key for key, response in responses.items()
        if response.get('global_reference', False)
    }
    if references - learnable:
        raise ValueError('global_reference requires learnable: true')
    if learnable and learnable != set(responses):
        fixed = sorted(set(responses) - learnable)
        raise ValueError(
            'Learnable cross-instrument calibration requires learnable: true for '
            f'every configured instrument; fixed anchors: {fixed}'
        )
    if learnable and len(references) != 1:
        raise ValueError(
            'Learnable response calibration requires exactly one global_reference; '
            f'found {sorted(references)}'
        )


def _normalize_absorption_artifact_path(absorption_config):
    normalized = copy.deepcopy(
        absorption_config if absorption_config is not None else {'type': None}
    )
    if normalized.get('artifact') is not None:
        normalized['artifact'] = normalize_artifact_reference(normalized['artifact'])
    return normalized


def _response_abundance_identity(response_provenance):
    spectral = response_provenance.get('spectral_emissivity', {})
    abundance = spectral.get('abundance', response_provenance.get('abundance'))
    if not isinstance(abundance, dict):
        return None
    return {
        key: str(abundance.get(key, '')).strip()
        for key in ('name', 'version', 'sha256')
    }


def _validate_absorption_response_abundance(renderer, absorption_model, instrument_key):
    if not getattr(absorption_model, 'is_deterministic_physical', False):
        return
    response_identity = _response_abundance_identity(renderer.response_provenance)
    bundle_abundance = absorption_model.provenance.get('abundance', {}).get('model')
    if response_identity is None or not isinstance(bundle_abundance, dict):
        raise ValueError(
            f"Instrument '{instrument_key}' response and absorption artifacts must both "
            'record an abundance model identity.'
        )
    bundle_identity = {
        key: str(bundle_abundance.get(key, '')).strip()
        for key in ('name', 'version', 'sha256')
    }
    if response_identity != bundle_identity:
        raise ValueError(
            f"Instrument '{instrument_key}' response abundance {response_identity} does "
            f'not match absorption bundle abundance {bundle_identity}.'
        )


class PlasmaSuNeRFModule(BaseSuNeRFModule):
    def __init__(self, Rs_per_ds, seconds_per_dt, instruments_config,
                 lambda_config=None, sampling_config=None, hierarchical_sampling_config=None, absorption_config=None,
                 model_config=None, shuffle_config=None,
                 regularization_density_scale_cm3=None, light_travel_time=False,
                 cool_column_scale_cm2=1.0e19, **kwargs):
        instruments_config = _normalize_instrument_response_paths(instruments_config)
        _validate_instrument_calibration_gauge(instruments_config)
        raw_instruments_config = copy.deepcopy(instruments_config)
        raw_lambda_config = copy.deepcopy(lambda_config)
        raw_sampling_config = copy.deepcopy(sampling_config)
        raw_hierarchical_sampling_config = copy.deepcopy(hierarchical_sampling_config)
        absorption_config = _normalize_absorption_artifact_path(absorption_config)
        raw_absorption_config = copy.deepcopy(absorption_config)
        raw_model_config = copy.deepcopy(model_config if model_config is not None else {})
        raw_shuffle_config = copy.deepcopy(shuffle_config)
        raw_module_kwargs = copy.deepcopy(kwargs)

        model_config = copy.deepcopy(raw_model_config)
        temperature_grid_config = model_config.pop('temperature_grid', None)
        log_T_range = build_log_temperature_grid(temperature_grid_config)
        self.log_T_range = log_T_range
        self.temperature_grid_config = copy.deepcopy(temperature_grid_config)

        absorption_type = absorption_config.get('type')
        shared_absorption_model = None
        if absorption_type != 'photoionization':
            shared_absorption_model = init_absorption_model(absorption_config)

        rendering_modules = {}
        scaling_modules = {}
        instrument_metadata = {}
        instrument_loss_weights = {}
        channel_loss_weights = {}
        for raw_instrument in copy.deepcopy(instruments_config):
            instrument = raw_instrument.copy()
            instrument_key = instrument.pop('key')
            instrument_type = str(instrument.pop('type')).lower()
            if instrument_type != 'plasma':
                raise ValueError(f"Unknown instrument type: {instrument_type}")

            response_config = copy.deepcopy(instrument.pop('temperature_response'))
            channels = list(response_config.get('channels', ()))
            if not channels:
                raise ValueError(
                    f"Instrument '{instrument_key}' must define an ordered, non-empty "
                    "temperature_response.channels list."
                )
            response_path = response_config.get('artifact')
            if response_path is None:
                raise ValueError(
                    f"Instrument '{instrument_key}' temperature response must define an artifact."
                )
            response_config.setdefault('Rs_per_ds', Rs_per_ds)
            absorption_model = (
                init_absorption_model(
                    absorption_config,
                    instrument_key=instrument_key,
                    channels=channels,
                )
                if absorption_type == 'photoionization'
                else shared_absorption_model
            )
            rendering_module = PlasmaRadiativeTransfer(
                temperature_response_config=response_config,
                log_T_range=log_T_range,
                absorption_model=absorption_model,
            )
            _validate_absorption_response_abundance(
                rendering_module, absorption_model, instrument_key
            )
            rendering_modules[instrument_key] = rendering_module

            loss_weight = float(instrument.pop('loss_weight', 1.0))
            if not np.isfinite(loss_weight) or loss_weight <= 0:
                raise ValueError(f"Instrument '{instrument_key}' loss_weight must be finite and positive.")
            configured_channel_weights = instrument.pop('channel_weights', None)
            if configured_channel_weights is None:
                configured_channel_weights = [1.0] * len(channels)
            elif isinstance(configured_channel_weights, dict):
                configured_channel_weights = [configured_channel_weights[str(channel)] for channel in channels]
            configured_channel_weights = [float(weight) for weight in configured_channel_weights]
            if len(configured_channel_weights) != len(channels):
                raise ValueError(
                    f"Instrument '{instrument_key}' defines {len(configured_channel_weights)} channel weights "
                    f"for {len(channels)} response channels."
                )
            if any(not np.isfinite(weight) or weight <= 0 for weight in configured_channel_weights):
                raise ValueError(f"Instrument '{instrument_key}' channel weights must be finite and positive.")

            instrument_loss_weights[instrument_key] = loss_weight
            channel_loss_weights[instrument_key] = configured_channel_weights
            instrument_metadata[instrument_key] = {
                'type': 'plasma',
                'channels': [
                    {
                        'id': str(channel),
                        'wavelength_angstrom': _channel_wavelength(channel),
                    }
                    for channel in channels
                ],
                'response': {
                    key: response_config[key]
                    for key in ('artifact', 'response_id', 'measurement_unit')
                    if key in response_config
                },
            }
            instrument_metadata[instrument_key]['response'].update({
                'path': normalize_artifact_reference(response_path),
                'sha256': _sha256_file(response_path),
                'channels': [str(channel) for channel in rendering_module.channels],
                'response_unit': str(rendering_module.response_unit),
                'emission_measure_convention': rendering_module.emission_measure_convention,
                'provenance': copy.deepcopy(rendering_module.response_provenance),
            })
            if absorption_model is not None:
                absorption_metadata = {'type': absorption_type}
                if getattr(absorption_model, 'is_deterministic_physical', False):
                    absorption_metadata.update({
                        'artifact': absorption_config['artifact'],
                        'sha256': _sha256_file(absorption_config['artifact']),
                        'bundle_id': absorption_model.bundle_id,
                        'species': list(absorption_model.species),
                        'channels': list(absorption_model.channels),
                        'hydrogen_density_convention': absorption_model.hydrogen_density_convention,
                        'cool_ion_fractions': dict(absorption_model.cool_ion_fractions),
                        'cool_cross_section_per_hydrogen_cm2': (
                            absorption_model.cool_cross_section_per_hydrogen_cm2.tolist()
                        ),
                        'provenance': copy.deepcopy(absorption_model.provenance),
                    })
                instrument_metadata[instrument_key]['absorption'] = absorption_metadata

            # The divisor is a fixed per-channel constant of the instrument, shared
            # by every dataset mapped to it; images and predictions both stay in
            # physical units until this loss-space transform.
            scaling_config = copy.deepcopy(instrument.pop('scaling', {}))
            scaling_type = scaling_config.pop('type', 'asinh')
            divisor_source = scaling_config.pop('divisor_source', None)
            divisor = scaling_config.pop('divisor', 1.0)
            if isinstance(divisor, str):
                raise ValueError(
                    f"Instrument '{instrument_key}' scaling.divisor is an unresolved table "
                    'path; resolve it with sunerf.data.euv.estimate_scaling.'
                    'resolve_instrument_scaling first.'
                )
            try:
                divisors, image_scaling_metadata = normalize_image_scaling(
                    divisor, rendering_module.channels
                )
            except ValueError as error:
                raise ValueError(
                    f"Instrument '{instrument_key}' scaling.divisor is invalid: {error}"
                ) from error
            if divisor_source is not None:
                image_scaling_metadata['source'] = str(divisor_source)
            instrument_metadata[instrument_key]['image_scaling'] = image_scaling_metadata
            scaling_config['divisor'] = divisors
            if scaling_type == 'asinh':
                scaling_modules[instrument_key] = ImageAsinhScaling(**scaling_config)
            elif scaling_type == 'linear':
                scaling_modules[instrument_key] = ImageLinearScaling(**scaling_config)
            elif scaling_type == 'log':
                scaling_modules[instrument_key] = ImageLogScaling(**scaling_config)
            else:
                raise ValueError(f"Unknown scaling type: {scaling_type}")

        if model_config.get('cool_absorber', False) and absorption_type != 'photoionization':
            raise ValueError(
                'model.cool_absorber requires absorption.type: photoionization; the cool '
                'hydrogen field has no effect without its photoionization cross sections.'
            )
        backend = model_config.pop('type', 'siren')
        if backend == 'generic':
            # Legacy name of the mlp backend, which used sine activations by default.
            backend = 'mlp'
            model_config.setdefault('activation', 'sine')
        model = PlasmaModel(log_T=log_T_range, backend=backend, **model_config)
        rendering = BasicRenderingModule(
            model=model,
            rendering_modules=rendering_modules,
            Rs_per_ds=Rs_per_ds,
            seconds_per_dt=seconds_per_dt,
            sampling_config=copy.deepcopy(sampling_config),
            hierarchical_sampling_config=copy.deepcopy(hierarchical_sampling_config),
            shuffle_config=copy.deepcopy(shuffle_config),
            # FITS timestamps are detector times: every sample is evaluated at
            # its emission time, t_obs - |x - x_obs| / c. Observers at different
            # heliocentric distances (e.g. Solar Orbiter and SDO) then agree on
            # one Sun-time axis.
            light_travel_time=bool(light_travel_time),
        )

        super().__init__(Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         rendering=rendering, **kwargs)

        lambda_config = lambda_config if lambda_config is not None else {}
        self.lambda_image = float(lambda_config.get('image', 1.0))
        self.lambda_regularization = float(lambda_config.get('regularization', 1.0e-4))
        self.lambda_absorption = float(lambda_config.get('absorption', 0.0))
        self.lambda_calibration = float(lambda_config.get('calibration', 1.0e-4))
        # L1 sparsity prior on the LOS hydrogen column of the cool absorber: it is
        # only constrained where it sits in front of bright emission.
        self.lambda_cool_absorber = float(lambda_config.get('cool_absorber', 1.0e-4))
        cool_column_scale_cm2 = float(cool_column_scale_cm2)
        if not np.isfinite(cool_column_scale_cm2) or cool_column_scale_cm2 <= 0:
            raise ValueError('cool_column_scale_cm2 must be finite and positive.')
        self.cool_column_scale_cm2 = cool_column_scale_cm2
        lambdas = {
            'image': self.lambda_image,
            'regularization': self.lambda_regularization,
            'absorption': self.lambda_absorption,
            'calibration': self.lambda_calibration,
            'cool_absorber': self.lambda_cool_absorber,
        }
        invalid_lambdas = [key for key, value in lambdas.items() if not np.isfinite(value) or value < 0]
        if invalid_lambdas or self.lambda_image == 0:
            raise ValueError(
                f"Loss weights must be finite and non-negative, with image > 0; invalid: {invalid_lambdas}."
            )

        if regularization_density_scale_cm3 is None:
            if self.lambda_regularization > 0:
                raise ValueError(
                    'regularization_density_scale_cm3 must be explicit and positive when '
                    'lambda.regularization is enabled.'
                )
            regularization_density_scale_cm3 = 1.0
        regularization_density_scale_cm3 = float(regularization_density_scale_cm3)
        if (
            not np.isfinite(regularization_density_scale_cm3)
            or regularization_density_scale_cm3 <= 0
        ):
            raise ValueError('regularization_density_scale_cm3 must be finite and positive.')
        self.register_buffer(
            'regularization_density_scale_cm3',
            torch.tensor(regularization_density_scale_cm3, dtype=torch.float32),
        )

        # Only legacy absorption has a single shared model. Physical opacity is
        # channel-specific and is owned by each instrument renderer.
        self.absorption_model = shared_absorption_model
        self.image_scaling = nn.ModuleDict(scaling_modules)
        self.instrument_metadata = instrument_metadata
        self.instrument_loss_weights = instrument_loss_weights
        self.channel_loss_weights = channel_loss_weights
        self.construction_spec = {
            'Rs_per_ds': float(Rs_per_ds),
            'seconds_per_dt': float(seconds_per_dt),
            'instruments_config': raw_instruments_config,
            'lambda_config': raw_lambda_config,
            'sampling_config': raw_sampling_config,
            'hierarchical_sampling_config': raw_hierarchical_sampling_config,
            'absorption_config': raw_absorption_config,
            'model_config': raw_model_config,
            'shuffle_config': raw_shuffle_config,
            'regularization_density_scale_cm3': regularization_density_scale_cm3,
            'light_travel_time': bool(light_travel_time),
            'cool_column_scale_cm2': cool_column_scale_cm2,
            **raw_module_kwargs,
        }

    @staticmethod
    def _expanded_valid_mask(valid_mask, target):
        if valid_mask is None:
            return torch.isfinite(target)
        valid_mask = valid_mask.to(device=target.device, dtype=torch.bool)
        while valid_mask.ndim < target.ndim:
            valid_mask = valid_mask.unsqueeze(-1)
        try:
            return torch.broadcast_to(valid_mask, target.shape)
        except RuntimeError as error:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} is not broadcastable to "
                f"image shape {tuple(target.shape)}."
            ) from error

    def _masked_image_loss(self, prediction, target, valid_mask, instrument_key, require_valid=True):
        """Average each channel independently, then weight channels explicitly."""
        mask = self._expanded_valid_mask(valid_mask, target) & torch.isfinite(target)
        invalid_prediction = mask & ~torch.isfinite(prediction)
        if torch.any(invalid_prediction):
            count = int(invalid_prediction.sum().detach().cpu())
            raise FloatingPointError(
                f"Predicted image for '{instrument_key}' contains {count} non-finite supervised values."
            )
        mask = mask & torch.isfinite(prediction)
        reduction_dims = tuple(range(prediction.ndim - 1))
        counts = mask.sum(dim=reduction_dims)
        channel_present = counts > 0
        if not torch.any(channel_present):
            if not require_valid:
                return prediction.new_zeros(()), mask
            raise ValueError(f"Batch for instrument '{instrument_key}' has no valid supervised pixels.")

        squared_error = torch.where(mask, (prediction - target).square(), torch.zeros_like(prediction))
        channel_mse = squared_error.sum(dim=reduction_dims) / counts.clamp_min(1)
        weights = torch.as_tensor(
            self.channel_loss_weights[instrument_key],
            dtype=channel_mse.dtype,
            device=channel_mse.device,
        )
        if weights.numel() != channel_mse.numel():
            raise ValueError(
                f"Instrument '{instrument_key}' rendered {channel_mse.numel()} channels but its "
                f"artifact/config metadata defines {weights.numel()}."
            )
        weights = weights * channel_present.to(weights.dtype)
        return (channel_mse * weights).sum() / weights.sum(), mask

    def _supervision_mask(self, image_batch, rendered, target):
        mask = self._expanded_valid_mask(image_batch.get('valid_mask'), target)
        ray_valid = rendered.get('ray_valid')
        if ray_valid is not None:
            ray_valid = self._expanded_valid_mask(ray_valid, target)
            mask = mask & ray_valid
        return mask

    def training_step(self, batch, batch_idx):
        instrument_batch = {
            key: value for key, value in batch.items()
            if key != 'random' and value is not None
        }
        if not instrument_batch:
            raise ValueError('A plasma training batch must contain at least one instrument dataset.')
        rendering_out = self.rendering(instrument_batch)['model_out']

        image_losses = []
        image_weights = []
        absorption_regularization = []
        calibration_regularization = []
        calibration_instruments = set()
        density_regularization = []
        cool_column_regularization = []

        for ds_key, image_batch in instrument_batch.items():
            instrument_key = image_batch['instrument']
            pred_image = self.image_scaling[instrument_key](rendering_out[ds_key]['image'])
            target_image = self.image_scaling[instrument_key](image_batch['image'])
            supervision_mask = self._supervision_mask(
                image_batch, rendering_out[ds_key], target_image
            )
            image_loss, _ = self._masked_image_loss(
                pred_image,
                target_image,
                supervision_mask,
                instrument_key,
            )
            image_losses.append(image_loss)
            image_weights.append(self.instrument_loss_weights[instrument_key])

            rendering_module = self.rendering.rendering_modules[instrument_key]
            if getattr(rendering_module.absorption_model, 'is_deterministic_physical', False):
                absorption_regularization.append(image_loss.new_zeros(()))
            else:
                absorption_regularization.append(
                    rendering_out[ds_key]['mean_absorption'].mean()
                )
            if (
                instrument_key not in calibration_instruments
                and hasattr(rendering_module, 'calibration_regularization')
            ):
                calibration_regularization.append(rendering_module.calibration_regularization())
                calibration_instruments.add(instrument_key)
            if 'cool_hydrogen_column_cm2' in rendering_out[ds_key]:
                cool_column_regularization.append(
                    (rendering_out[ds_key]['cool_hydrogen_column_cm2']
                     / self.cool_column_scale_cm2).mean()
                )
            if self.lambda_regularization != 0:
                density_regularization.append(
                    self.compute_radial_density_regularization(
                        rendering_out[ds_key]['em'],
                        rendering_out[ds_key]['distance'],
                    )
                )

        image_losses = torch.stack(image_losses)
        image_weights = torch.as_tensor(
            image_weights, dtype=image_losses.dtype, device=image_losses.device
        )
        image_loss = (image_losses * image_weights).sum() / image_weights.sum()
        absorption_regularization = torch.stack(absorption_regularization).mean()
        calibration_regularization = (
            torch.stack(calibration_regularization).mean()
            if calibration_regularization else image_loss.new_zeros(())
        )

        if self.lambda_regularization == 0:
            regularization = image_loss.new_zeros(())
        elif 'random' in batch and batch['random'] is not None:
            query_points = batch['random']['coords'].detach().requires_grad_(True)
            random_out = self.rendering.model(query_points)
            regularization = self.compute_static_regularization(
                random_out['total_log_ne'], random_out['mean_log_T'], query_points
            )
        else:
            regularization = torch.stack(density_regularization).mean()

        cool_column_regularization = (
            torch.stack(cool_column_regularization).mean()
            if cool_column_regularization else image_loss.new_zeros(())
        )

        loss = (
            self.lambda_image * image_loss
            + self.lambda_regularization * regularization
            + self.lambda_absorption * absorption_regularization
            + self.lambda_calibration * calibration_regularization
            + self.lambda_cool_absorber * cool_column_regularization
        )
        psnr = -10. * torch.log10(image_loss.clamp_min(torch.finfo(image_loss.dtype).tiny))

        self.log('loss', loss, sync_dist=True)
        self.log_dict({
            'train.image': image_loss,
            'train.psnr': psnr,
            'train.regularization': regularization,
            'train.absorption_regularization': absorption_regularization,
            'train.calibration_regularization': calibration_regularization,
            'train.cool_column_regularization': cool_column_regularization,
            'train.regularization_density_scale_cm3': self.regularization_density_scale_cm3,
        }, sync_dist=True)
        return loss

    def compute_continuity(self, total_ne, velocity, query_points):
        in_tensor = torch.cat([total_ne, velocity], dim=-1)
        jac_matrix = jacobian(in_tensor, query_points)

        dRho_dx = jac_matrix[:, 0, 0]
        dVx_dx = jac_matrix[:, 1, 0]
        dRho_dy = jac_matrix[:, 0, 1]
        dVy_dy = jac_matrix[:, 2, 1]
        dRho_dz = jac_matrix[:, 0, 2]
        dVz_dz = jac_matrix[:, 3, 2]
        dRho_dt = jac_matrix[:, 0, 3]

        div_v = dVx_dx + dVy_dy + dVz_dz
        grad_rho = torch.stack([dRho_dx, dRho_dy, dRho_dz], -1)
        v_dot_grad_rho = (velocity * grad_rho).sum(-1)
        continuity_eq = dRho_dt + total_ne * div_v + v_dot_grad_rho
        return continuity_eq.square().mean()

    def compute_radial_density_regularization(self, emission_measure_density_cm6, distance):
        """Penalize outer-corona density relative to an explicit physical scale."""
        dimensionless_density_squared = (
            emission_measure_density_cm6
            / self.regularization_density_scale_cm3.square()
        )
        radial_weight = torch.clip(distance - 1.2, min=0).square()
        return (dimensionless_density_squared * radial_weight).mean()

    def compute_static_regularization(self, total_log_ne, mean_log_T, query_points):
        # Differentiate dimensionless logarithmic plasma variables. Using
        # physical total_ne made this term scale as ~1e8 cm^-3 before squaring.
        in_tensor = torch.cat([total_log_ne, mean_log_T], dim=-1)
        jac_matrix = jacobian(in_tensor, query_points)

        dlogNe_dt = jac_matrix[:, 0, 3]
        dlogT_dt = jac_matrix[:, 1, 3]
        radius = torch.norm(query_points[..., :3], dim=-1)
        radius_weight = torch.clip(radius - 1.1, min=0).square()

        # Penalize the two derivatives separately. Squaring their sum allowed a
        # time-dependent density and temperature to cancel exactly.
        return (
            (dlogNe_dt * radius_weight).square()
            + (dlogT_dt * radius_weight).square()
        ).mean()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        valid_ds_id = self.validation_dataset_mapping[dataloader_idx]
        if valid_ds_id == 'absorption':
            log_T, log_ne = batch['log_T'], batch['log_ne']
            absorption_input = torch.cat([log_ne, log_T], dim=-1)
            abs_out = self.absorption_model(absorption_input)
            return {**abs_out, 'log_T': log_T, 'log_ne': log_ne}

        ds_key = valid_ds_id
        instrument_key = batch['instrument']
        # BasicRenderingModule owns the train-only shuffler gate and accepts this
        # explicit override so validation is deterministic.
        rendering_out = self.rendering(
            {ds_key: batch}, shuffle=False, diagnostics=True
        )
        model_out = rendering_out['model_out']
        image_scaling = self.image_scaling[instrument_key]
        target_image = image_scaling(batch['image'])
        pred_image = image_scaling(model_out[ds_key]['image'])
        supervision_mask = self._supervision_mask(batch, model_out[ds_key], target_image)
        # Validation walks full images in fixed chunks, so a chunk may fall entirely
        # in a masked region (e.g. image corners); only the mask is needed here.
        _, valid_mask = self._masked_image_loss(
            pred_image, target_image, supervision_mask, instrument_key, require_valid=False
        )

        # Invalid values are zero-filled only for fixed-shape callback gathering;
        # their mask is retained and they never enter the loss above.
        target_for_callback = torch.where(valid_mask, target_image, torch.zeros_like(target_image))
        pred_for_callback = torch.where(valid_mask, pred_image, torch.zeros_like(pred_image))
        return {
            'target_image': target_for_callback,
            'pred_image': pred_for_callback,
            'valid_mask': valid_mask,
            'mean_T': model_out[ds_key]['mean_T'],
            'total_ne': model_out[ds_key]['total_ne'],
            'column_electron_density_cm2': model_out[ds_key][
                'column_electron_density_cm2'
            ],
            'emission_measure_cm5': model_out[ds_key]['emission_measure_cm5'],
            'differential_emission_measure_cm5_per_dex': model_out[ds_key][
                'differential_emission_measure_cm5_per_dex'
            ],
            'height_map': model_out[ds_key]['height_map'],
            'mean_absorption': model_out[ds_key]['mean_absorption'],
            'z_vals_stratified': rendering_out['z_vals_stratified'],
            'z_vals_hierarchical': rendering_out['z_vals'],
            'distance': model_out[ds_key]['distance'],
        }

    def _instrument_scaling_metrics(self):
        metrics = {}
        for instrument_key, module in self.rendering.rendering_modules.items():
            scaling = getattr(module, 'effective_instrument_scaling', module.instrument_scaling)
            scaling = scaling.detach().reshape(-1)
            channel_ids = [channel['id'] for channel in self.instrument_metadata[instrument_key]['channels']]
            if scaling.numel() == 1:
                metrics[f'instrument_scaling.{instrument_key}'] = scaling[0]
            else:
                for channel_id, value in zip(channel_ids, scaling):
                    metrics[f'instrument_scaling.{instrument_key}.{channel_id}'] = value
            common_gain = getattr(module, 'common_gain_delta_dex', None)
            if common_gain is not None:
                metrics[f'common_instrument_scaling.{instrument_key}'] = (
                    common_gain.detach().reshape(())
                )
        return metrics

    def on_train_batch_end(self, *args, **kwargs):
        metrics = self._instrument_scaling_metrics()
        if metrics:
            self.log_dict(metrics, sync_dist=True)
        super().on_train_batch_end(*args, **kwargs)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        metrics = self._instrument_scaling_metrics()
        if metrics:
            self.log_dict(metrics, sync_dist=True)


def _artifact_instrument_metadata(sunerf, data_module):
    metadata = copy.deepcopy(sunerf.instrument_metadata)
    for instrument_key, instrument in metadata.items():
        rendering_modules = getattr(
            getattr(sunerf, 'rendering', None), 'rendering_modules', {}
        )
        renderer = (
            rendering_modules[instrument_key]
            if instrument_key in rendering_modules else None
        )
        renderer_channels = getattr(renderer, 'channels', None)
        if renderer_channels is None:
            renderer_channels = instrument.get('response', {}).get('channels')
        if renderer_channels is None:
            renderer_channels = [channel.get('id') for channel in instrument['channels']]
        renderer_channels = tuple(str(channel) for channel in renderer_channels)
        stored_response_channels = tuple(
            str(channel.get('response_channel_id', channel.get('id')))
            for channel in instrument['channels']
        )
        if stored_response_channels != renderer_channels:
            raise ValueError(
                f"Instrument '{instrument_key}' response channel order "
                f'{stored_response_channels} does not exactly match renderer channels '
                f'{renderer_channels}.'
            )

        dataset_configs = [
            (ds_key, config) for ds_key, config in data_module.config.items()
            if config.get('instrument_key', ds_key) == instrument_key
        ]
        dataset_config = dataset_configs[0][1] if dataset_configs else {}
        agreement_fields = ('channel_ids', 'cmaps', 'measurement_units')
        for ds_key, candidate in dataset_configs[1:]:
            disagreements = [
                field for field in agreement_fields
                if candidate.get(field) != dataset_config.get(field)
            ]
            if disagreements:
                raise ValueError(
                    f"Dataset '{ds_key}' disagrees with the other datasets mapped to "
                    f"instrument '{instrument_key}' for {disagreements}."
                )

        expected_response_id = getattr(renderer, 'response_id', None)
        response_metadata = instrument.get('response', {})
        response_provenance = response_metadata.get('provenance', {})
        expected_response_id = expected_response_id or response_provenance.get(
            'source_response_id', response_metadata.get(
                'response_id', response_provenance.get('response_id')
            )
        )
        response_ids = (
            None if expected_response_id is None
            else (str(expected_response_id).strip(),) * len(renderer_channels)
        )

        channel_fields = {
            'id': dataset_config.get('channel_ids'),
            'cmap': dataset_config.get('cmaps'),
            'measurement_unit': dataset_config.get('measurement_units'),
            'response_id': response_ids,
        }
        for field, values in channel_fields.items():
            if values is None:
                continue
            if len(values) != len(instrument['channels']):
                raise ValueError(
                    f"Instrument '{instrument_key}' has {len(instrument['channels'])} response channels "
                    f"but {len(values)} stored values for {field!r}."
                )
            for channel, value in zip(instrument['channels'], values):
                if field == 'id':
                    channel['response_channel_id'] = channel['id']
                channel[field] = str(value)
        if renderer is not None and hasattr(renderer, 'effective_instrument_scaling'):
            effective_dex = (
                renderer.effective_instrument_scaling.detach().cpu().reshape(-1)
            )
            relative_dex = (
                renderer.instrument_gain_delta_dex.detach().cpu().reshape(-1)
            )
            common_dex = float(
                renderer.common_gain_delta_dex.detach().cpu().reshape(())
            )
            if effective_dex.numel() != len(renderer_channels):
                raise ValueError(
                    f"Instrument '{instrument_key}' calibration gain shape does not "
                    "match its response channels."
                )
            instrument['calibration'] = {
                'schema': 'sunerf.response_calibration.v1',
                'mode': 'learned' if renderer.learnable else 'fixed_nominal',
                'response_semantics': 'base_response_times_effective_gain',
                'gain_coordinate': 'base10_logarithm',
                'channel_ids': list(renderer_channels),
                'global_reference': bool(renderer.global_reference),
                'density_gain_gauge': (
                    'fixed_global_reference'
                    if renderer.global_reference
                    else (
                        'learned_relative_to_global_reference'
                        if renderer.learnable else 'nominal_response_amplitude'
                    )
                ),
                'common_gain_delta_dex': common_dex,
                'relative_channel_gain_delta_dex': relative_dex.tolist(),
                'effective_gain_delta_dex': effective_dex.tolist(),
                'effective_multiplicative_gain': torch.pow(
                    torch.tensor(10.0, dtype=effective_dex.dtype), effective_dex
                ).tolist(),
            }
    return metadata


def plasma_safe_artifact_path(save_path):
    save_path = os.path.abspath(os.fspath(save_path))
    return save_path if save_path.endswith(PLASMA_SAFE_ARTIFACT_SUFFIX) else f'{save_path}{PLASMA_SAFE_ARTIFACT_SUFFIX}'


def _response_artifact_references(instrument_metadata):
    references = {}
    for instrument_key, metadata in instrument_metadata.items():
        response = copy.deepcopy(metadata.get('response', {}))
        missing = [key for key in ('path', 'sha256', 'provenance') if key not in response]
        if missing:
            raise ValueError(
                f"Instrument '{instrument_key}' response metadata is missing immutable fields {missing}."
            )
        references[instrument_key] = response
    return references


def _absorption_artifact_references(instrument_metadata):
    references = {}
    for instrument_key, metadata in instrument_metadata.items():
        absorption = copy.deepcopy(metadata.get('absorption'))
        if absorption is None or absorption.get('type') != 'photoionization':
            continue
        missing = [
            key for key in ('artifact', 'sha256', 'bundle_id', 'provenance')
            if key not in absorption
        ]
        if missing:
            raise ValueError(
                f"Instrument '{instrument_key}' absorption metadata is missing "
                f'immutable fields {missing}.'
            )
        references[instrument_key] = absorption
    return references


def _safe_plasma_state(sunerf, data_module, instrument_metadata, provenance):
    if not hasattr(sunerf, 'construction_spec'):
        raise ValueError(
            'Plasma module is missing its primitive construction_spec and cannot be saved safely.'
        )
    model_state = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else _to_safe_primitive(value)
        for key, value in sunerf.state_dict().items()
    }
    construction_spec = copy.deepcopy(sunerf.construction_spec)
    if 'instruments_config' in construction_spec:
        construction_spec['instruments_config'] = _normalize_instrument_response_paths(
            construction_spec['instruments_config']
        )
    if 'absorption_config' in construction_spec:
        construction_spec['absorption_config'] = _normalize_absorption_artifact_path(
            construction_spec['absorption_config']
        )
    for key in ('Rs_per_ds', 'seconds_per_dt'):
        module_value = float(construction_spec[key])
        data_value = float(getattr(data_module, key))
        if not np.isclose(module_value, data_value, rtol=0.0, atol=1e-12):
            raise ValueError(
                f'Cannot save plasma artifact: module {key}={module_value} differs '
                f'from data module {key}={data_value}.'
            )
        for dataset_key, dataset in data_module.config.items():
            if key in dataset and not np.isclose(
                float(dataset[key]), data_value, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Cannot save plasma artifact: dataset '{dataset_key}' "
                    f'{key}={dataset[key]} differs from data module {data_value}.'
                )
    return {
        'artifact_type': 'sunerf.plasma.weights',
        'artifact_format_version': PLASMA_SAFE_ARTIFACT_FORMAT_VERSION,
        'artifact_security': 'weights_only',
        'construction_spec': _to_safe_primitive(construction_spec),
        'state_dict': model_state,
        'data_config': _to_safe_primitive(data_module.config),
        'Rs_per_ds': float(data_module.Rs_per_ds),
        'seconds_per_dt': float(data_module.seconds_per_dt),
        'ref_date': _to_safe_primitive(data_module.ref_date),
        'regularization_density_scale_cm3': float(
            sunerf.regularization_density_scale_cm3.detach().cpu()
        ),
        'temperature_grid': _to_safe_primitive(sunerf.temperature_grid_config),
        'log_T_range': torch.as_tensor(sunerf.log_T_range, dtype=torch.float32).cpu(),
        'instrument_metadata': _to_safe_primitive(instrument_metadata),
        'provenance': _to_safe_primitive(provenance),
    }


@rank_zero_only
def save_plasma_sunerf(sunerf: PlasmaSuNeRFModule, data_module: BaseDataModule, save_path):
    """Publish one weights-only plasma reconstruction artifact."""
    save_path = os.path.abspath(os.fspath(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    configured_instruments = {
        instrument['key']
        for instrument in sunerf.construction_spec.get('instruments_config', ())
    }
    represented_instruments = {
        config.get('instrument_key', dataset_key)
        for dataset_key, config in data_module.config.items()
    }
    missing_instruments = configured_instruments - represented_instruments
    if missing_instruments:
        raise ValueError(
            'Cannot save plasma artifact because configured instruments have no '
            f'prepared dataset metadata: {sorted(missing_instruments)}'
        )
    instrument_metadata = _artifact_instrument_metadata(sunerf, data_module)
    response_artifacts = _response_artifact_references(instrument_metadata)
    absorption_artifacts = _absorption_artifact_references(instrument_metadata)
    provenance = {
        'config_schema_version': getattr(sunerf, 'config_schema_version', None),
        'config_fingerprint': getattr(sunerf, 'config_fingerprint', None),
        'config_source_records': copy.deepcopy(
            getattr(sunerf, 'config_source_records', [])
        ),
        'data_cache_format_version': getattr(data_module, 'cache_format_version', None),
        'data_cache_fingerprint': getattr(data_module, 'cache_fingerprint', None),
        'data_source_records': copy.deepcopy(
            getattr(data_module, 'cache_source_records', [])
        ),
        'response_artifacts': response_artifacts,
        'absorption_artifacts': absorption_artifacts,
        'response_calibration': {
            key: copy.deepcopy(value.get('calibration'))
            for key, value in instrument_metadata.items()
            if value.get('calibration') is not None
        },
    }
    safe_path = plasma_safe_artifact_path(save_path)
    safe_state = _safe_plasma_state(sunerf, data_module, instrument_metadata, provenance)
    atomic_torch_save(safe_state, safe_path)
    # Verify that the just-published artifact is accepted by PyTorch's
    # restricted unpickler.
    torch.load(safe_path, map_location='cpu', weights_only=True)
    return safe_path
