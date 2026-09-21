from datetime import datetime
from typing import Tuple, Iterable
import hashlib
import os
import pickle
import re

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from torch import nn
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.data.ray_sampling import get_rays, hpc_impact_parameter
from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.evaluation.util import convert_spherical_to_cartesian
from sunerf.model.plasma import (
    PLASMA_GRID_ARTIFACT_FORMAT_VERSION,
    PLASMA_GRID_ARTIFACT_TYPE,
    PLASMA_SAFE_ARTIFACT_FORMAT_VERSION,
    PLASMA_SAFE_ARTIFACT_SUFFIX,
    PlasmaSuNeRFModule,
)
from sunerf.model.spherical_grid import SphericalGridPlasmaModel
from sunerf.rendering.base_tracing import BasicRenderingModule, MultiResolutionRenderingModule
from sunerf.rendering.plasma import PlasmaRadiativeTransfer
from sunerf.train.coordinate_transformation import pose_spherical, spherical_to_cartesian


_SAFE_TYPE_KEY = '__sunerf_safe_type__'


def _carrington_pose_from_hci_observer(observer, Rs_per_ds):
    """Build the model pose in the Carrington frame used during training."""
    obstime = observer.obstime
    carrington = observer.transform_to(
        frames.HeliographicCarrington(observer='self', obstime=obstime)
    )
    return pose_spherical(
        carrington.lon.to_value(u.rad),
        carrington.lat.to_value(u.rad),
        carrington.radius.to_value(u.solRad) / float(Rs_per_ds),
    )


def _validate_artifact_coordinate_contract(state, construction_spec):
    """Reject safe states whose camera/time scales disagree with rendering."""
    render_spec = construction_spec.get('rendering', construction_spec)
    for key in ('Rs_per_ds', 'seconds_per_dt'):
        top_level = float(state[key])
        renderer_value = float(render_spec[key])
        if not np.isclose(top_level, renderer_value, rtol=0.0, atol=1e-12):
            raise ValueError(
                f'Safe artifact {key} mismatch: top-level value {top_level} '
                f'differs from renderer construction value {renderer_value}.'
            )
        data_config = _from_safe_primitive(state.get('data_config', {}))
        for dataset_key, dataset in data_config.items():
            if key in dataset and not np.isclose(
                float(dataset[key]), top_level, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"Safe artifact dataset '{dataset_key}' {key}={dataset[key]} "
                    f'differs from top-level value {top_level}.'
                )


def _from_safe_primitive(value):
    if isinstance(value, list):
        return [_from_safe_primitive(item) for item in value]
    if not isinstance(value, dict):
        return value
    safe_type = value.get(_SAFE_TYPE_KEY)
    if safe_type == 'datetime':
        return datetime.fromisoformat(value['value'].replace('Z', '+00:00'))
    if safe_type == 'quantity':
        return _from_safe_primitive(value['value']) * u.Unit(value['unit'])
    if safe_type in {'fits_header', 'fits_wcs'}:
        header = fits.Header.fromstring(value['value'], sep='\n')
        return header if safe_type == 'fits_header' else WCS(header)
    if safe_type == 'ndarray':
        return np.asarray(_from_safe_primitive(value['value']), dtype=value['dtype'])
    if safe_type is not None:
        raise ValueError(f'Unsupported safe-artifact metadata type {safe_type!r}.')
    return {
        _from_safe_primitive(key): _from_safe_primitive(item)
        for key, item in value.items()
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_plasma_candidate(state_path):
    state_path = os.path.abspath(os.fspath(state_path))
    if state_path.endswith(PLASMA_SAFE_ARTIFACT_SUFFIX):
        return state_path
    candidate = f'{state_path}{PLASMA_SAFE_ARTIFACT_SUFFIX}'
    return candidate if os.path.isfile(candidate) else None


def _verify_response_reference(instrument_key, response, configured_response):
    reference_path = response.get('path')
    if reference_path is None:
        raise ValueError(
            f"Safe artifact response reference for '{instrument_key}' has no path."
        )
    from sunerf.resources import normalize_artifact_reference, resolve_artifact_path

    configured_path = normalize_artifact_reference(configured_response['artifact'])
    if configured_path != normalize_artifact_reference(reference_path):
        raise ValueError(
            f"Safe artifact response reference for '{instrument_key}' does not match "
            'its primitive construction spec.'
        )
    path = os.fspath(resolve_artifact_path(reference_path))
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Safe artifact requires response file for '{instrument_key}': {path}"
        )
    actual_hash = _sha256_file(path)
    if actual_hash != response['sha256']:
        raise ValueError(
            f"Response artifact hash mismatch for '{instrument_key}': expected "
            f"{response['sha256']}, got {actual_hash}."
        )


def _verify_immutable_plasma_state(module, saved_state):
    """Prevent saved weights from replacing artifact/config-derived physics."""
    fresh_state = module.state_dict()
    immutable_names = set()
    for instrument_key, renderer in module.rendering.rendering_modules.items():
        prefix = f'rendering.rendering_modules.{instrument_key}.'
        immutable_names.update(prefix + name for name, _ in renderer.named_buffers())
        immutable_names.update(
            prefix + name
            for name, parameter in renderer.named_parameters()
            if not parameter.requires_grad
        )

    model = module.rendering.model
    immutable_names.update(
        f'rendering.model.{name}' for name, _ in model.named_buffers()
    )
    immutable_names.update(
        f'rendering.model.{name}'
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )

    for name in sorted(immutable_names):
        if name not in saved_state or name not in fresh_state:
            continue  # strict load below reports missing or unexpected state.
        saved = saved_state[name]
        expected = fresh_state[name]
        if not isinstance(saved, torch.Tensor) or not torch.equal(
            saved.detach().cpu(), expected.detach().cpu()
        ):
            raise ValueError(
                f"Safe artifact immutable physics state {name!r} disagrees with "
                "the verified response/configuration."
            )


def _verify_immutable_grid_rendering_state(rendering, saved_state):
    """Protect grid-artifact response and fixed-gain buffers before state load."""
    fresh_state = rendering.state_dict()
    immutable_names = set()
    for instrument_key, renderer in rendering.rendering_modules.items():
        prefix = f'rendering_modules.{instrument_key}.'
        immutable_names.update(prefix + name for name, _ in renderer.named_buffers())
        immutable_names.update(
            prefix + name
            for name, parameter in renderer.named_parameters()
            if not parameter.requires_grad
        )
    for name in sorted(immutable_names):
        if name not in saved_state or name not in fresh_state:
            continue
        saved = saved_state[name]
        expected = fresh_state[name]
        if not isinstance(saved, torch.Tensor) or not torch.equal(
            saved.detach().cpu(), expected.detach().cpu()
        ):
            raise ValueError(
                f"Safe grid artifact immutable physics state {name!r} disagrees "
                "with the verified response/configuration."
            )


def _load_safe_plasma_artifact(state_path, device, state=None):
    # Reconstruct on CPU first to avoid holding duplicate GPU state tensors;
    # SuNeRFLoader moves the completed renderer to the requested device below.
    state = (
        torch.load(state_path, map_location='cpu', weights_only=True)
        if state is None else state
    )
    if state.get('artifact_type') != 'sunerf.plasma.weights':
        raise ValueError(f'{state_path!r} is not a safe plasma weights artifact.')
    if state.get('artifact_format_version') != PLASMA_SAFE_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported safe plasma artifact version: {state.get('artifact_format_version')!r}."
        )

    provenance = _from_safe_primitive(state.get('provenance', {}))
    response_references = provenance.get('response_artifacts', {})
    construction_spec = _from_safe_primitive(state['construction_spec'])
    _validate_artifact_coordinate_contract(state, construction_spec)
    configured_instruments = {
        instrument['key']: instrument for instrument in construction_spec['instruments_config']
    }
    for instrument_key, response in response_references.items():
        configured_response = configured_instruments[instrument_key]['temperature_response']
        _verify_response_reference(instrument_key, response, configured_response)
    if set(response_references) != set(configured_instruments):
        raise ValueError(
            'Safe artifact response references do not cover every configured instrument.'
        )

    module = PlasmaSuNeRFModule(**construction_spec)
    _verify_immutable_plasma_state(module, state['state_dict'])
    module.load_state_dict(state['state_dict'], strict=True)
    module.eval()
    runtime_state = {
        'artifact_type': state['artifact_type'],
        'artifact_format_version': state['artifact_format_version'],
        'artifact_security': state['artifact_security'],
        'rendering': module.rendering,
        'data_config': _from_safe_primitive(state['data_config']),
        'Rs_per_ds': float(state['Rs_per_ds']),
        'seconds_per_dt': float(state['seconds_per_dt']),
        'ref_date': _from_safe_primitive(state['ref_date']),
        'regularization_density_scale_cm3': float(
            state['regularization_density_scale_cm3']
        ),
        'temperature_grid': _from_safe_primitive(state['temperature_grid']),
        'log_T_range': state['log_T_range'].detach().cpu().numpy(),
        'instrument_metadata': _from_safe_primitive(state['instrument_metadata']),
        'provenance': provenance,
    }
    return runtime_state, module


def _load_safe_grid_artifact(state_path, state=None):
    state = (
        torch.load(state_path, map_location='cpu', weights_only=True)
        if state is None else state
    )
    if state.get('artifact_type') != PLASMA_GRID_ARTIFACT_TYPE:
        raise ValueError(f'{state_path!r} is not a safe plasma-grid artifact.')
    if state.get('artifact_format_version') != PLASMA_GRID_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported safe plasma-grid artifact version: "
            f"{state.get('artifact_format_version')!r}."
        )
    if state.get('artifact_security') != 'weights_only':
        raise ValueError('Safe plasma-grid artifact is missing its weights-only marker.')

    construction_spec = _from_safe_primitive(state['construction_spec'])
    _validate_artifact_coordinate_contract(state, construction_spec)
    model_spec = construction_spec['model']
    if model_spec.get('type') != 'spherical_grid_plasma':
        raise ValueError(f"Unsupported grid model type: {model_spec.get('type')!r}.")
    grid_state = state['grid_state_dict']
    required_grid_buffers = {
        'values', 'time', 'radius', 'latitude', 'longitude', 'log_T'
    }
    if not required_grid_buffers.issubset(grid_state):
        raise ValueError(
            'Safe plasma-grid artifact is missing model buffers: '
            f'{sorted(required_grid_buffers.difference(grid_state))}.'
        )
    values = grid_state['values']
    if values.ndim != 5 or values.shape[-1] != 2:
        raise ValueError(
            f"Safe plasma-grid values must have shape (t,r,lat,lon,2), got {tuple(values.shape)}."
        )
    model = SphericalGridPlasmaModel(
        log_density=values[..., 0],
        log_temperature=values[..., 1],
        time=grid_state['time'],
        radius=grid_state['radius'],
        latitude=grid_state['latitude'],
        longitude=grid_state['longitude'],
        log_T=grid_state['log_T'],
        longitude_period=model_spec['longitude_period'],
        fill_log_density=model_spec['fill_log_density'],
        fill_log_temperature=model_spec['fill_log_temperature'],
        clamp_time=model_spec['clamp_time'],
    )

    rendering_spec = construction_spec['rendering']
    instrument_key = rendering_spec['instrument_key']
    response_config = rendering_spec['temperature_response_config']
    provenance = _from_safe_primitive(state.get('provenance', {}))
    response_references = provenance.get('response_artifacts', {})
    if set(response_references) != {instrument_key}:
        raise ValueError(
            'Safe plasma-grid response references must contain exactly its instrument.'
        )
    _verify_response_reference(
        instrument_key, response_references[instrument_key], response_config
    )
    from sunerf.data.psi.build_synthetic import build_grid_absorption_model

    radiative_transfer = PlasmaRadiativeTransfer(
        temperature_response_config=response_config,
        log_T_range=grid_state['log_T'].detach().cpu().numpy(),
        absorption_model=build_grid_absorption_model(
            rendering_spec.get('absorption_config'), response_config['channels']
        ),
    )
    rendering = BasicRenderingModule(
        model=model,
        rendering_modules={instrument_key: radiative_transfer},
        Rs_per_ds=rendering_spec['Rs_per_ds'],
        seconds_per_dt=rendering_spec['seconds_per_dt'],
        sampling_config=rendering_spec['sampling_config'],
        hierarchical_sampling_config=rendering_spec['hierarchical_sampling_config'],
    )
    combined_state = {
        **{f'model.{key}': value for key, value in grid_state.items()},
        **state['rendering_state_dict'],
    }
    _verify_immutable_grid_rendering_state(
        rendering, state['rendering_state_dict']
    )
    rendering.load_state_dict(combined_state, strict=True)
    rendering.eval()

    runtime_state = {
        'artifact_type': state['artifact_type'],
        'artifact_format_version': state['artifact_format_version'],
        'artifact_security': state['artifact_security'],
        'rendering': rendering,
        'data_config': _from_safe_primitive(state['data_config']),
        'Rs_per_ds': float(state['Rs_per_ds']),
        'seconds_per_dt': float(state['seconds_per_dt']),
        'ref_date': _from_safe_primitive(state['ref_date']),
        'temperature_grid': _from_safe_primitive(state['temperature_grid']),
        'log_T_range': state['log_T_range'].detach().cpu().numpy(),
        'instrument_metadata': _from_safe_primitive(state['instrument_metadata']),
        'temperature_response': _from_safe_primitive(state['temperature_response']),
        'synthetic_grid': _from_safe_primitive(state['synthetic_grid']),
        'provenance': provenance,
    }
    return runtime_state, rendering


def _load_safe_artifact(state_path, state=None):
    state = (
        torch.load(state_path, map_location='cpu', weights_only=True)
        if state is None else state
    )
    artifact_type = state.get('artifact_type')
    if artifact_type == 'sunerf.plasma.weights':
        # The neural-field loader constructs the Lightning module, so retain it
        # for callers that need access to its full state.
        return _load_safe_plasma_artifact(state_path, device='cpu', state=state)
    if artifact_type == PLASMA_GRID_ARTIFACT_TYPE:
        return _load_safe_grid_artifact(state_path, state=state)
    raise ValueError(f'Unsupported weights-only SuNeRF artifact type: {artifact_type!r}.')


class SuNeRFLoader:

    def __init__(self, state_path, device=None, trusted=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.device = device

        safe_path = _safe_plasma_candidate(state_path)
        restricted_state = None
        requested_path = os.path.abspath(os.fspath(state_path))
        if safe_path is None:
            if not os.path.isfile(requested_path):
                raise FileNotFoundError(requested_path)
            try:
                restricted_state = torch.load(
                    requested_path, map_location='cpu', weights_only=True
                )
            except (pickle.UnpicklingError, RuntimeError, EOFError):
                restricted_state = None
            if (
                isinstance(restricted_state, dict)
                and restricted_state.get('artifact_type') in {
                    'sunerf.plasma.weights', PLASMA_GRID_ARTIFACT_TYPE
                }
            ):
                safe_path = requested_path
        self.loaded_safely = safe_path is not None
        self.loaded_artifact_path = safe_path or requested_path
        self._reconstructed_module = None
        if safe_path is not None:
            state, self._reconstructed_module = _load_safe_artifact(
                safe_path, state=restricted_state
            )
        else:
            if trusted is not True:
                raise ValueError(
                    "This SuNeRF .snf artifact has no weights-only-safe sidecar and "
                    "contains executable pickled Python modules. Load only a trusted "
                    "local artifact and pass trusted=True."
                )
            state = torch.load(state_path, map_location=device, weights_only=False)
        self.state = state
        artifact_type = state.get('artifact_type')
        artifact_version = state.get('artifact_format_version')
        if artifact_type is not None and artifact_type not in {
            'sunerf.plasma.weights', PLASMA_GRID_ARTIFACT_TYPE,
            'sunerf.thomson'
        }:
            raise ValueError(f"Unsupported SuNeRF artifact type: {artifact_type!r}")
        if (
            artifact_type == 'sunerf.plasma.weights'
            and artifact_version != PLASMA_SAFE_ARTIFACT_FORMAT_VERSION
        ):
            raise ValueError(
                f"Unsupported safe plasma artifact format version: {artifact_version!r}"
            )
        if (
            artifact_type == PLASMA_GRID_ARTIFACT_TYPE
            and artifact_version != PLASMA_GRID_ARTIFACT_FORMAT_VERSION
        ):
            raise ValueError(
                f"Unsupported safe plasma-grid artifact format version: {artifact_version!r}"
            )

        data_config = state['data_config']
        self.ds_keys = list(data_config.keys())
        self.config = data_config
        self.observers = [o for k in data_config.keys() if 'observers' in data_config[k] for o in
                          data_config[k]['observers']]

        rendering = state['rendering']
        self.rendering = rendering.to(device).eval()
        model = rendering.fine_model if isinstance(rendering, MultiResolutionRenderingModule) else rendering.model
        self.model = model.to(device).eval()
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.instrument_keys = list(self.rendering.rendering_modules.keys())
        self.instrument_metadata = self._load_instrument_metadata(state.get('instrument_metadata'))
        self.log_T_range = state.get('log_T_range')
        self.temperature_grid = state.get('temperature_grid')
        self.regularization_density_scale_cm3 = state.get(
            'regularization_density_scale_cm3'
        )

        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        self.ref_date = state['ref_date']

        self.ref_maps = {k: Map(np.zeros(self.resolution(k)), self.wcs(k)) for k in self.ds_keys}

    @staticmethod
    def _channel_id_from_cmap(cmap, index):
        match = re.search(r'([0-9]+(?:\.[0-9]+)?)$', str(cmap))
        if match is None:
            return str(index)
        value = float(match.group(1))
        return str(int(value)) if value.is_integer() else str(value)

    def _load_instrument_metadata(self, metadata):
        if metadata is not None:
            missing = [key for key in self.instrument_keys if key not in metadata]
            if missing:
                raise ValueError(f"Artifact is missing channel metadata for instruments: {missing}")
            for instrument_key in self.instrument_keys:
                channels = metadata[instrument_key].get('channels', ())
                channel_ids = [channel.get('id') for channel in channels]
                if not channels or any(channel_id is None for channel_id in channel_ids):
                    raise ValueError(
                        f"Artifact has incomplete channel metadata for '{instrument_key}'."
                    )
                if len(set(channel_ids)) != len(channel_ids):
                    raise ValueError(
                        f"Artifact has duplicate channel IDs for '{instrument_key}': {channel_ids}."
                    )
                rendering_module = self.rendering.rendering_modules[instrument_key]
                response = getattr(rendering_module, 'temperature_response', None)
                if response is not None and int(response.shape[-1]) != len(channels):
                    raise ValueError(
                        f"Artifact declares {len(channels)} channels for '{instrument_key}' "
                        f"but its renderer contains {int(response.shape[-1])}."
                    )
                renderer_channels = getattr(rendering_module, 'channels', None)
                if renderer_channels is None:
                    renderer_channels = metadata[instrument_key].get('response', {}).get(
                        'channels'
                    )
                if renderer_channels is not None:
                    renderer_channels = tuple(str(value) for value in renderer_channels)
                    artifact_response_channels = tuple(
                        str(channel.get('response_channel_id', channel['id']))
                        for channel in channels
                    )
                    if artifact_response_channels != renderer_channels:
                        raise ValueError(
                            f"Artifact response channel order {artifact_response_channels} for "
                            f"'{instrument_key}' does not exactly match renderer channels "
                            f'{renderer_channels}.'
                        )

                channel_response_ids = tuple(
                    str(channel.get('response_id', '')).strip() for channel in channels
                )
                if response is not None and any(not value for value in channel_response_ids):
                    raise ValueError(
                        f"Artifact has missing response IDs for '{instrument_key}'."
                    )
                response_metadata = metadata[instrument_key].get('response', {})
                response_provenance = response_metadata.get('provenance', {})
                expected_response_id = getattr(rendering_module, 'response_id', None)
                expected_response_id = expected_response_id or response_provenance.get(
                    'source_response_id', response_metadata.get(
                        'response_id', response_provenance.get('response_id')
                    )
                )
                if expected_response_id is not None:
                    expected_response_ids = (str(expected_response_id).strip(),) * len(
                        channels
                    )
                    if channel_response_ids != expected_response_ids:
                        raise ValueError(
                            f"Artifact response IDs {channel_response_ids} for "
                            f"'{instrument_key}' do not match renderer response identity "
                            f'{expected_response_ids}.'
                        )

                calibration = metadata[instrument_key].get('calibration')
                if (
                    getattr(self, 'state', {}).get('artifact_type')
                    == 'sunerf.plasma.weights'
                    and calibration is None
                ):
                    raise ValueError(
                        f"Safe plasma artifact is missing response-calibration metadata "
                        f"for '{instrument_key}'."
                    )
                if calibration is not None:
                    if calibration.get('schema') != 'sunerf.response_calibration.v1':
                        raise ValueError(
                            f"Artifact has invalid calibration metadata for '{instrument_key}'."
                        )
                    calibration_channels = tuple(
                        str(value) for value in calibration.get('channel_ids', ())
                    )
                    if calibration_channels != tuple(renderer_channels):
                        raise ValueError(
                            f"Artifact calibration channel order for '{instrument_key}' "
                            "does not match its response channels."
                        )
                    effective_dex = np.asarray(
                        calibration.get('effective_gain_delta_dex', ()),
                        dtype=np.float64,
                    )
                    relative_dex = np.asarray(
                        calibration.get('relative_channel_gain_delta_dex', ()),
                        dtype=np.float64,
                    )
                    multiplicative = np.asarray(
                        calibration.get('effective_multiplicative_gain', ()),
                        dtype=np.float64,
                    )
                    expected_effective = (
                        rendering_module.effective_instrument_scaling.detach()
                        .cpu().numpy().astype(np.float64)
                    )
                    expected_relative = (
                        rendering_module.instrument_gain_delta_dex.detach()
                        .cpu().numpy().astype(np.float64)
                    )
                    expected_common = float(
                        rendering_module.common_gain_delta_dex.detach().cpu()
                    )
                    expected_mode = (
                        'learned' if rendering_module.learnable else 'fixed_nominal'
                    )
                    expected_gauge = (
                        'fixed_global_reference'
                        if rendering_module.global_reference
                        else (
                            'learned_relative_to_global_reference'
                            if rendering_module.learnable
                            else 'nominal_response_amplitude'
                        )
                    )
                    if (
                        effective_dex.shape != expected_effective.shape
                        or relative_dex.shape != expected_relative.shape
                        or multiplicative.shape != expected_effective.shape
                        or not np.all(np.isfinite(effective_dex))
                        or not np.all(np.isfinite(relative_dex))
                        or not np.all(np.isfinite(multiplicative))
                        or not np.allclose(effective_dex, expected_effective, atol=1e-7, rtol=0)
                        or not np.allclose(relative_dex, expected_relative, atol=1e-7, rtol=0)
                        or not np.allclose(multiplicative, 10.0 ** expected_effective, atol=1e-7, rtol=1e-7)
                        or not np.isclose(
                            float(calibration.get('common_gain_delta_dex', np.nan)),
                            expected_common,
                            atol=1e-7,
                            rtol=0,
                        )
                        or bool(calibration.get('global_reference'))
                        != bool(rendering_module.global_reference)
                        or calibration.get('mode') != expected_mode
                        or calibration.get('density_gain_gauge') != expected_gauge
                    ):
                        raise ValueError(
                            f"Artifact calibration metadata for '{instrument_key}' "
                            "does not match its loaded renderer state."
                        )

                matching_configs = [
                    (ds_key, config) for ds_key, config in self.config.items()
                    if config.get('instrument_key', ds_key) == instrument_key
                ]
                artifact_cmaps = tuple(
                    str(channel.get('cmap', '')) for channel in channels
                )
                artifact_units = tuple(
                    str(channel.get('measurement_unit', '')).strip()
                    for channel in channels
                )
                artifact_scaling = metadata[instrument_key].get('image_scaling')
                for ds_key, config in matching_configs:
                    configured_channels = tuple(
                        str(value) for value in config.get('channel_ids', ())
                    )
                    if response is not None and configured_channels != tuple(channel_ids):
                        raise ValueError(
                            f"Dataset '{ds_key}' channel order {configured_channels} does not "
                            f"match artifact metadata {tuple(channel_ids)} for "
                            f"'{instrument_key}'."
                        )
                    configured_cmaps = tuple(
                        str(value) for value in config.get('cmaps', ())
                    )
                    if configured_cmaps and configured_cmaps != artifact_cmaps:
                        raise ValueError(
                            f"Dataset '{ds_key}' colormaps {configured_cmaps} do not "
                            f"match artifact metadata {artifact_cmaps} for "
                            f"'{instrument_key}'."
                        )
                    configured_units = tuple(
                        str(value).strip()
                        for value in config.get('measurement_units', ())
                    )
                    if configured_units and configured_units != artifact_units:
                        raise ValueError(
                            f"Dataset '{ds_key}' measurement units {configured_units} do not "
                            f"match artifact metadata {artifact_units} for "
                            f"'{instrument_key}'."
                        )
                    if (
                        config.get('image_scaling') is not None
                        and config.get('image_scaling') != artifact_scaling
                    ):
                        raise ValueError(
                            f"Dataset '{ds_key}' image scaling does not match artifact "
                            f"metadata for '{instrument_key}'."
                        )
            return metadata

        # One-way compatibility for trusted legacy pickles: reconstruct channel
        # identity from their stored data configuration, never from an instrument
        # name or a hard-coded channel table.
        reconstructed = {}
        for instrument_key in self.instrument_keys:
            matching_configs = [
                config for ds_key, config in self.config.items()
                if config.get('instrument_key', ds_key) == instrument_key
            ]
            cmaps = next((config.get('cmaps') for config in matching_configs if config.get('cmaps')), None)
            rendering_module = self.rendering.rendering_modules[instrument_key]
            response = getattr(rendering_module, 'temperature_response', None)
            n_channels = int(response.shape[-1]) if response is not None else None
            is_thomson = any(config.get('type') == 'thomson' for config in matching_configs)
            if cmaps is None and is_thomson:
                reconstructed[instrument_key] = {
                    'type': 'thomson',
                    'channels': [{'id': 'tB', 'cmap': 'gray'}, {'id': 'pB', 'cmap': 'gray'}],
                }
                continue
            if cmaps is None and n_channels is None:
                raise ValueError(
                    f"Legacy artifact contains no channel metadata for instrument '{instrument_key}'."
                )
            if cmaps is None:
                cmaps = ['gray'] * n_channels
            if n_channels is not None and len(cmaps) != n_channels:
                raise ValueError(
                    f"Legacy artifact has {n_channels} rendered channels but {len(cmaps)} "
                    f"colormaps for instrument '{instrument_key}'."
                )
            reconstructed[instrument_key] = {
                'type': 'legacy',
                'channels': [
                    {
                        'id': self._channel_id_from_cmap(cmap, index),
                        'cmap': str(cmap),
                    }
                    for index, cmap in enumerate(cmaps)
                ],
            }
        return reconstructed

    def channels(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return tuple(channel['id'] for channel in self.instrument_metadata[instrument_key]['channels'])

    def channel_metadata(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return tuple(dict(channel) for channel in self.instrument_metadata[instrument_key]['channels'])

    def dataset_key(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        matches = [
            ds_key for ds_key in self.ds_keys
            if self.instrument_key(ds_key) == instrument_key
        ]
        if not matches:
            raise ValueError(
                f"No dataset metadata maps to instrument '{instrument_key}'. "
                f"Available instruments: {self.instrument_keys}."
            )
        return matches[0]

    def instrument_key(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key].get('instrument_key', ds_key)

    def start_time(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return np.min(self.config[ds_key]['times'])

    def end_time(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return np.max(self.config[ds_key]['times'])

    def times(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['times']

    def wcs(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['wcs']

    def resolution(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['image_shape']

    def ref_map(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.ref_maps[ds_key]

    @torch.no_grad()
    def load_observer_image(self, lat: u, lon: u, time: datetime,
                            distance=(1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            instrument_key=None,
                            **kwargs):
        """Render from an explicitly Carrington ``lat``/``lon`` observer."""
        if center is not None:
            raise NotImplementedError(
                "Offset look-at centers are not supported by pose_spherical."
            )
        observer = SkyCoord(
            lat=lat,
            lon=lon,
            obstime=time,
            radius=distance,
            frame=frames.HeliographicCarrington(observer='self'),
        )
        target_pose = pose_spherical(
            observer.lon.to_value(u.rad),
            observer.lat.to_value(u.rad),
            observer.radius.to_value(u.solRad) / self.Rs_per_ds,
        )
        # load rays
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        ref_map = self.ref_map(self.dataset_key(instrument_key))
        if resolution is not None:
            ref_map = ref_map.resample(resolution)
        img_coords = get_azimuthal_equidistant_coordinates(ref_map)

        pose_out = self.load_pose(
            img_coords, target_pose, time, instrument_key=instrument_key, **kwargs
        )
        scale = [ref_map.scale[0].to_value(u.arcsec / u.pix),
                 ref_map.scale[1].to_value(u.arcsec / u.pix)] * u.arcsec / u.pix

        reference_coord = ref_map.reference_coordinate
        reference_coord = SkyCoord(Tx=reference_coord.Tx, Ty=reference_coord.Ty, obstime=time,
                                   observer=observer, frame=frames.Helioprojective)
        maps = self.get_maps(pose_out['image'], reference_coord, scale, instrument_key)
        pose_out['maps'] = maps
        return pose_out

    @torch.no_grad()
    def load_image(self, lat: u, lon: u,
                   time: datetime,
                   distance=(1 * u.AU).to(u.solRad),
                   hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                   resolution=(256, 256) * u.pix, scale=[2400 / 256, 2400 / 256] * u.arcsec / u.pix,
                   instrument_key=None, **kwargs):
        """Render from an HCI ``lat``/``lon`` observer and preserve it in WCS."""
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]

        obs = SkyCoord(lat=lat, lon=lon, distance=distance,
                       frame=frames.HeliocentricInertial, obstime=time)
        reference_coord = SkyCoord(Tx=hpc_lon, Ty=hpc_lat, obstime=time, observer=obs,
                                   frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution])
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)

        # The public trajectory is specified in HCI, while the neural field and
        # all training rays use Carrington coordinates. Transform the observer
        # before constructing the camera pose; keep the HCI coordinate for WCS.
        target_pose = _carrington_pose_from_hci_observer(obs, self.Rs_per_ds)
        # load image coordinates
        img_coords = get_azimuthal_equidistant_coordinates(ref_map)

        pose_out = self.load_pose(
            img_coords, target_pose, time, instrument_key=instrument_key, **kwargs
        )
        pose_out['maps'] = self.get_maps(pose_out['image'], reference_coord, scale, instrument_key)
        return pose_out

    @torch.no_grad()
    def load_pose(self, img_coords, target_pose, time, batch_size=int(2 ** 10), instrument_key=None, progress=True,
                  model_outputs=[
                      'image', 'mean_T', 'total_ne',
                      'column_electron_density_cm2', 'mean_absorption',
                  ]):
        # load rays
        rays_o, rays_d = get_rays(img_coords[..., 0], img_coords[..., 1], target_pose)
        rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
        img_shape = rays_o.shape[:2]

        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)
        time = self.normalize_datetime(time)
        flat_time = torch.ones_like(flat_rays_o[:, 0:1]) * time

        # make batches
        rays_o, rays_d, time = torch.split(flat_rays_o, batch_size), \
            torch.split(flat_rays_d, batch_size), \
            torch.split(flat_time, batch_size)
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        outputs = {k: [] for k in model_outputs} if model_outputs is not None else {}
        global_outputs = {}
        global_output_names = {
            'calibration_regularization', 'instrument_gain_delta_dex',
            'common_gain_delta_dex',
        }
        lean_outputs = {
            'image', 'weights', 'mean_absorption', 'em', 'distance',
            'calibration_regularization', 'instrument_gain_delta_dex',
            'common_gain_delta_dex',
        }
        diagnostics = model_outputs is None or any(
            key not in lean_outputs for key in model_outputs
        )
        iter = tqdm(zip(rays_o, rays_d, time), total=len(rays_o)) if progress else zip(rays_o, rays_d, time)
        for b_rays_o, b_rays_d, b_time in iter:
            b_rays = torch.stack([b_rays_o, b_rays_d], 1)
            batch = {instrument_key: {'rays': b_rays, 'time': b_time, 'instrument': instrument_key}}
            rendering_out = self.rendering(batch, diagnostics=diagnostics)
            for k, v in rendering_out['model_out'][instrument_key].items():
                if k not in outputs and model_outputs is None:
                    outputs[k] = []
                if k not in outputs:
                    continue
                value = v.detach().cpu()
                if k in global_output_names:
                    if k in global_outputs and not torch.equal(global_outputs[k], value):
                        raise RuntimeError(
                            f'Global renderer output {k!r} changed between ray batches.'
                        )
                    global_outputs[k] = value
                else:
                    outputs[k].append(value)
        missing = sorted(
            key for key, values in outputs.items()
            if not values and key not in global_outputs
        )
        if missing:
            raise ValueError(
                f'Renderer did not produce requested outputs {missing!r}. '
                f'Available outputs: {sorted(set(outputs) - set(missing))!r}.'
            )
        results = {
            key: torch.cat(values).view(
                *img_shape, *values[0].shape[1:]
            ).numpy()
            for key, values in outputs.items()
            if values
        }
        results.update({key: value.numpy() for key, value in global_outputs.items()})
        return results

    def normalize_datetime(self, time):
        if isinstance(time, Iterable):
            return [normalize_datetime(t, self.seconds_per_dt, self.ref_date) for t in time]
        return normalize_datetime(time, self.seconds_per_dt, self.ref_date)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_date)

    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048, progress=False):
        query_points = torch.from_numpy(query_points_npy).float()

        nan_mask = ~torch.isnan(query_points).any(-1)
        flat_query_points = query_points[nan_mask]
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        out_dict = {}
        iter = range(n_batches) if not progress else tqdm(range(n_batches))
        for j in iter:
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)
            for k, v in out.items():
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v.detach().cpu())

        output = {}
        for k in out_dict.keys():
            v = out_dict[k][0]
            out_v = torch.ones(query_points.shape[:-1] + v.shape[1:], dtype=v.dtype) * torch.nan
            out_v[nan_mask] = torch.cat(out_dict[k])
            output[k] = out_v.numpy()

        return output

    def load_spherical(self, latitude_range=None, longitude_range=None, time=None, radius_range=None, **kwargs):
        latitude_range = np.arange(-90, 90, 1) * u.deg if latitude_range is None else latitude_range
        longitude_range = np.arange(0, 360, 1) * u.deg if longitude_range is None else longitude_range
        radius_range = np.linspace(1, 2, 10) * u.solRad if radius_range is None else radius_range

        time = self.ref_date if time is None else time
        time = [time] if not isinstance(time, Iterable) else time
        time = [self.normalize_datetime(t) for t in time]

        coords = np.stack(np.meshgrid(latitude_range.to_value(u.rad),
                                      longitude_range.to_value(u.rad),
                                      radius_range.to_value(u.solRad), time, indexing='ij'), -1)
        x, y, z = convert_spherical_to_cartesian(coords[..., 2], coords[..., 0], coords[..., 1])
        cartesian_coords = np.stack([x, y, z, coords[..., 3]], -1)
        cartesian_coords[..., :3] /= float(self.Rs_per_ds)
        return self.load_coords(cartesian_coords, **kwargs)

    def get_maps(self, channel_images, reference_coord, scale, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        channels = self.instrument_metadata[instrument_key]['channels']
        response_metadata = self.instrument_metadata[instrument_key].get(
            'response', {}
        )
        calibration_metadata = self.instrument_metadata[instrument_key].get(
            'calibration'
        )
        if channel_images.ndim < 3:
            raise ValueError(
                f"Expected a channel-last image cube, got shape {channel_images.shape}."
            )
        if channel_images.shape[-1] != len(channels):
            raise ValueError(
                f"Instrument '{instrument_key}' rendered {channel_images.shape[-1]} channels, "
                f"but its artifact declares {len(channels)}."
            )

        maps = {}
        for i, channel in enumerate(channels):
            img = channel_images[..., i]
            header = make_fitswcs_header(img, reference_coord, scale=scale)
            wavelength = channel.get('wavelength_angstrom')
            if wavelength is not None:
                header['wavelnth'] = wavelength
                header['waveunit'] = 'Angstrom'
            measurement_unit = channel.get('measurement_unit')
            if measurement_unit:
                header['bunit'] = str(measurement_unit)
            response_id = channel.get('response_id')
            if response_id:
                header['resp_id'] = str(response_id)
            response_sha = response_metadata.get('sha256')
            if response_sha:
                header['rsp_sha'] = str(response_sha)
            if calibration_metadata is not None:
                header['calvers'] = str(calibration_metadata['schema'])
                header['rspbase'] = True
                header['calref'] = bool(calibration_metadata['global_reference'])
                header['calmode'] = str(calibration_metadata['mode'])
                header['calgauge'] = str(
                    calibration_metadata['density_gain_gauge']
                )
                header['comgdex'] = float(
                    calibration_metadata['common_gain_delta_dex']
                )
                header['relgdex'] = float(
                    calibration_metadata['relative_channel_gain_delta_dex'][i]
                )
                header['effgdex'] = float(
                    calibration_metadata['effective_gain_delta_dex'][i]
                )
                header['effgain'] = float(
                    calibration_metadata['effective_multiplicative_gain'][i]
                )
            response_provenance = response_metadata.get('provenance', {})
            sensitivity_convention = response_provenance.get(
                'sensitivity_convention'
            )
            if sensitivity_convention:
                header['senscon'] = str(sensitivity_convention)
            calibration_epoch = response_provenance.get('calibration_epoch')
            if calibration_epoch:
                header['cal_epoc'] = str(calibration_epoch)
            measurement_semantics = response_provenance.get(
                'measurement_semantics'
            )
            if measurement_semantics:
                header['radsem'] = str(measurement_semantics)
            response_pixel_solid_angle = response_provenance.get(
                'native_pixel_solid_angle_sr'
            )
            if response_pixel_solid_angle is not None:
                header['rsppxsr'] = float(response_pixel_solid_angle)
            response_pixel_tolerance = response_provenance.get(
                'native_pixel_solid_angle_relative_tolerance'
            )
            if response_pixel_tolerance is not None:
                header['rsppxrt'] = float(response_pixel_tolerance)
            channel_id = channel['id']
            if channel_id in maps:
                raise ValueError(
                    f"Artifact contains duplicate channel ID {channel_id!r} for '{instrument_key}'."
                )
            maps[channel_id] = Map(img, header)
        return maps


class ThomsonSuNeRFLoader(SuNeRFLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'thomson_normalization' not in self.state:
            raise KeyError("Missing 'thomson_normalization' in saved Thomson state.")

        normalization = self.state['thomson_normalization']
        required_keys = ['msb', 'sigma_ne', 'msb_norm', 'drho_cm3']
        missing_keys = [k for k in required_keys if k not in normalization]
        if missing_keys:
            raise KeyError(
                f"Saved Thomson state is missing normalization keys: {', '.join(missing_keys)}"
            )

        self.msb = normalization['msb']  # ph/cm2/s/sr
        self.sigma_ne = normalization['sigma_ne']
        self.msb_norm = normalization['msb_norm']
        self.drho_cm3 = normalization['drho_cm3']
        self.correction_modules = self.state.get('correction_modules', nn.ModuleDict()).to(self.device)
        self.calibration_modules = self.state.get('calibration_modules', nn.ModuleDict()).to(self.device)
        self.correction_modules.eval()
        self.calibration_modules.eval()

    def _get_correction_norms(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        matching_ds_keys = [
            ds_key for ds_key in self.ds_keys
            if self.instrument_key(ds_key) == instrument_key
        ]
        if not matching_ds_keys:
            raise ValueError(
                f"No dataset config maps to instrument_key '{instrument_key}'. "
                f"Available instrument keys: {', '.join(self.instrument_keys)}."
            )

        norms = {
            (
                float(self.config[ds_key].get('image_norm', 512.0)),
                float(self.config[ds_key].get('hpc_norm', 1e4)),
            )
            for ds_key in matching_ds_keys
        }
        if len(norms) != 1:
            raise ValueError(
                f"Dataset configs for instrument_key '{instrument_key}' use inconsistent correction norms: "
                f"{sorted(norms)}."
            )
        return next(iter(norms))

    @staticmethod
    def _get_image_coords(shape, image_norm):
        ny, nx = shape
        image_coords = np.stack(np.mgrid[:ny, :nx], axis=-1).astype(np.float32)
        image_coords[..., 0] -= 0.5 * (ny - 1)
        image_coords[..., 1] -= 0.5 * (nx - 1)
        image_coords /= float(image_norm)
        return image_coords

    def _get_hpc_coords(self, ref_map, hpc_norm):
        coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        x = coords.Tx.to_value(u.arcsec)
        y = coords.Ty.to_value(u.arcsec)
        distance = np.ones_like(x, dtype=np.float32) * ref_map.dsun.to_value(u.solRad)
        hpc_coords = np.stack([x, y, distance], axis=-1).astype(np.float32)
        hpc_coords[..., :2] /= float(hpc_norm)
        hpc_coords[..., 2] /= float(self.Rs_per_ds)
        return hpc_coords

    def _build_correction_inputs(self, ref_map, instrument_key=None):
        image_norm, hpc_norm = self._get_correction_norms(instrument_key)
        image_coords = self._get_image_coords(ref_map.data.shape, image_norm)
        hpc_coords = self._get_hpc_coords(ref_map, hpc_norm)
        time_value = self.normalize_datetime(ref_map.date.datetime)
        time = np.full(ref_map.data.shape + (1,), time_value, dtype=np.float32)
        return image_coords, hpc_coords, time

    @staticmethod
    def _mask_invalid_coords(output, ref_map):
        finite_mask = np.isfinite(ref_map.data)
        for key, value in output.items():
            if value.shape[:2] != finite_mask.shape:
                continue
            value = value.copy()
            value[~finite_mask] = np.nan
            output[key] = value
        return output

    @torch.no_grad()
    def load_correction_masks(self, ref_map, instrument_key=None, apply_valid_mask=True):
        ref_map = Map(ref_map)
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        image_coords, hpc_coords, time = self._build_correction_inputs(ref_map, instrument_key)

        outputs = {}
        correction_module = self.correction_modules[instrument_key] if instrument_key in self.correction_modules else None
        calibration_module = self.calibration_modules[instrument_key] if instrument_key in self.calibration_modules else None

        if correction_module is not None:
            zero_image = torch.zeros(ref_map.data.shape + (2,), dtype=torch.float32, device=self.device)
            _, corrections = correction_module(
                zero_image,
                torch.from_numpy(image_coords).to(self.device),
                torch.from_numpy(hpc_coords).to(self.device),
                torch.from_numpy(time).to(self.device),
            )
            outputs.update({k: v.detach().cpu().numpy()[..., 0] for k, v in corrections.items()})

        if calibration_module is not None:
            calibration_scalar = torch.exp(calibration_module.calibration.detach()).item()
            outputs['instrument_calibration'] = np.full(ref_map.data.shape, calibration_scalar, dtype=np.float32)

        if apply_valid_mask:
            outputs = self._mask_invalid_coords(outputs, ref_map)

        return {k: Map(v.astype(np.float32), ref_map.meta) for k, v in outputs.items()}

    @torch.no_grad()
    def load_correction_image(self, lat: u, lon: u, time: datetime,
                              distance=(1 * u.AU).to(u.solRad),
                              hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                              resolution=(256, 256) * u.pix, scale=None,
                              instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        if scale is None:
            scale = [2400 / resolution[0].to_value(u.pix), 2400 / resolution[1].to_value(u.pix)] * u.arcsec / u.pix

        obs = SkyCoord(lat=lat, lon=lon, distance=distance, frame=frames.HeliocentricInertial, obstime=time)
        reference_coord = SkyCoord(Tx=hpc_lon, Ty=hpc_lat, obstime=time,
                                   observer=obs, frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution], dtype=np.float32)
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)
        return self.load_correction_masks(ref_map, instrument_key=instrument_key, apply_valid_mask=False)

    def convert_rho(self, model_rho):
        # convert to electron density in cm^-3
        physical_rho = model_rho * self.drho_cm3
        return physical_rho

    def convert_column_density(self, model_column_density):
        """Convert ``integral rho_model d(model distance)`` to electrons cm^-2."""
        ds_cm = float(self.Rs_per_ds) * (1 * u.R_sun).to_value(u.cm)
        return model_column_density * self.drho_cm3 * ds_cm

    @torch.no_grad()
    def load_image(self, lat: u, lon: u,
                   time: datetime,
                   distance=(1 * u.AU).to(u.solRad),
                   hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                   resolution=(256, 256) * u.pix, scale=None,
                   occ_min=None, occ_max=None,
                   instrument_key=None, **kwargs):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        if scale is None:
            if occ_max is not None:
                scale = _get_scale_from_occ_max(occ_max, distance, resolution)
            else:
                scale = [2400 / 256, 2400 / 256] * u.arcsec / u.pix

        obs = SkyCoord(lat=lat, lon=lon, distance=distance, frame=frames.HeliocentricInertial, obstime=time)
        reference_coord = SkyCoord(Tx=hpc_lon, Ty=hpc_lat, obstime=time, observer=obs,
                                   frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution])
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)

        # apply occulter mask
        mask = _get_mask(ref_map, occ_min, occ_max)
        ref_map.data[mask] = np.nan

        return self.load_map(ref_map, instrument_key=instrument_key, **kwargs)

    @torch.no_grad()
    def load_map(self, ref_map, filter_occ=True, **kwargs):
        map_loader = MapDataLoader(self.Rs_per_ds, 'inertial', azimuthal_equidistant=False)
        map_data = map_loader.load(ref_map)  # image, pose, rays, time, observer
        # convert to pose
        target_pose = pose_spherical(map_data['observer']['longitude'].to_value(u.rad),
                                     map_data['observer']['latitude'].to_value(u.rad),
                                     map_data['observer']['radius'].to_value(u.solRad) / self.Rs_per_ds)
        # load image coordinates
        img_coords = all_coordinates_from_map(ref_map)
        img_coords = np.stack([img_coords.Tx, img_coords.Ty], -1)

        # occulter mask
        if filter_occ:
            mask = np.isnan(ref_map.data)
            img_coords[mask] = np.nan

        pose_out = self.load_pose(img_coords, target_pose, map_data['observer']['time'],
                                  model_outputs=['image', 'density'], **kwargs)

        # create maps
        tB_map = Map(pose_out['image'][..., 0], ref_map.meta)
        pB_map = Map(pose_out['image'][..., 1], ref_map.meta)
        density_map = Map(pose_out['density'], ref_map.meta)
        #
        return {'tB_map': tB_map, 'pB_map': pB_map, 'density_map': density_map}

    def load_spherical_cube(self, radius, latitude, longitude, time, **kwargs):
        spherical_coords = np.stack(np.meshgrid(
            radius.to_value(u.R_sun),
            latitude.to_value(u.rad),
            longitude.to_value(u.rad),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_latitude(self, radius_range, time, latitude, Nr=128, Nphi=128, longitude_range=None, **kwargs):
        longitude_range = [0, 2 * np.pi] * u.rad if longitude_range is None else longitude_range
        spherical_coords = np.stack(np.meshgrid(
            np.linspace(radius_range[0].to_value(u.R_sun), radius_range[1].to_value(u.R_sun), Nr),
            latitude.to_value(u.rad),
            np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), Nphi, endpoint=False),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_longitude(self, radius_range, time, longitude, latitude_range=None, Nr=128, Ntheta=128, **kwargs):
        latitude_range = [0, 2 * np.pi] * u.rad if latitude_range is None else latitude_range
        spherical_coords = np.stack(np.meshgrid(
            np.linspace(radius_range[0].to_value(u.R_sun), radius_range[1].to_value(u.R_sun), Nr),
            np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), Ntheta, endpoint=False),
            longitude.to_value(u.rad),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_radius(self, radius, time, Ntheta=128, Nphi=256, projection='lat', **kwargs):
        if projection == 'lat':
            latitude = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
            latitude_axis = np.rad2deg(latitude)
        elif projection == 'sinlat':
            latitude_axis = np.linspace(-1.0, 1.0, Ntheta)
            latitude = np.arcsin(latitude_axis)
        else:
            raise ValueError(f"Unsupported projection '{projection}'. Use 'lat' or 'sinlat'.")

        coords = np.stack(np.meshgrid(
            radius.to_value(u.R_sun),
            latitude,
            np.linspace(0, 2 * np.pi, Nphi, endpoint=False),
            [1],
            indexing='ij'
        ), -1)
        sky_coords = SkyCoord(radius=coords[..., 0] * u.R_sun,
                              lat=coords[..., 1] * u.rad,
                              lon=coords[..., 2] * u.rad,
                              frame=frames.HeliographicCarrington, obstime=time,
                              observer='self')
        sky_coords = sky_coords.transform_to(frames.HeliocentricInertial)

        spherical_coords = np.stack([sky_coords.distance.to_value(u.R_sun),
                                     sky_coords.lat.to_value(u.rad),
                                     sky_coords.lon.to_value(u.rad)], axis=-1)
        cartesian_coords = spherical_to_cartesian(spherical_coords, np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        time_coords = np.ones_like(cartesian_coords[..., 0:1]) * self.normalize_datetime(time)
        query_points = np.concatenate([cartesian_coords, time_coords], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {
            'rho': rho,
            'v': v,
            'spherical_coords': spherical_coords,
            'projection': projection,
            'latitude_axis': latitude_axis
        }

    def load_coords(self, *args, **kwargs):
        output = super().load_coords(*args, **kwargs)
        # convert rho to physical units
        output['rho'] = self.convert_rho(output['rho'])
        output['log_rho'] = np.log(output['rho'])
        output['v'] = output['v'] * (self.Mm_per_ds / self.seconds_per_dt) * 1e3  # convert to km/s

        return output

    def load_cube(self, radius_range: u.solRad, time, pixel_per_Rs, **kwargs):
        radius_range = radius_range.to_value(u.R_sun)
        max_radius = radius_range[1]
        #
        cartesian_coords = np.stack(np.meshgrid(
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        # only load the points in the radius range
        r = np.linalg.norm(cartesian_coords[..., :3], axis=-1)
        mask = (r >= radius_range[0]) & (r <= radius_range[1])
        sub_coords = cartesian_coords[mask]

        # normalize coordinates
        sub_coords[..., 0:3] = sub_coords[..., 0:3] / self.Rs_per_ds
        # load the coordinates
        model_out = self.load_coords(sub_coords, **kwargs)
        rho = model_out['rho']
        v = model_out['v']

        rho_cube = np.zeros((*cartesian_coords.shape[:-1],))
        rho_cube[mask] = rho.squeeze(-1)

        v_cube = np.zeros((*cartesian_coords.shape[:-1], 3))
        v_cube[mask] = v

        return {'rho': rho_cube, 'v': v_cube, 'cartesian_coords': cartesian_coords}

    def load_slice(self, radius_range, time, z, pixel_per_Rs, **kwargs):
        max_radius = radius_range[1].to_value(u.R_sun)
        #
        cartesian_coords = np.stack(np.meshgrid(
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            z,
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        # only load the points in the radius range
        r = np.linalg.norm(cartesian_coords[..., :3], axis=-1)
        mask = (r >= radius_range[0].to_value(u.R_sun)) & (r <= radius_range[1].to_value(u.R_sun))
        sub_coords = cartesian_coords[mask]

        # normalize coordinates
        sub_coords[..., 0:3] = sub_coords[..., 0:3] / self.Rs_per_ds
        # load the coordinates
        model_out = self.load_coords(sub_coords, **kwargs)
        rho = model_out['rho']
        v = model_out['v']

        rho_cube = np.zeros((*cartesian_coords.shape[:-1],))
        rho_cube[mask] = rho.squeeze(-1)

        v_cube = np.zeros((*cartesian_coords.shape[:-1], 3))
        v_cube[mask] = v

        return {'rho': rho_cube, 'v': v_cube, 'cartesian_coords': cartesian_coords}

    def load_pose(self, *args, **kwargs):
        output = super().load_pose(*args, **kwargs)
        # convert image
        output['image'] = output['image'] * self.msb_norm
        if 'density' in output:
            output['density'] = self.convert_column_density(output['density'])
        return output


class PlasmaSuNeRFLoader(SuNeRFLoader):

    def __init__(self, state_path, *args, **kwargs):
        super().__init__(state_path, *args, trusted=False, **kwargs)
        if self.log_T_range is None:
            raise KeyError("Plasma artifact is missing 'log_T_range'.")


def _get_scale_from_occ_max(occ_max, distance, resolution):
    occ_max = u.Quantity(occ_max).to(u.R_sun)
    distance = u.Quantity(distance).to(u.R_sun)
    # NumPy/SunPy image shapes are ordered (y, x), while WCS scale is (x, y).
    ny = int(resolution[0].to_value(u.pix))
    nx = int(resolution[1].to_value(u.pix))

    impact_ratio = (occ_max / distance).to_value(u.one)
    if not 0 < impact_ratio < 1:
        raise ValueError("occ_max must be positive and smaller than the observer distance.")

    # FITS TAN uses a gnomonic projection-plane coordinate.  Convert the
    # desired physical impact parameter to its exact sky angle and then to the
    # corresponding tangent-plane radius at the outermost pixel centers.
    sky_radius = np.arcsin(impact_ratio)
    tan_plane_radius = (np.tan(sky_radius) * u.rad).to(u.arcsec)
    x_half_span = max(nx - 1, 1) / 2
    y_half_span = max(ny - 1, 1) / 2

    return u.Quantity(
        [tan_plane_radius / x_half_span, tan_plane_radius / y_half_span]
    ) / u.pix


def _get_mask(s_map, occ_min, occ_max):
    # mask occultor
    img_coords = all_coordinates_from_map(s_map)
    x = img_coords.Tx
    y = img_coords.Ty
    impact_radius = hpc_impact_parameter(x, y, s_map.dsun).to(u.R_sun)

    mask = np.zeros(x.shape, dtype=bool)
    if occ_min is not None:
        occ_min_cond = impact_radius < u.Quantity(occ_min).to(u.R_sun)
        mask[occ_min_cond] = True
    if occ_max is not None:
        occ_max_cond = impact_radius > u.Quantity(occ_max).to(u.R_sun)
        mask[occ_max_cond] = True
    return mask
