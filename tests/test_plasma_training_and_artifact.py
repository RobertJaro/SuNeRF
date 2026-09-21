from types import SimpleNamespace
import copy
import hashlib
from datetime import datetime

import numpy as np
import pytest
import torch
from astropy.wcs import WCS
from torch import nn

from sunerf.model.plasma import (
    PLASMA_SAFE_ARTIFACT_FORMAT_VERSION,
    PlasmaSuNeRFModule,
    build_log_temperature_grid,
    save_plasma_sunerf,
)
from sunerf.evaluation.loader import PlasmaSuNeRFLoader
from sunerf.response import ResponseArtifact


class _LossHarness:
    _expanded_valid_mask = staticmethod(PlasmaSuNeRFModule._expanded_valid_mask)
    _masked_image_loss = PlasmaSuNeRFModule._masked_image_loss
    compute_radial_density_regularization = PlasmaSuNeRFModule.compute_radial_density_regularization
    compute_static_regularization = PlasmaSuNeRFModule.compute_static_regularization

    def __init__(self):
        self.channel_loss_weights = {'A': [1.0, 1.0]}
        self.regularization_density_scale_cm3 = torch.tensor(1e8)


def test_masked_image_loss_averages_channels_before_combining_them():
    harness = _LossHarness()
    prediction = torch.tensor([
        [1.0, 3.0],
        [1.0, 999.0],
        [1.0, 999.0],
        [1.0, 999.0],
    ])
    target = torch.zeros_like(prediction)
    valid_mask = torch.tensor([
        [True, True],
        [True, False],
        [True, False],
        [True, False],
    ])

    loss, returned_mask = harness._masked_image_loss(
        prediction, target, valid_mask, 'A'
    )

    # Channel MSEs are 1 and 9. Equal channel weighting gives 5; a flattened
    # valid-pixel average would incorrectly give 13/5.
    torch.testing.assert_close(loss, torch.tensor(5.0))
    torch.testing.assert_close(returned_mask, valid_mask)


def test_masked_image_loss_rejects_nonfinite_supervised_predictions():
    harness = _LossHarness()
    prediction = torch.tensor([[1.0, torch.nan]])
    target = torch.zeros_like(prediction)

    with pytest.raises(FloatingPointError, match='non-finite supervised'):
        harness._masked_image_loss(
            prediction, target, torch.ones_like(target, dtype=torch.bool), 'A'
        )


def test_static_density_and_temperature_derivatives_cannot_cancel():
    harness = _LossHarness()
    query_points = torch.tensor(
        [[2.0, 0.0, 0.0, 0.5], [2.0, 0.0, 0.0, -0.5]],
        requires_grad=True,
    )
    total_ne = query_points[:, 3:4]
    mean_log_temperature = -query_points[:, 3:4]

    regularization = harness.compute_static_regularization(
        total_ne, mean_log_temperature, query_points
    )

    radial_weight = (2.0 - 1.1) ** 2
    expected = 2 * radial_weight ** 2
    torch.testing.assert_close(regularization, torch.tensor(expected))
    assert regularization > 0


def test_radial_density_regularization_is_invariant_to_matching_unit_scale():
    harness = _LossHarness()
    distance = torch.tensor([[1.3, 1.5]])
    emission_measure_density = torch.tensor([[1e16, 4e16]])
    baseline = harness.compute_radial_density_regularization(
        emission_measure_density, distance
    )

    harness.regularization_density_scale_cm3 = torch.tensor(1e9)
    rescaled = harness.compute_radial_density_regularization(
        emission_measure_density * 100, distance
    )

    torch.testing.assert_close(rescaled, baseline)


def test_temperature_grid_is_explicit_inclusive_and_exactly_spaced():
    grid = build_log_temperature_grid({
        'log10_K_min': 4.0,
        'log10_K_max': 4.2,
        'step_dex': 0.05,
    })

    np.testing.assert_allclose(grid, [4.0, 4.05, 4.1, 4.15, 4.2])
    with pytest.raises(ValueError, match='integer number'):
        build_log_temperature_grid({
            'log10_K_min': 4.0,
            'log10_K_max': 4.2,
            'step_dex': 0.07,
        })


def test_plasma_artifact_is_atomic_versioned_and_carries_channel_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response_path = tmp_path / 'response.npz'
    response_path.write_bytes(b'response artifact fixture')
    response_sha256 = hashlib.sha256(response_path.read_bytes()).hexdigest()
    model = SimpleNamespace(
        rendering=nn.Linear(1, 1),
        state_dict=lambda: {'rendering.weight': torch.ones(1)},
        construction_spec={
            'fixture': True,
            'Rs_per_ds': 1.0,
            'seconds_per_dt': 86400.0,
        },
        temperature_grid_config={
            'log10_K_min': 5.0,
            'log10_K_max': 6.0,
            'step_dex': 1.0,
        },
        regularization_density_scale_cm3=torch.tensor(1e8),
        log_T_range=np.array([5.0, 6.0], dtype=np.float32),
        instrument_metadata={
            'AIA': {
                'type': 'plasma',
                'channels': [
                    {'id': 'A94', 'wavelength_angstrom': 94},
                    {'id': 'A171', 'wavelength_angstrom': 171},
                ],
                'response': {
                    'artifact': str(response_path),
                    'path': str(response_path),
                    'sha256': response_sha256,
                    'provenance': {'response_id': 'fixture-response'},
                },
            }
        },
        config_schema_version=2,
        config_fingerprint='config-sha256',
        config_source_records=[{'path': '/input/config-source', 'size': 12, 'mtime_ns': 34}],
    )
    data_module = SimpleNamespace(
        config={
            'aia': {
                'instrument_key': 'AIA',
                'channel_ids': ('94', '171'),
                'cmaps': ('sdoaia94', 'sdoaia171'),
                'measurement_units': ('DN/s', 'DN/s'),
            }
        },
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        ref_date=None,
        cache_format_version=3,
        cache_fingerprint='data-sha256',
        cache_source_records=[{'path': '/input/observation.fits', 'size': 56, 'mtime_ns': 78}],
    )

    safe_path = save_plasma_sunerf(model, data_module, 'state.safe.pt')

    saved = torch.load(safe_path, weights_only=True)
    assert saved['artifact_type'] == 'sunerf.plasma.weights'
    assert saved['artifact_format_version'] == PLASMA_SAFE_ARTIFACT_FORMAT_VERSION
    channels = saved['instrument_metadata']['AIA']['channels']
    assert [channel['id'] for channel in channels] == ['94', '171']
    assert [channel['response_channel_id'] for channel in channels] == ['A94', 'A171']
    provenance = saved['provenance']
    assert provenance['config_schema_version'] == 2
    assert provenance['config_fingerprint'] == 'config-sha256'
    assert provenance['data_cache_fingerprint'] == 'data-sha256'
    assert provenance['config_source_records'][0]['path'] == '/input/config-source'
    assert provenance['data_source_records'][0]['path'] == '/input/observation.fits'
    response_reference = provenance['response_artifacts']['AIA']
    assert response_reference['sha256'] == response_sha256
    assert response_reference['provenance']['response_id'] == 'fixture-response'
    assert saved['artifact_security'] == 'weights_only'
    assert saved['regularization_density_scale_cm3'] == 1e8
    assert not list(tmp_path.glob('state.safe.pt.tmp-*'))


def test_plasma_loader_prefers_weights_only_safe_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    response_path = tmp_path / 'response.npz'
    response_artifact = ResponseArtifact(
        channels=('A94',),
        log_temperature=np.array([5.0, 6.0]),
        response=np.ones((1, 2)),
        response_unit='cm5 DN / (pix s)',
        emission_measure_convention='ne2',
        provenance={'provider': 'test-fixture'},
    )
    response_artifact.save(response_path)
    module = PlasmaSuNeRFModule(
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        instruments_config=[
            {
                'key': 'AIA',
                'type': 'plasma',
                'temperature_response': {
                    'artifact': response_path.name,
                    'channels': ['A94'],
                    'response_id': response_artifact.response_id,
                    'learnable': True,
                    'global_reference': True,
                    'common_gain_limit_dex': 1.0,
                },
                'scaling': {'type': 'linear', 'divisor': [25.0]},
            },
            {
                'key': 'EUVI-A',
                'type': 'plasma',
                'temperature_response': {
                    'artifact': response_path.name,
                    'channels': ['A94'],
                    'response_id': response_artifact.response_id,
                    'learnable': True,
                    'common_gain_limit_dex': 1.0,
                },
                'scaling': {'type': 'linear', 'divisor': [50.0]},
            },
        ],
        model_config={
            'type': 'siren',
            'dim': 4,
            'n_layers': 2,
            'density_offset_log10_cm3': 8.0,
            'temperature_grid': {
                'log10_K_min': 5.0,
                'log10_K_max': 6.0,
                'step_dex': 1.0,
            },
        },
        sampling_config={
            'type': 'spherical',
            'min_distance': 1.0,
            'max_distance': 1.5,
            'n_samples': 2,
            'perturb': False,
        },
        hierarchical_sampling_config={
            'type': 'hierarchical',
            'n_samples': 2,
            'perturb': False,
        },
        absorption_config={'type': None},
        lambda_config={'regularization': 0.0},
        regularization_density_scale_cm3=1e8,
        validation_dataset_mapping={},
    )
    module.config_schema_version = 2
    module.config_fingerprint = 'config-fingerprint'
    module.config_source_records = []
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [2, 2]
    wcs.wcs.cdelt = [-0.01, 0.01]
    wcs.wcs.crval = [0, 0]
    wcs.wcs.ctype = ['HPLN-TAN', 'HPLT-TAN']
    wcs.wcs.cunit = ['deg', 'deg']
    data_module = SimpleNamespace(
        config={
            'aia': {
                'type': 'plasma',
                'instrument_key': 'AIA',
                'wcs': wcs,
                'image_shape': (3, 4),
                'times': [datetime(2025, 1, 1)],
                'channel_ids': ('94',),
                'cmaps': ('sdoaia94',),
                'measurement_units': ('DN / s',),
            },
            'euvi': {
                'type': 'plasma',
                'instrument_key': 'EUVI-A',
                'wcs': wcs,
                'image_shape': (3, 4),
                'times': [datetime(2025, 1, 1)],
                'channel_ids': ('94',),
                'cmaps': ('gray',),
                'measurement_units': ('DN / s',),
            },
        },
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        ref_date=datetime(2025, 1, 1),
        cache_format_version=3,
        cache_fingerprint='data-fingerprint',
        cache_source_records=[],
    )
    renderer = module.rendering.rendering_modules['EUVI-A']
    assert renderer.common_instrument_scaling.requires_grad
    with torch.no_grad():
        renderer.common_instrument_scaling.fill_(0.4)
    safe_path = save_plasma_sunerf(module, data_module, tmp_path / 'model.safe.pt')

    safe_state = torch.load(safe_path, weights_only=True)
    saved_response_path = safe_state['construction_spec']['instruments_config'][0][
        'temperature_response'
    ]['artifact']
    assert saved_response_path == str(response_path)
    euvi_calibration = safe_state['instrument_metadata']['EUVI-A']['calibration']
    assert euvi_calibration['schema'] == 'sunerf.response_calibration.v1'
    assert euvi_calibration['mode'] == 'learned'
    assert not euvi_calibration['global_reference']
    assert euvi_calibration['effective_multiplicative_gain'][0] > 1
    assert (
        safe_state['provenance']['response_calibration']['EUVI-A']
        == euvi_calibration
    )
    for suffix in ('temperature_response', 'instrument_scaling_center'):
        state_key = next(key for key in safe_state['state_dict'] if key.endswith(suffix))
        immutable_tamper = copy.deepcopy(safe_state)
        immutable_tamper['state_dict'][state_key] = (
            immutable_tamper['state_dict'][state_key] + 1
        )
        immutable_tamper_path = tmp_path / f'tampered-{suffix}.safe.pt'
        torch.save(immutable_tamper, immutable_tamper_path)
        with pytest.raises(ValueError, match='immutable physics state'):
            PlasmaSuNeRFLoader(immutable_tamper_path, device='cpu')
    calibration_tamper = copy.deepcopy(safe_state)
    calibration_tamper['instrument_metadata']['EUVI-A']['calibration'][
        'effective_gain_delta_dex'
    ][0] += 0.1
    calibration_tamper_path = tmp_path / 'tampered-calibration.safe.pt'
    torch.save(calibration_tamper, calibration_tamper_path)
    with pytest.raises(ValueError, match='does not match its loaded renderer state'):
        PlasmaSuNeRFLoader(calibration_tamper_path, device='cpu')
    tampered_path = tmp_path / 'tampered.safe.pt'
    tampered_state = copy.deepcopy(safe_state)
    tampered_state['Rs_per_ds'] = 2.0
    torch.save(tampered_state, tampered_path)
    with pytest.raises(ValueError, match='Rs_per_ds mismatch'):
        PlasmaSuNeRFLoader(tampered_path, device='cpu')
    other_cwd = tmp_path / 'other-cwd'
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    loader = PlasmaSuNeRFLoader(safe_path, device='cpu')

    assert loader.loaded_safely is True
    assert loader.loaded_artifact_path == safe_path
    assert loader.state['artifact_security'] == 'weights_only'
    assert loader.dataset_key('AIA') == 'aia'
    assert loader.ref_map('aia').data.shape == (3, 4)
    assert loader.start_time('aia') == datetime(2025, 1, 1)
    assert 'image_scaling' not in loader.config['aia']
    assert loader.instrument_metadata['AIA']['image_scaling']['divisor'] == [25.0]
    assert loader.instrument_metadata['EUVI-A']['image_scaling']['divisor'] == [50.0]
    torch.testing.assert_close(
        loader._reconstructed_module.rendering.rendering_modules[
            'EUVI-A'
        ].common_instrument_scaling,
        torch.tensor(0.4),
    )
    expected = module.state_dict()
    reconstructed = loader._reconstructed_module.state_dict()
    assert expected.keys() == reconstructed.keys()
    for key in expected:
        torch.testing.assert_close(reconstructed[key], expected[key])

    wrong_channels = copy.deepcopy(loader.instrument_metadata)
    wrong_channels['AIA']['channels'][0]['response_channel_id'] = 'A171'
    with pytest.raises(ValueError, match='does not exactly match renderer channels'):
        loader._load_instrument_metadata(wrong_channels)

    wrong_response_id = copy.deepcopy(loader.instrument_metadata)
    wrong_response_id['AIA']['channels'][0]['response_id'] = 'different-response'
    with pytest.raises(ValueError, match='do not match renderer response identity'):
        loader._load_instrument_metadata(wrong_response_id)

    data_module.seconds_per_dt = 1.0
    with pytest.raises(ValueError, match='module seconds_per_dt'):
        save_plasma_sunerf(module, data_module, tmp_path / 'invalid-scale.snf')
