"""Module calibration rules must agree with validation-only configuration support."""
import copy

import numpy as np
import pytest
import torch

from sunerf.model.plasma import PlasmaSuNeRFModule, _validate_instrument_calibration_gauge
from sunerf.response import ResponseArtifact


def instruments():
    return [
        {'key': 'AIA', 'type': 'plasma', 'temperature_response': {'learnable': True, 'global_reference': True}},
        {'key': 'EUVI-A', 'type': 'plasma', 'temperature_response': {'learnable': True}},
        {'key': 'EUI', 'type': 'plasma', 'temperature_response': {'learnable': False}},
    ]


def test_fixed_validation_instrument_does_not_anchor_training_calibration():
    config = instruments()
    original = copy.deepcopy(config)
    assert _validate_instrument_calibration_gauge(config, ['AIA', 'EUVI-A']) == ('AIA', 'EUVI-A')
    assert config == original
    # Direct callers without supervision metadata retain the strict old policy.
    with pytest.raises(ValueError, match='fixed anchors'):
        _validate_instrument_calibration_gauge(config)
    with pytest.raises(ValueError, match='fixed anchors'):
        _validate_instrument_calibration_gauge(config, ['AIA', 'EUVI-A', 'EUI'])


@pytest.mark.parametrize('keys', [[], ['missing']])
def test_invalid_supervision_is_rejected(keys):
    with pytest.raises(ValueError, match='nonempty subset'):
        _validate_instrument_calibration_gauge(instruments(), keys)


def test_held_out_reference_cannot_fix_training_gauge():
    config = instruments()
    config[0]['temperature_response']['global_reference'] = False
    config[2]['temperature_response'].update(learnable=True, global_reference=True)
    with pytest.raises(ValueError, match='supervised instrument'):
        _validate_instrument_calibration_gauge(config, ['AIA', 'EUVI-A'])


@pytest.mark.parametrize('held_out_learnable', [False, True])
def test_module_freezes_held_out_gains_and_preserves_scope_on_reconstruction(tmp_path, held_out_learnable):
    path = tmp_path / 'response.npz'
    ResponseArtifact(channels=('A94',), log_temperature=np.array([5., 6.]),
                     response=np.ones((1, 2)), response_unit='cm5 DN / (pix s)',
                     emission_measure_convention='ne2', provenance={'provider': 'fixture'}).save(path)
    config = instruments()
    config[-1]['temperature_response']['learnable'] = held_out_learnable
    for instrument in config:
        instrument['temperature_response'].update(artifact=str(path), channels=['A94'])
        instrument['scaling'] = {'type': 'linear', 'divisor': 1.}
    module = PlasmaSuNeRFModule(
        Rs_per_ds=1., seconds_per_dt=86400., instruments_config=config,
        supervised_instrument_keys=['AIA', 'EUVI-A'], validation_dataset_mapping={},
        model_config={'type': 'siren', 'dim': 4, 'n_layers': 2, 'density_offset_log10_cm3': 8.,
                      'temperature_grid': {'log10_K_min': 5., 'log10_K_max': 6., 'step_dex': 1.}},
        sampling_config={'type': 'spherical', 'min_distance': 1., 'max_distance': 1.5, 'n_samples': 2, 'perturb': False},
        hierarchical_sampling_config={'type': 'hierarchical', 'n_samples': 2, 'perturb': False},
        absorption_config={'type': None}, lambda_config={'regularization': 0.}, regularization_density_scale_cm3=1e8,
    )
    restored = PlasmaSuNeRFModule(**module.construction_spec)
    restored.load_state_dict(module.state_dict())
    for model in (module, restored):
        renderers = model.rendering.rendering_modules
        assert renderers['EUVI-A'].common_instrument_scaling.requires_grad
        assert not renderers['AIA'].common_instrument_scaling.requires_grad
        assert not renderers['EUI'].learnable
        assert not renderers['EUI'].instrument_scaling.requires_grad
        assert not renderers['EUI'].common_instrument_scaling.requires_grad
        assert torch.count_nonzero(renderers['EUI'].instrument_scaling) == 0
        assert model.construction_spec['supervised_instrument_keys'] == ['AIA', 'EUVI-A']
