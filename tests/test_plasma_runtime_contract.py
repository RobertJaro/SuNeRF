from types import SimpleNamespace

import numpy as np
import pytest

from sunerf.response import ResponseArtifact
from sunerf.run_plasma import (
    validate_configured_response_artifacts,
    validate_plasma_runtime_contract,
)


def _runtime_objects(*, prepared_channels=('94', '171'), measurement_units=('DN / s', 'DN / s'),
                     sensitivity_conventions=('reference_epoch', 'reference_epoch'),
                     measurement_semantics=('per_native_pixel', 'per_native_pixel'),
                     native_pixel_solid_angle_sr=(8.46e-12, 8.46e-12)):
    renderer = SimpleNamespace(
        channels=('A94', 'A171'),
        response_unit='cm5 DN s-1 pix-1',
        response_provenance={
            'sensitivity_convention': 'reference_epoch',
            'measurement_semantics': 'per_native_pixel',
            'native_pixel_solid_angle_sr': 8.46e-12,
            'native_pixel_solid_angle_relative_tolerance': 0.01,
        },
    )
    sunerf = SimpleNamespace(
        rendering=SimpleNamespace(rendering_modules={'AIA': renderer})
    )
    data_module = SimpleNamespace(config={
        'aia_train': {
            'channel_ids': prepared_channels,
            'measurement_units': measurement_units,
            'sensitivity_conventions': sensitivity_conventions,
            'measurement_semantics': measurement_semantics,
            'native_pixel_solid_angle_sr': native_pixel_solid_angle_sr,
        },
    })
    data_config = {
        'train_datasets': [{
            'key': 'aia_train',
            'instrument_key': 'AIA',
        }],
        'valid_datasets': [{
            'key': 'aia_train',
            'instrument_key': 'AIA',
        }],
    }
    instruments_config = [{
        'key': 'AIA',
        'temperature_response': {'response_id': 'response-v1'},
    }]
    return sunerf, data_module, data_config, instruments_config


def test_runtime_contract_accepts_physical_units_and_records_instrument_mapping():
    objects = _runtime_objects()

    assert validate_plasma_runtime_contract(*objects)
    assert objects[1].config['aia_train']['instrument_key'] == 'AIA'


def test_runtime_contract_rejects_reordered_channels():
    objects = _runtime_objects(prepared_channels=('171', '94'))

    with pytest.raises(ValueError, match='channel order'):
        validate_plasma_runtime_contract(*objects)


def test_runtime_contract_rejects_incompatible_measurement_units():
    objects = _runtime_objects(measurement_units=('W / m2', 'DN / s'))

    with pytest.raises(ValueError, match='incompatible with response output'):
        validate_plasma_runtime_contract(*objects)


def test_runtime_contract_rejects_sensitivity_convention_mismatch():
    objects = _runtime_objects(
        sensitivity_conventions=('reference_epoch', 'native_epoch')
    )

    with pytest.raises(ValueError, match='sensitivity conventions'):
        validate_plasma_runtime_contract(*objects)


def test_runtime_contract_accepts_static_assumed_instrument_response():
    objects = _runtime_objects(
        sensitivity_conventions=('static_assumed', 'static_assumed')
    )
    renderer = objects[0].rendering.rendering_modules['AIA']
    renderer.response_provenance['sensitivity_convention'] = 'static_assumed'

    assert validate_plasma_runtime_contract(*objects)


def test_runtime_contract_honors_explicit_response_measurement_unit():
    objects = list(_runtime_objects(measurement_units=('ct / s', 'ct / s')))
    objects[3][0]['temperature_response']['measurement_unit'] = 'ct / s'

    assert validate_plasma_runtime_contract(*objects)


def test_runtime_contract_uses_immutable_source_response_id_from_provenance():
    objects = list(_runtime_objects())
    objects[3][0]['temperature_response'].pop('response_id')
    renderer = objects[0].rendering.rendering_modules['AIA']
    renderer.response_provenance = {
        'response_id': 'runtime-interpolation-sha',
        'source_response_id': 'artifact-sha',
        'sensitivity_convention': 'reference_epoch',
        'measurement_semantics': 'per_native_pixel',
        'native_pixel_solid_angle_sr': 8.46e-12,
        'native_pixel_solid_angle_relative_tolerance': 0.01,
    }

    assert validate_plasma_runtime_contract(*objects)


def test_runtime_contract_rejects_native_pixel_radiometry_mismatch():
    objects = _runtime_objects(native_pixel_solid_angle_sr=(8.46e-12, 1.0e-11))

    with pytest.raises(ValueError, match='native pixel solid angles'):
        validate_plasma_runtime_contract(*objects)


def test_configured_response_artifacts_fail_fast_on_calibration_contract(tmp_path):
    artifact = ResponseArtifact(
        channels=('A94',),
        log_temperature=np.array([5.0, 6.0]),
        response=np.ones((1, 2)),
        response_unit='cm5 DN / (pix s)',
        emission_measure_convention='ne2',
        provenance={
            'sensitivity_convention': 'reference_epoch',
            'calibration_epoch': '2025-01-01T00:00:00Z',
            'measurement_semantics': 'per_native_pixel',
            'native_pixel_solid_angle_sr': 8.46e-12,
            'native_pixel_solid_angle_relative_tolerance': 0.02,
        },
    )
    path = tmp_path / 'response.npz'
    artifact.save(path)
    instruments = [{
        'key': 'AIA',
        'temperature_response': {
            'artifact': str(path),
            'channels': [94],
        },
    }]

    validated = validate_configured_response_artifacts(instruments)
    assert validated['AIA']['channels'] == ('A94',)

    static = artifact.updated(provenance={
        **artifact.provenance,
        'sensitivity_convention': 'static_assumed',
    })
    static.save(path)
    assert validate_configured_response_artifacts(instruments)['AIA']['channels'] == (
        'A94',
    )

    invalid = artifact.updated(provenance={
        'sensitivity_convention': 'reference_epoch',
        'calibration_epoch': '2025-01-01T00:00:00Z',
        'measurement_semantics': 'per_native_pixel',
    })
    invalid.save(path)
    with pytest.raises(ValueError, match='native-pixel solid-angle contract'):
        validate_configured_response_artifacts(instruments)
