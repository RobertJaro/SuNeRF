import copy

import pytest
import yaml

from sunerf.configuration import (
    PlasmaConfigError,
    apply_instrument_saturation,
    validate_plasma_config,
)


def _config():
    with open('config/plasma/euv_wave_2022_03.yaml') as stream:
        return yaml.safe_load(stream)


def test_instrument_saturation_level_reaches_every_dataset_of_the_instrument():
    config = validate_plasma_config(_config())
    data = apply_instrument_saturation(config)
    levels = {d['key']: d.get('saturation_dn') for d in data['train_datasets']}
    assert levels == {'AIA': 13000.0, 'AIA-focus': 13000.0, 'EUVI-A': 13000.0,
                      'EUVI-A-focus': 13000.0, 'EUI': None, 'EUI-focus': None}
    assert all(d.get('saturation_dn') in (13000.0, None) for d in data['valid_datasets'])
    # The validated config itself is unchanged.
    assert 'saturation_dn' not in config['data']['train_datasets'][0]


@pytest.mark.parametrize('value', [0, -1, 'high', True])
def test_invalid_saturation_levels_are_rejected(value):
    config = _config()
    config['instruments'][0]['saturation_dn'] = value
    with pytest.raises(PlasmaConfigError, match='saturation_dn'):
        validate_plasma_config(config)


def test_datasets_cannot_set_a_saturation_level():
    config = _config()
    config['data']['train_datasets'][0]['saturation_dn'] = 13000
    with pytest.raises(PlasmaConfigError, match='belongs to instruments'):
        validate_plasma_config(copy.deepcopy(config))
