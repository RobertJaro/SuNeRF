from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from sunerf.data.euv.estimate_scaling import (
    estimate_config_scaling,
    load_scaling_table,
    resolve_instrument_scaling,
    write_scaling_table,
)
from sunerf.data.euv.observation import normalize_image_scaling
from sunerf.train.scaling import ImageAsinhScaling
from test_euv_data_contract import _aia_map


def _instrument(divisor, channels=('A171', 'A193')):
    return {
        'key': 'AIA',
        'type': 'plasma',
        'scaling': {'type': 'asinh', 'divisor': divisor},
        'temperature_response': {'artifact': 'unused', 'channels': list(channels)},
    }


def test_table_reference_resolves_to_explicit_response_ordered_divisors(tmp_path):
    path = tmp_path / 'tables' / 'image_scaling.yaml'
    _, entry = normalize_image_scaling([10.0, 40.0], ('171', '193'))
    write_scaling_table(path, {'AIA': entry})
    instruments = [_instrument(str(path))]

    resolved = resolve_instrument_scaling(instruments)

    assert resolved[0]['scaling']['divisor'] == [10.0, 40.0]
    assert resolved[0]['scaling']['divisor_source'] == str(path)
    # The input configuration is left untouched and explicit values pass through.
    assert instruments[0]['scaling']['divisor'] == str(path)
    explicit = [_instrument({'A171': 2.0, 'A193': 3.0})]
    assert resolve_instrument_scaling(explicit) == explicit


def test_table_reference_rejects_missing_tables_instruments_and_channels(tmp_path):
    path = tmp_path / 'image_scaling.yaml'
    with pytest.raises(FileNotFoundError, match='estimate_scaling'):
        resolve_instrument_scaling([_instrument(str(path))])

    _, entry = normalize_image_scaling([10.0, 40.0], ('171', '193'))
    write_scaling_table(path, {'EUVI-A': entry})
    with pytest.raises(ValueError, match="no entry for instrument 'AIA'"):
        resolve_instrument_scaling([_instrument(str(path))])

    write_scaling_table(path, {'AIA': entry})
    with pytest.raises(ValueError, match='does not match response channels'):
        resolve_instrument_scaling([_instrument(str(path), channels=('A193', 'A171'))])


def test_asinh_scaling_applies_per_channel_divisor():
    scaling = ImageAsinhScaling(divisor=[10.0, 40.0])
    reference = ImageAsinhScaling()

    torch.testing.assert_close(
        scaling(torch.tensor([[10.0, 40.0], [-20.0, 80.0]])),
        reference(torch.tensor([[1.0, 1.0], [-2.0, 2.0]])),
    )
    with pytest.raises(ValueError, match='finite and positive'):
        ImageAsinhScaling(divisor=[1.0, 0.0])


def test_config_tables_pool_training_datasets_and_are_reused(tmp_path):
    start = datetime(2020, 1, 1)
    levels = {'background': (10.0, 20.0, 30.0), 'event': (1000.0,)}
    datasets = []
    for key, values in levels.items():
        directory = tmp_path / key
        directory.mkdir()
        for index, value in enumerate(values):
            obstime = start + timedelta(hours=index)
            for channel, factor in ((171, 1.0), (193, 4.0)):
                data = np.full((4, 5), value * factor, dtype=np.float32)
                _aia_map(
                    channel, obstime, data, prepared_schema='sunerf.prepared_euv.v2',
                    sensitivity_convention='reference_epoch',
                ).save(directory / f'{channel}_{index}.prepared.fits')
        datasets.append({
            'type': 'AIA',
            'key': key,
            'instrument_key': 'AIA',
            'data_path': str(directory / '*.prepared.fits'),
            'wavelengths': [171, 193],
        })
    table_path = tmp_path / 'image_scaling.yaml'
    config = {
        'instruments': [_instrument(str(table_path))],
        'data': {'train_datasets': datasets, 'holdout': None},
    }

    estimate_config_scaling(config, datasets={'background'}, percentile=100, workers=1)

    entry = load_scaling_table(table_path)['instruments']['AIA']
    assert entry['divisor'] == [20.0, 80.0]
    assert entry['source']['datasets'] == ['background']
    assert entry['source']['observation_count'] == 3

    # An existing table is reused so resumed runs keep their constants ...
    assert estimate_config_scaling(config, percentile=100, workers=1) == []
    assert load_scaling_table(table_path)['instruments']['AIA']['divisor'] == [20.0, 80.0]
    # ... until it is re-estimated explicitly, here pooling both sequences.
    estimate_config_scaling(config, percentile=100, workers=1, overwrite=True)
    entry = load_scaling_table(table_path)['instruments']['AIA']
    assert entry['divisor'] == [25.0, 100.0]
    assert entry['source']['datasets'] == ['background', 'event']
