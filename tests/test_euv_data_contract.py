from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header

from sunerf.data.euv.observation import (
    estimate_image_scaling,
    match_prepared_channels,
    normalize_image_scaling,
    prepare_aia_observation,
)
from sunerf.data.loader.multi_instrument import GenericEUVDataset
from sunerf.data.loader.multi_instrument import (
    MultiInstrumentDataModule,
    _load_euv_observation,
    _observation_split_indices,
)


def _aia_map(
    channel, obstime, data=None, *, crpix_shift=0.0,
    prepared_schema=None, observer_lon=0.0, sensitivity_convention=None,
):
    shape = (4, 5)
    observer = SkyCoord(
        observer_lon * u.deg,
        0 * u.deg,
        1 * u.AU,
        frame=frames.HeliographicStonyhurst,
        obstime=obstime,
    )
    reference = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=frames.Helioprojective,
        observer=observer,
        obstime=obstime,
    )
    header = make_fitswcs_header(
        shape, reference, scale=np.array([1.0, 1.0]) * u.arcsec / u.pix
    )
    header['wavelnth'] = channel
    header['waveunit'] = 'angstrom'
    header['instrume'] = 'AIA'
    header['quality'] = 0
    header['exptime'] = 1.0
    header['bunit'] = 'DN / s'
    if sensitivity_convention is not None:
        header['senscon'] = sensitivity_convention
    if prepared_schema is not None:
        header['prepschm'] = prepared_schema
        header['radsem'] = 'per_native_pixel'
        header['natpxsr'] = float((1 * u.arcsec).to_value(u.rad) ** 2)
    header['crpix1'] += crpix_shift
    if data is None:
        data = np.ones(shape, dtype=np.float32) * channel
    return Map(np.asarray(data, dtype=np.float32), header)


def test_unique_time_matching_does_not_reuse_nearest_candidate():
    start = datetime(2020, 1, 1)
    file_dict = {
        171: ['r0', 'r1'],
        193: ['c0', 'c1'],
    }
    date_dict = {
        171: [start, start + timedelta(seconds=30)],
        193: [start + timedelta(seconds=18), start + timedelta(seconds=54)],
    }
    original = {key: list(value) for key, value in file_dict.items()}

    files, dates, reference = match_prepared_channels(
        file_dict, date_dict, tolerance=timedelta(minutes=1), channel_order=(171, 193)
    )

    assert reference == 171
    assert files[193].tolist() == ['c0', 'c1']
    assert len(set(files[193])) == 2
    assert file_dict == original
    assert dates[171].tolist() == date_dict[171]


def test_multi_channel_time_matching_is_globally_complete_not_pairwise_greedy():
    start = datetime(2020, 1, 1)

    def seconds(values):
        return [start + timedelta(seconds=value) for value in values]

    file_dict = {
        'reference': ['r0', 'r1', 'r2'],
        'a': ['a0', 'a1', 'a4'],
        'b': ['b0', 'b3', 'b4'],
    }
    date_dict = {
        'reference': seconds([0, 1, 2]),
        'a': seconds([0, 1, 4]),
        'b': seconds([0, 3, 4]),
    }

    files, _, reference = match_prepared_channels(
        file_dict,
        date_dict,
        tolerance=timedelta(seconds=1),
        channel_order=('reference', 'a', 'b'),
    )

    assert reference == 'reference'
    assert files['reference'].tolist() == ['r0', 'r2']
    assert files['a'].tolist() == ['a0', 'a1']
    assert files['b'].tolist() == ['b0', 'b3']


def test_image_scaling_is_ordered_explicit_and_reversible():
    divisors, metadata = normalize_image_scaling(
        {'A193': 20, 'A171': 10}, (171, 193)
    )
    np.testing.assert_array_equal(divisors, [10, 20])
    assert metadata == {
        'schema': 'sunerf.image_scaling.v1',
        'operation': 'divide',
        'channel_ids': ['171', '193'],
        'divisor': [10.0, 20.0],
        'inverse_operation': 'multiply',
    }


def test_image_scaling_rejects_duplicate_channel_aliases():
    with pytest.raises(ValueError, match='duplicate channel aliases'):
        normalize_image_scaling({171: 10, 'A171': 20, 193: 30}, (171, 193))


def test_robust_scaling_uses_temporal_median():
    images = np.array([
        [[[1.0, -2.0], [4.0, -8.0]]],
        [[[2.0, -4.0], [8.0, -16.0]]],
        [[[100.0, -200.0], [400.0, -800.0]]],
    ])
    valid = np.ones_like(images, dtype=bool)
    valid[1, 0, 1, 1] = False

    divisors, metadata = estimate_image_scaling(
        images, valid, (171, 193), percentile=100
    )

    # Per-observation maxima are [4, 8, 400] and [8, 4, 800]. Their temporal
    # medians are robust to the final outlier observation.
    np.testing.assert_array_equal(divisors, [8.0, 8.0])
    assert metadata['estimator'] == {
        'strategy': 'robust_percentile',
        'percentile': 100.0,
        'pixel_statistic': 'absolute_valid_magnitude',
        'temporal_reduction': 'median',
        'observation_count_by_channel': [3, 3],
    }
    reused, reused_metadata = normalize_image_scaling(metadata, (171, 193))
    np.testing.assert_array_equal(reused, divisors)
    assert reused_metadata == metadata


def test_prepared_contract_rejects_channel_order_and_wcs_mismatch():
    obstime = datetime(2020, 1, 1)
    map_171 = _aia_map(171, obstime)
    map_193 = _aia_map(193, obstime + timedelta(seconds=5))

    observation = prepare_aia_observation(
        [map_171, map_193], [171, 193], source_paths=['171.fits', '193.fits']
    )
    assert observation.image.shape == (2, 4, 5)
    assert observation.channel_ids == ('171', '193')
    assert observation.measurement_units == ('DN / s', 'DN / s')

    with pytest.raises(ValueError, match='Channel order mismatch'):
        prepare_aia_observation([map_171, map_193], [193, 171])

    shifted = _aia_map(193, obstime, crpix_shift=1.0)
    with pytest.raises(ValueError, match='not co-registered'):
        prepare_aia_observation([map_171, shifted], [171, 193])

    other_viewpoint = _aia_map(193, obstime, observer_lon=20.0)
    with pytest.raises(ValueError, match='incompatible observer viewpoints'):
        prepare_aia_observation([map_171, other_viewpoint], [171, 193])


def test_multi_channel_mean_time_drives_model_supervision(tmp_path):
    start = datetime(2020, 1, 1)
    paths = []
    for channel, offset in ((171, 0), (193, 20)):
        path = tmp_path / f'{channel}.fits'
        _aia_map(channel, start + timedelta(seconds=offset)).save(path)
        paths.append(str(path))
    record = _load_euv_observation(
        (paths, (171, 193), 'AIA', 'AIA', 1, None, 0.25, False)
    )

    assert record['time'] == start + timedelta(seconds=10)
    assert record['metadata']['time_offsets_seconds'] == [-10.0, 10.0]


def test_strict_prepared_contract_requires_calibration_metadata(tmp_path):
    obstime = datetime(2020, 1, 1)
    missing = _aia_map(171, obstime)
    with pytest.raises(ValueError, match='must declare SENSCON'):
        prepare_aia_observation(
            [missing], [171], strict_metadata=True, source_paths=['171.fits']
        )

    bound = _aia_map(
        171,
        obstime,
        prepared_schema='sunerf.prepared_euv.v2',
        sensitivity_convention='reference_epoch',
    )
    bound_path = tmp_path / '171.fits'
    bound.save(bound_path)
    with fits.open(bound_path, mode='update') as hdul:
        hdul.append(
            fits.ImageHDU(np.ones(bound.data.shape, dtype=np.uint8), name='VALID_MASK')
        )
        hdul[0].header['MASKEXT'] = 'VALID_MASK'
    observation = prepare_aia_observation(
        [bound_path], [171], strict_metadata=True, source_paths=[str(bound_path)]
    )
    assert observation.sensitivity_conventions == ('reference_epoch',)

    static = _aia_map(
        171,
        obstime,
        prepared_schema='sunerf.prepared_euv.v2',
        sensitivity_convention='static_assumed',
    )
    static_path = tmp_path / 'static.fits'
    static.save(static_path)
    with fits.open(static_path, mode='update') as hdul:
        hdul.append(
            fits.ImageHDU(np.ones(static.data.shape, dtype=np.uint8), name='VALID_MASK')
        )
        hdul[0].header['MASKEXT'] = 'VALID_MASK'
    static_observation = prepare_aia_observation(
        [static_path], [171], strict_metadata=True, source_paths=[str(static_path)]
    )
    assert static_observation.sensitivity_conventions == ('static_assumed',)

    legacy = _aia_map(
        171,
        obstime,
        sensitivity_convention='reference_epoch',
    )
    with pytest.raises(ValueError, match='unsupported PREPSCHM'):
        prepare_aia_observation(
            [legacy], [171], strict_metadata=True, source_paths=['legacy.fits']
        )


def test_strict_contract_combines_finite_primary_data_with_valid_mask(tmp_path):
    obstime = datetime(2020, 1, 1)
    s_map = _aia_map(
        171,
        obstime,
        data=np.ones((4, 5), dtype=np.float32),
        prepared_schema='sunerf.prepared_euv.v2',
        sensitivity_convention='reference_epoch',
    )
    path = tmp_path / 'masked.fits'
    s_map.save(path)
    mask = np.ones(s_map.data.shape, dtype=np.uint8)
    mask[1, 2] = 0
    with fits.open(path, mode='update') as hdul:
        hdul.append(fits.ImageHDU(mask, name='VALID_MASK'))
        hdul[0].header['MASKEXT'] = 'VALID_MASK'

    observation = prepare_aia_observation(
        [path], [171], strict_metadata=True, source_paths=[str(path)]
    )
    assert observation.image[0, 1, 2] == 1
    assert not observation.valid_mask[0, 1, 2]


def test_generic_euv_dataset_persists_masks_and_channel_order_in_physical_units(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.log_overview',
        lambda *args, **kwargs: None,
    )
    obstime = datetime(2020, 1, 1)
    first_data = np.full((4, 5), 10.0, dtype=np.float32)
    first_data[0, 0] = -10.0
    second_data = np.full((4, 5), 40.0, dtype=np.float32)
    second_data[1, 2] = np.nan
    paths = []
    maps = [
        _aia_map(171, obstime, first_data),
        _aia_map(193, obstime + timedelta(seconds=5), second_data),
    ]
    for channel, s_map in zip((171, 193), maps):
        path = tmp_path / f'{channel}.fits'
        s_map.save(path)
        paths.append(str(path))

    dataset = GenericEUVDataset(
        file_dict={171: [paths[0]], 193: [paths[1]]},
        date_dict={171: [obstime], 193: [obstime + timedelta(seconds=5)]},
        work_directory=tmp_path / 'cache',
        ds_key='AIA-test',
        instrument_key='AIA',
        instrument_type='AIA',
        test=True,
        load_workers=0,
        batch_size=100,
    )
    batch = dataset[0]

    assert dataset.channel_ids == ('171', '193')
    assert dataset.data_config['channel_ids'] == ('171', '193')
    # The loss divisor belongs to the instrument; datasets keep prepared units.
    assert 'image_scaling' not in dataset.data_config
    assert 'image_scaling' not in batch
    assert batch['channel_ids'] == ('171', '193')
    assert batch['image'][0, 0].item() == -10.0
    np.testing.assert_allclose(batch['image'][1:, 0].numpy(), 10.0)
    valid_second = batch['valid_mask'][:, 1].numpy().astype(bool)
    np.testing.assert_allclose(batch['image'][:, 1].numpy()[valid_second], 40.0)
    invalid_row = 1 * 5 + 2
    assert batch['valid_mask'][invalid_row, 1].item() == 0
    assert batch['image'][invalid_row, 1].item() == 0
    assert torch_is_finite(batch['image'])
    assert torch_is_finite(batch['rays'])
    dataset.clear()


def test_explicit_center_holdout_has_no_train_validation_overlap():
    training, held_out = _observation_split_indices(
        7, test=False, holdout={'strategy': 'center', 'count': 3}
    )
    validation, validation_held_out = _observation_split_indices(
        7, test=True, holdout={'strategy': 'center', 'count': 3}
    )

    assert training.tolist() == [0, 1, 5, 6]
    assert validation.tolist() == [2, 3, 4]
    assert held_out.tolist() == validation_held_out.tolist() == validation.tolist()
    assert set(training).isdisjoint(validation)


def test_debug_selection_is_safe_for_fewer_than_twenty_observations():
    selected, _ = _observation_split_indices(5, test=False, debug=True)
    assert selected.tolist() == [0, 1, 2, 3, 4]


def test_holdout_sources_and_times_are_persisted_without_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.log_overview', lambda *args, **kwargs: None
    )
    start = datetime(2020, 1, 1)
    file_dict = {171: [], 193: []}
    date_dict = {171: [], 193: []}
    for observation_index in range(3):
        for channel, offset in ((171, 0), (193, 5)):
            obstime = start + timedelta(minutes=observation_index, seconds=offset)
            path = tmp_path / f'{channel}_{observation_index}.fits'
            _aia_map(channel, obstime).save(path)
            file_dict[channel].append(str(path))
            date_dict[channel].append(obstime)

    common = {
        'file_dict': file_dict,
        'date_dict': date_dict,
        'ds_key': 'AIA-split',
        'instrument_key': 'AIA',
        'instrument_type': 'AIA',
        'holdout': {'strategy': 'center', 'count': 1},
        'load_workers': 0,
        'batch_size': 100,
    }
    training = GenericEUVDataset(
        work_directory=tmp_path / 'train-cache', test=False, **common
    )
    validation = GenericEUVDataset(
        work_directory=tmp_path / 'valid-cache', test=True, **common
    )

    training_sources = {
        tuple(metadata['source_paths'])
        for metadata in training.data_config['prepared_observations']
    }
    validation_sources = {
        tuple(metadata['source_paths'])
        for metadata in validation.data_config['prepared_observations']
    }
    assert training_sources.isdisjoint(validation_sources)
    assert training.data_config['split']['selected_matched_indices'] == [0, 2]
    assert validation.data_config['split']['selected_matched_indices'] == [1]
    held_out = training.data_config['split']['held_out_observations'][0]
    assert held_out['source_paths'] == file_dict[171][1:2] + file_dict[193][1:2]
    assert held_out['channel_times'][0] == date_dict[171][1].isoformat()
    training.clear()
    validation.clear()


def test_multi_instrument_batch_size_is_per_rank_not_multiplied_by_gpu_count(monkeypatch):
    captured = []

    class _Dataset:
        def __init__(self, **kwargs):
            captured.append(kwargs['batch_size'])
            self.ref_date = datetime(2020, 1, 1)

    monkeypatch.setattr('sunerf.data.loader.multi_instrument.AIADataset', _Dataset)
    monkeypatch.setattr('sunerf.data.loader.multi_instrument.torch.cuda.device_count', lambda: 8)
    module = MultiInstrumentDataModule.__new__(MultiInstrumentDataModule)
    config = [{'type': 'AIA', 'key': 'AIA', 'instrument_key': 'AIA', 'data_path': 'unused'}]
    base = {
        'batch_size': 128,
        'validation_batch_size': 256,
        'ref_date': datetime(2020, 1, 1),
    }

    module._load_dataset(config, base, test_ds=False)
    module._load_dataset(config, base, test_ds=True)

    assert captured == [128, 256]


def test_data_module_persists_metadata_for_distinct_validation_keys(
    tmp_path, monkeypatch
):
    loaded_configs = []

    def fake_dataset(key):
        return SimpleNamespace(
            ref_date=datetime(2020, 1, 1),
            times=[datetime(2020, 1, 1)],
            normalized_times=np.array([0.0], dtype=np.float32),
            data_config={
                'instrument_key': 'AIA',
                'wcs': None,
                'image_shape': (2, 3),
                'cmaps': ('sdoaia171',),
                'channel_ids': ('171',),
                'measurement_units': ('DN / s',),
                'sensitivity_conventions': ('reference_epoch',),
                'measurement_semantics': ('per_native_pixel',),
                'native_pixel_solid_angle_sr': (1.0,),
                'native_pixel_solid_angle_sr_by_observation': [[1.0]],
                'prepared_schema': 'sunerf.prepared_euv.v2',
                'key': key,
            },
        )

    def fake_load_dataset(self, data_config, base_config, test_ds=False):
        del self
        loaded_configs.append(data_config)
        base_config['ref_date'] = datetime(2020, 1, 1)
        key = 'AIA_valid' if test_ds else 'AIA_train'
        return {key: fake_dataset(key)}

    monkeypatch.setattr(MultiInstrumentDataModule, '_load_dataset', fake_load_dataset)
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.RandomSphericalCoordinateDataset',
        lambda **kwargs: SimpleNamespace(config=kwargs),
    )
    module = MultiInstrumentDataModule(
        train_datasets=[{
            'type': 'AIA',
            'key': 'AIA_train',
            'instrument_key': 'AIA',
        }],
        valid_datasets=[{
            'type': 'AIA',
            'key': 'AIA_valid',
            'instrument_key': 'AIA',
        }],
        work_directory=tmp_path,
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        num_workers=0,
        random_config={},
    )

    assert set(module.config) == {'AIA_train', 'AIA_valid'}
    assert 'random' in module.training_datasets
    assert module.config['AIA_valid']['instrument_key'] == 'AIA'
    assert module.validation_dataset_mapping == {0: 'AIA_valid'}
    assert all('image_scaling' not in config for config in module.config.values())
    assert [config[0]['key'] for config in loaded_configs] == ['AIA_train', 'AIA_valid']


def test_data_module_rejects_dataset_level_scaling():
    module = MultiInstrumentDataModule.__new__(MultiInstrumentDataModule)
    config = [{
        'type': 'AIA', 'key': 'AIA', 'instrument_key': 'AIA',
        'data_path': 'unused', 'scaling': 1.0,
    }]

    with pytest.raises(ValueError, match='instruments\\[\\].scaling.divisor'):
        module._load_dataset(config, {'batch_size': 1, 'validation_batch_size': 1})


def torch_is_finite(value):
    # Keep torch out of module import until this integration test actually runs.
    import torch
    return bool(torch.isfinite(value).all())
