import os
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
    _cadence_indices,
    _select_observations,
    _time_window_indices,
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

    files, dates, times = match_prepared_channels(
        file_dict, date_dict, tolerance=timedelta(minutes=1), channel_order=(171, 193)
    )

    assert files[171].tolist() == ['r0', 'r1']
    assert files[193].tolist() == ['c0', 'c1']
    assert len(set(files[193])) == 2
    assert file_dict == original
    assert dates[171].tolist() == date_dict[171]
    assert times.tolist() == [start + timedelta(seconds=9), start + timedelta(seconds=42)]


def test_time_matching_keeps_time_steps_with_missing_channels():
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

    files, dates, times = match_prepared_channels(
        file_dict,
        date_dict,
        tolerance=timedelta(seconds=1),
        channel_order=('reference', 'a', 'b'),
    )

    # Every file is used once; a time step survives with the channels it has.
    assert files['reference'].tolist() == ['r0', 'r1', 'r2', None]
    assert files['a'].tolist() == ['a0', 'a1', None, 'a4']
    assert files['b'].tolist() == ['b0', None, 'b3', 'b4']
    assert dates['a'].tolist()[2] is None
    assert times.tolist() == seconds([0, 1, 2.5, 4])


def test_prepared_observation_fills_missing_channel_with_nan(tmp_path):
    obstime = datetime(2020, 1, 1)
    path = tmp_path / '171.fits'
    _aia_map(171, obstime).save(path)

    observation = prepare_aia_observation(
        [str(path), None], (171, 193), source_paths=[str(path), None]
    )

    assert observation.channel_present == (True, False)
    assert observation.valid_mask[0].all()
    assert not observation.valid_mask[1].any()
    assert np.isnan(observation.image[1]).all()
    assert observation.measurement_units == ('DN / s', None)
    assert observation.obstime == obstime
    assert observation.metadata()['channel_times'][1] is None
    with pytest.raises(ValueError, match='at least one present channel'):
        prepare_aia_observation([None, None], (171, 193))


def test_non_positive_pixels_are_invalid_per_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.log_overview', lambda *args, **kwargs: None
    )
    obstime = datetime(2020, 1, 1)
    data = {171: np.full((4, 5), 171.0, dtype=np.float32), 193: np.full((4, 5), 193.0, dtype=np.float32)}
    data[171][0, 0] = 0.0      # one channel non-positive: pixel kept, channel masked
    data[193][1, 1] = -3.0     # same
    data[171][2, 2] = -1.0     # both channels invalid: pixel dropped
    data[193][2, 2] = 0.0
    data[193][3, 3] = np.nan   # NaN behaves like non-positive
    file_dict, date_dict = {}, {}
    for channel, values in data.items():
        path = tmp_path / f'{channel}.fits'
        _aia_map(channel, obstime, values).save(path)
        file_dict[channel] = [str(path)]
        date_dict[channel] = [obstime]

    observation = prepare_aia_observation(
        [file_dict[171][0], file_dict[193][0]], (171, 193)
    )
    assert np.isnan(observation.image[0, 0, 0]) and not observation.valid_mask[0, 0, 0]
    assert np.isnan(observation.image[1, 1, 1]) and not observation.valid_mask[1, 1, 1]
    assert observation.valid_mask[1, 0, 0] and observation.valid_mask[0, 1, 1]
    assert observation.valid_mask.sum() == 2 * 20 - 5

    dataset = GenericEUVDataset(
        file_dict=file_dict, date_dict=date_dict, work_directory=tmp_path / 'cache',
        ds_key='AIA', instrument_key='AIA', instrument_type='AIA', test=False,
        load_workers=0, batch_size=100,
    )
    try:
        image = np.load(dataset.batches_file_paths['image'])
        valid = np.load(dataset.batches_file_paths['valid_mask']).astype(bool)
        assert len(image) == 19  # the pixel invalid in both channels is dropped
        assert np.isfinite(image).all()
        assert valid.any(axis=-1).all()
        assert np.count_nonzero(~valid[:, 0]) == 1
        assert np.count_nonzero(~valid[:, 1]) == 2
        assert (image[valid] > 0).all()
        assert (image[~valid] == 0).all()
    finally:
        dataset.clear()


def test_dataset_ignores_missing_channel_in_training_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.log_overview', lambda *args, **kwargs: None
    )
    start = datetime(2020, 1, 1)
    file_dict = {171: [], 193: []}
    date_dict = {171: [], 193: []}
    for observation_index in range(3):
        for channel in (171, 193):
            if channel == 193 and observation_index == 1:
                continue  # 193 is missing at the second time step
            obstime = start + timedelta(hours=observation_index)
            path = tmp_path / f'{channel}_{observation_index}.fits'
            _aia_map(channel, obstime).save(path)
            file_dict[channel].append(str(path))
            date_dict[channel].append(obstime)

    dataset = GenericEUVDataset(
        file_dict=file_dict, date_dict=date_dict, work_directory=tmp_path / 'cache',
        ds_key='AIA', instrument_key='AIA', instrument_type='AIA', test=False,
        load_workers=0, batch_size=3, seconds_per_dt=3600,
    )
    try:
        time = np.load(dataset.batches_file_paths['time'])[:, 0]
        image = np.load(dataset.batches_file_paths['image'])
        valid = np.load(dataset.batches_file_paths['valid_mask'])
        assert len(time) == 3 * 20
        assert np.isfinite(image).all()
        missing = time == 1.0
        assert not valid[missing, 1].any()
        assert valid[missing, 0].all()
        assert valid[~missing].all()
        np.testing.assert_allclose(image[~missing, 1], 193.0)
        assert dataset.data_config['measurement_units'] == ('DN / s', 'DN / s')
        assert dataset.data_config['native_pixel_solid_angle_sr_by_observation'][1][1] > 0
        observations = dataset.data_config['prepared_observations']
        assert [metadata['channel_present'] for metadata in observations] == [
            [True, True], [True, False], [True, True]
        ]
        assert observations[1]['source_paths'][1] is None
    finally:
        dataset.clear()


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
        (paths, (171, 193), 'AIA', 'AIA', 1, None, 0.25, False, None)
    )

    assert record['time'] == start + timedelta(seconds=10)
    assert record['metadata']['time_offsets_seconds'] == [-10.0, 10.0]


def test_dataset_field_of_view_is_applied_at_load_time(tmp_path):
    path = tmp_path / '171.fits'
    _aia_map(171, datetime(2020, 1, 1)).save(path)
    # 1.7 arcsec at 1 AU as a line-of-sight impact parameter in solar radii.
    max_radius_rsun = (1.7 * u.arcsec).to_value(u.rad) * (1 * u.AU).to_value(u.R_sun)
    full = _load_euv_observation(((str(path),), (171,), 'AIA', 'AIA', 1, None, 0.25, False, None))
    cropped = _load_euv_observation(
        ((str(path),), (171,), 'AIA', 'AIA', 1, max_radius_rsun, 0.25, False, None)
    )

    assert np.all(np.isfinite(full['rays']))
    # Pixel centers lie at x in {-2..2} and y in {-1.5..1.5} arcsec.
    x, y = np.meshgrid(np.arange(5) - 2.0, np.arange(4) - 1.5)
    inside = np.hypot(x, y) < 1.7
    np.testing.assert_array_equal(np.all(np.isfinite(cropped['rays']), axis=(-2, -1)), inside)
    # The image itself stays uncropped; the NaN rays drop the pixels downstream.
    np.testing.assert_array_equal(cropped['valid_mask'], full['valid_mask'])


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
    # Pixels masked by VALID_MASK are NaN in the observation image.
    assert np.isnan(observation.image[0, 1, 2])
    assert not observation.valid_mask[0, 1, 2]
    assert observation.valid_mask.sum() == 20 - 1


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
        time=obstime,
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
    # Non-positive values are invalid: zeroed in the cache and masked.
    assert batch['image'][0, 0].item() == 0.0
    assert batch['valid_mask'][0, 0].item() == 0
    assert batch['valid_mask'][0, 1].item() == 1
    np.testing.assert_allclose(batch['image'][1:, 0].numpy(), 10.0)
    valid_second = batch['valid_mask'][:, 1].numpy().astype(bool)
    np.testing.assert_allclose(batch['image'][:, 1].numpy()[valid_second], 40.0)
    invalid_row = 1 * 5 + 2
    assert batch['valid_mask'][invalid_row, 1].item() == 0
    assert batch['image'][invalid_row, 1].item() == 0
    assert torch_is_finite(batch['image'])
    assert torch_is_finite(batch['rays'])
    dataset.clear()


def test_debug_selection_is_safe_for_fewer_than_twenty_observations():
    dates = [datetime(2025, 10, 14) + timedelta(hours=hour) for hour in range(5)]
    selected = _select_observations(dates, [True] * 5, test=False, debug=True)
    assert selected.tolist() == [0, 1, 2, 3, 4]


def test_validation_selection_is_persisted(tmp_path, monkeypatch):
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
        'load_workers': 0,
        'batch_size': 100,
    }
    training = GenericEUVDataset(
        work_directory=tmp_path / 'train-cache', test=False, **common
    )
    validation = GenericEUVDataset(
        work_directory=tmp_path / 'valid-cache', test=True,
        time='2020-01-01T00:01:10', **common
    )

    assert training.data_config['split']['selected_matched_indices'] == [0, 1, 2]
    assert validation.data_config['split']['selected_matched_indices'] == [1]
    assert validation.data_config['split']['time'] == '2020-01-01T00:01:10'
    (observation,) = validation.data_config['prepared_observations']
    assert list(observation['source_paths']) == file_dict[171][1:2] + file_dict[193][1:2]
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
                'observers': [{'radius': 215.0, 'latitude': 0.0, 'longitude': 0.0,
                               'time': datetime(2020, 1, 1)}],
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


def test_prepared_eui_map_without_pc_keys_loads(tmp_path):
    # sunpy's EUIMap declares _rotation_matrix_from_crota as a property that
    # GenericMap.rotation_matrix calls; a CROTA-only header must still load.
    from astropy.io import fits

    from sunerf.data.euv.observation import load_prepared_map

    header = fits.Header({
        'OBSRVTRY': 'Solar Orbiter', 'INSTRUME': 'EUI', 'DETECTOR': 'FSI', 'WAVELNTH': 174,
        'DATE-OBS': '2025-10-14T00:00:00', 'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CDELT1': 6.0, 'CDELT2': 6.0,
        'CRPIX1': 2.5, 'CRPIX2': 2.5, 'CRVAL1': 0.0, 'CRVAL2': 0.0, 'CROTA2': 0.0,
        'DSUN_OBS': 8.9e10, 'HGLN_OBS': 0.0, 'HGLT_OBS': 16.8, 'RSUN_REF': 695700000.0,
    })
    path = tmp_path / 'eui.prepared.fits'
    fits.PrimaryHDU(np.ones((4, 4), dtype=np.float32), header).writeto(path)

    s_map = load_prepared_map(path)
    assert type(s_map).__name__ == 'EUIMap'
    np.testing.assert_allclose(s_map.rotation_matrix, np.eye(2))
    assert s_map.wcs is not None


def test_euv_overview_uses_a_log_stretch_per_channel(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.colors import LogNorm
    from sunerf.train import callback as callback_module

    captured = []

    def _capture(payload):
        for figure in payload.values():
            captured.append([
                axis.images[0].norm for axis in figure.axes if axis.images
            ])

    monkeypatch.setattr(callback_module.wandb, 'Image', lambda figure: figure)
    monkeypatch.setattr(callback_module.wandb, 'log', _capture)

    rng = np.random.default_rng(0)
    images = np.stack([
        10.0 ** rng.uniform(0.0, 3.0, size=(2, 8, 8)),  # three decades of counts
        np.full((2, 8, 8), np.nan),  # a channel without data is skipped
    ], axis=-1)
    images[0, 0, 0, 0] = 0.0  # non-positive pixels are masked, not plotted
    poses = np.tile(np.eye(4)[None], (2, 1, 1))
    poses[:, 0, 3] = 215.0
    callback_module.log_overview(
        images, poses, np.array([0.0, 1.0]), ['sdoaia171', 'sdoaia193'], 86400.0, 1.0,
        datetime(2020, 1, 1), ds_key='AIA', channel_labels=['A171', 'A193'],
    )

    assert len(captured) == 2
    for norms in captured:
        assert len(norms) == 1  # the empty channel has no image
        norm = norms[0]
        assert isinstance(norm, LogNorm)
        assert norm.vmin > 0 and 1.0 <= norm.vmin < 10.0 and 100.0 < norm.vmax <= 1000.0


def test_training_uses_every_observation_and_validation_the_closest_complete_one():
    dates = [datetime(2025, 10, 14, 6) + timedelta(hours=hour) for hour in range(6)]
    complete = [True] * 6
    time = "2025-10-14T09:20:00"
    assert _select_observations(dates, complete, test=False, time=time).tolist() == list(range(6))
    assert _select_observations(dates, complete, test=True, time=time).tolist() == [3]
    # Only observations with all channels are rendered.
    complete = [True, True, True, False, True, True]
    assert _select_observations(dates, complete, test=True, time=time).tolist() == [4]
    with pytest.raises(ValueError, match="all channels"):
        _select_observations(dates, [False] * 6, test=True, time=time)
    # Without a time: the complete observation nearest the middle.
    assert _select_observations(dates, complete, test=True).tolist() == [2]


def test_time_window_selects_half_open_interval_on_reference_times():
    dates = [datetime(2025, 10, 13, 8) + timedelta(hours=hour) for hour in range(4)]
    window = _time_window_indices(dates, "2025-10-13T09:00:00", "2025-10-13T11:00:00")
    assert window.tolist() == [1, 2]
    assert _time_window_indices(dates, start="2025-10-13T10:00:00Z").tolist() == [2, 3]
    assert _time_window_indices(dates, end=datetime(2025, 10, 13, 9)).tolist() == [0]
    with pytest.raises(ValueError, match="No observations"):
        _time_window_indices(dates, "2025-10-14T00:00:00", "2025-10-15T00:00:00")
    with pytest.raises(ValueError, match="earlier than"):
        _time_window_indices(dates, "2025-10-13T11:00:00", "2025-10-13T09:00:00")


def test_cadence_keeps_one_observation_per_interval():
    # Hourly background with a 12 min focus block between 02:00 and 03:00.
    start = datetime(2025, 10, 13)
    minutes = [0, 60, 120, 132, 144, 156, 168, 180, 240]
    dates = [start + timedelta(minutes=value) for value in minutes]
    complete = [True] * len(dates)
    kept = _cadence_indices(dates, complete, 60)
    assert [minutes[index] for index in kept] == [0, 60, 120, 180, 240]
    assert _cadence_indices(dates, complete, 12).tolist() == list(range(len(dates)))

    # An hourly sequence whose times jitter by seconds keeps every observation.
    jitter = [0, -4, 3, -2, 5, -6]
    dates = [start + timedelta(hours=hour, seconds=value) for hour, value in enumerate(jitter)]
    assert _cadence_indices(dates, [True] * 6, 60).tolist() == list(range(6))

    # Within one grid slot an observation with all channels is preferred.
    dates = [start, start + timedelta(minutes=12), start + timedelta(minutes=60)]
    assert _cadence_indices(dates, [False, True, True], 60).tolist() == [1, 2]


def test_training_cache_interleaves_observations_and_drops_invalid_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(
        'sunerf.data.loader.multi_instrument.log_overview', lambda *args, **kwargs: None
    )
    start = datetime(2020, 1, 1)
    file_dict = {171: [], 193: []}
    date_dict = {171: [], 193: []}
    for observation_index in range(3):
        for channel in (171, 193):
            obstime = start + timedelta(hours=observation_index)
            data = np.full((4, 5), float(channel), dtype=np.float32)
            data[observation_index, 0] = np.nan  # same pixel invalid in both channels
            if channel == 193:
                data[3, 4] = np.nan  # one channel invalid: the row is kept
            path = tmp_path / f'{channel}_{observation_index}.fits'
            _aia_map(channel, obstime, data).save(path)
            file_dict[channel].append(str(path))
            date_dict[channel].append(obstime)

    dataset = GenericEUVDataset(
        file_dict=file_dict, date_dict=date_dict, work_directory=tmp_path / 'cache',
        ds_key='AIA', instrument_key='AIA', instrument_type='AIA', test=False,
        load_workers=0, batch_size=3, seconds_per_dt=3600,
    )
    try:
        assert sorted(path.name for path in (tmp_path / 'cache').iterdir()) == sorted(
            os.path.basename(path) for path in dataset.batches_file_paths.values()
        )
        time = np.load(dataset.batches_file_paths['time'])[:, 0]
        image = np.load(dataset.batches_file_paths['image'])
        valid = np.load(dataset.batches_file_paths['valid_mask'])
        assert len(time) == 3 * 19
        assert all(sorted(group) == [0.0, 1.0, 2.0] for group in time.reshape(-1, 3))
        assert np.isfinite(image).all()
        assert valid.any(axis=-1).all()
        assert np.count_nonzero(valid[:, 1] == 0) == 3
        np.testing.assert_allclose(image[:, 0], 171.0)
        assert dataset.data_config['image_shape'] == (4, 5)
        assert len(dataset.data_config['prepared_observations']) == 3
        batch = dataset[0]
        assert sorted(batch['time'][:, 0].tolist()) == [0.0, 1.0, 2.0]
    finally:
        dataset.clear()
