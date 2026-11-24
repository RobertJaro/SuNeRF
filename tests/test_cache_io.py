import os
import pickle

import pytest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from sunerf.data.dataset import ObservationCacheWriter, TensorsDataset, shuffled_valid_rows
from sunerf.data.loader.base_loader import BaseDataModule
from sunerf.data.loader.thomson_instrument import _load_map_stack, _pool_size
from sunerf.run_thomson import (
    _data_module_cache_files,
    _data_cache_is_usable,
    _is_cache_generation,
    build_data_cache_fingerprint,
    cache_reload_token,
    trainer_device_config,
)
from sunerf.train.render_mode import RenderMode, RenderModeDataset


def test_tensor_cache_is_row_shuffled_unique_and_reuses_mmaps(tmp_path):
    tensors = {
        'coords': np.arange(36, dtype=np.float32).reshape(12, 3),
        'value': np.arange(12, dtype=np.float32)[:, None],
    }
    first = TensorsDataset(
        tensors, tmp_path, batch_size=4, ds_name='same/name', shuffle=True,
        shuffle_seed=7, filter_nans=False
    )
    second = TensorsDataset(
        tensors, tmp_path, batch_size=4, ds_name='same/name', shuffle=True,
        shuffle_seed=11, filter_nans=False
    )

    assert set(first.batches_file_paths.values()).isdisjoint(second.batches_file_paths.values())
    shuffled_coords = np.load(first.batches_file_paths['coords'])
    shuffled_values = np.load(first.batches_file_paths['value'])
    assert not np.array_equal(shuffled_values, tensors['value'])
    np.testing.assert_array_equal(shuffled_coords[:, 0] / 3, shuffled_values[:, 0])
    np.testing.assert_array_equal(np.sort(shuffled_values[:, 0]), tensors['value'][:, 0])

    first.close()
    with patch('sunerf.data.dataset.np.load', wraps=np.load) as mocked_load:
        batch_0 = first[0]
        batch_1 = first[1]
    assert mocked_load.call_count == len(tensors)
    assert batch_0['coords'].dtype == torch.float32
    np.testing.assert_array_equal(batch_1['coords'].numpy(), shuffled_coords[4:8])

    restored = pickle.loads(pickle.dumps(first))
    assert restored._mmap_arrays is None
    np.testing.assert_array_equal(restored[2]['value'].numpy(), shuffled_values[8:])

    first.clear()
    second.clear()
    assert not any(tmp_path.glob('*.npy'))


def test_tensor_cache_filters_and_shuffles_globally(tmp_path, monkeypatch):
    monkeypatch.setattr(TensorsDataset, '_WRITE_CHUNK_SIZE', 3)
    values = np.arange(10, dtype=np.float32)[:, None]
    companion = values.copy()
    values[[2, 7]] = np.nan
    companion[[2, 7]] = np.nan
    dataset = TensorsDataset(
        {'values': values, 'companion': companion},
        tmp_path,
        batch_size=20,
        filter_nans=True,
        shuffle=True,
        shuffle_seed=5,
    )

    expected = np.delete(np.arange(10, dtype=np.float32), [2, 7])
    actual = dataset[0]['values'][:, 0].numpy()
    assert not np.array_equal(actual, expected)
    np.testing.assert_array_equal(np.sort(actual), expected)
    np.random.default_rng(5).shuffle(expected)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        dataset[0]['values'].numpy(), dataset[0]['companion'].numpy()
    )
    dataset.clear()


def test_global_shuffle_is_independent_of_write_chunk_size(tmp_path, monkeypatch):
    values = np.arange(128, dtype=np.float32)[:, None]
    cached = []
    for chunk_size in (8, 128):
        monkeypatch.setattr(TensorsDataset, '_WRITE_CHUNK_SIZE', chunk_size)
        dataset = TensorsDataset(
            {'value': values, 'paired': values * 2}, tmp_path,
            batch_size=16, filter_nans=False, shuffle_seed=42,
        )
        cached.append(np.load(dataset.batches_file_paths['value']))
        np.testing.assert_array_equal(dataset[0]['paired'], dataset[0]['value'] * 2)
        # The first batch must draw from across the original source chunks.
        assert len(np.unique(cached[-1][:16, 0] // 8)) > 4
        dataset.clear()
    np.testing.assert_array_equal(*cached)


def test_training_shuffles_disk_chunks_without_changing_their_contents(tmp_path):
    dataset = TensorsDataset(
        {'value': np.arange(32, dtype=np.float32)[:, None]}, tmp_path,
        batch_size=4, filter_nans=False, shuffle_seed=42,
    )
    module = BaseDataModule(
        {'instrument': dataset}, {'instrument': dataset}, Rs_per_ds=1, seconds_per_dt=1,
        ref_date=None, module_config={}, num_workers=0,
    )
    loader = module.train_dataloader().flattened[0]
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.batch_size is None
    loader.sampler.generator = torch.Generator().manual_seed(42)
    cached = np.load(dataset.batches_file_paths['value'])
    expected_chunks = [cached[i:i + 4].tolist() for i in range(0, len(cached), 4)]
    epochs = [[batch['value'].tolist() for batch in loader] for _ in range(2)]
    for chunks in epochs:
        assert sorted(chunks) == sorted(expected_chunks)
    assert epochs[0] != epochs[1]
    np.testing.assert_array_equal(np.load(dataset.batches_file_paths['value']), cached)
    assert isinstance(module.val_dataloader()[0].sampler, SequentialSampler)

    rank_indices = []
    for rank in range(2):
        module.trainer = SimpleNamespace(world_size=2, global_rank=rank)
        sampler = module.train_dataloader().flattened[0].sampler
        assert isinstance(sampler, DistributedSampler)
        assert sampler.shuffle
        sampler.set_epoch(0)
        indices = list(sampler)
        sampler.set_epoch(1)
        assert list(sampler) != indices
        rank_indices.append(indices)
    assert not set(rank_indices[0]) & set(rank_indices[1])
    assert sorted(rank_indices[0] + rank_indices[1]) == list(range(len(dataset)))
    dataset.clear()


def test_tensor_cache_default_filter_rejects_nonfinite_value_in_any_tensor(tmp_path):
    values = np.array([[np.nan], [1.0]], dtype=np.float32)
    finite_companion = np.ones((2, 1), dtype=np.float32)
    dataset = TensorsDataset(
        {'values': values, 'finite_companion': finite_companion},
        tmp_path,
        batch_size=10,
        filter_nans=True,
        shuffle=False,
    )

    np.testing.assert_array_equal(dataset[0]['values'].numpy(), [[1.0]])
    dataset.clear()


def test_tensor_cache_accepts_precomputed_valid_mask(tmp_path):
    values = np.arange(6, dtype=np.float32)[:, None]
    dataset = TensorsDataset(
        {'values': values},
        tmp_path,
        batch_size=10,
        filter_nans=True,
        valid_mask=np.array([True, False, True, False, True, True]),
        shuffle=False,
    )
    np.testing.assert_array_equal(dataset[0]['values'][:, 0].numpy(), [0, 2, 4, 5])
    dataset.clear()


class _GeneratedDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {'value': torch.tensor([index])}


def test_every_loader_uses_num_workers(tmp_path):
    mmap_dataset = TensorsDataset(
        {'value': np.arange(9, dtype=np.float32)[:, None]},
        tmp_path,
        batch_size=4,
        filter_nans=False,
    )
    module = BaseDataModule(
        {'mmap': mmap_dataset, 'generated': _GeneratedDataset()},
        {},
        Rs_per_ds=1,
        seconds_per_dt=1,
        ref_date=None,
        module_config={},
        num_workers=4,
    )

    for dataset in (mmap_dataset, _GeneratedDataset()):
        for training in (True, False):
            kwargs = module._loader_kwargs(dataset, training=training)
            assert kwargs['num_workers'] == 4
            assert kwargs['prefetch_factor'] == 2
            assert kwargs['persistent_workers'] == training
    mmap_dataset.clear()


def test_clear_reaches_wrapped_mmap_dataset(tmp_path):
    validation_dataset = TensorsDataset(
        {'value': np.arange(4, dtype=np.float32)[:, None]},
        tmp_path,
        filter_nans=False,
    )
    training_dataset = TensorsDataset(
        {'value': np.arange(4, dtype=np.float32)[:, None]},
        tmp_path,
        filter_nans=False,
    )
    paths = [
        *validation_dataset.batches_file_paths.values(),
        *training_dataset.batches_file_paths.values(),
    ]
    wrapped = RenderModeDataset(validation_dataset, RenderMode.QUERY_POINTS)
    # Deliberately reuse the key: this is how train/validation instrument
    # datasets are configured in the CME pipeline.
    module = BaseDataModule(
        {'instrument': training_dataset}, {'instrument': wrapped},
        1, 1, None, {}, num_workers=0,
    )
    assert set(_data_module_cache_files(module)) == set(map(os.path.abspath, paths))
    module.clear()
    assert all(not os.path.exists(path) for path in paths)


class _MapLoader:
    @staticmethod
    def load(value):
        return {
            'image': np.full((2, 3), value, dtype=np.float32),
            'time': value,
            'observer': {'index': value},
        }


def test_map_stack_preallocates_and_pool_size_is_bounded():
    arrays, observers = _load_map_stack([1, 2, 3], _MapLoader(), 0, 'test')
    assert arrays['image'].shape == (3, 2, 3)
    np.testing.assert_array_equal(arrays['time'], [1, 2, 3])
    assert observers == [{'index': 1}, {'index': 2}, {'index': 3}]
    assert _pool_size(1000, 2) <= 2
    assert _pool_size(0, 2) == 0


def test_cache_fingerprint_tracks_sources_but_not_worker_count(tmp_path):
    source = tmp_path / 'source.dat'
    source.write_bytes(b'first')
    config = {'num_workers': 4, 'train_datasets': [{'data_path': str(source)}]}
    first, records = build_data_cache_fingerprint(config)
    with_more_workers, _ = build_data_cache_fingerprint({**config, 'num_workers': 16})
    assert first == with_more_workers
    assert records[0]['path'] == str(source)
    assert set(records[0]) == {'path', 'size', 'mtime_ns'}

    source.write_bytes(b'a different size')
    changed, _ = build_data_cache_fingerprint(config)
    assert changed != first

    changed_config, _ = build_data_cache_fingerprint({
        **config,
        'train_datasets': [{'data_path': str(source), 'scaling': 2}],
    })
    assert changed_config != changed


def test_cache_fingerprint_tracks_prepared_fits_by_stat_without_reading(tmp_path):
    prepared = tmp_path / 'prepared.fits'
    prepared.write_bytes(b'first-content')
    config = {'train_datasets': [{'data_path': str(prepared)}]}

    with patch('builtins.open', side_effect=AssertionError('fingerprint must not read files')):
        first, records = build_data_cache_fingerprint(config)
    assert {record['path'] for record in records} == {str(prepared)}

    # A rewritten file with a newer modification time changes the fingerprint.
    original_stat = prepared.stat()
    prepared.write_bytes(b'other-content')
    os.utime(prepared, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 10 ** 9))
    second, _ = build_data_cache_fingerprint(config)
    assert second != first


def test_generation_cleanup_guard_accepts_only_direct_managed_children(tmp_path):
    cache_root = tmp_path / '.sunerf_cache'
    valid = cache_root / 'generation-1234'
    nested = valid / 'generation-nested'
    unrelated = tmp_path / 'generation-1234'
    assert _is_cache_generation(valid, cache_root)
    assert not _is_cache_generation(nested, cache_root)
    assert not _is_cache_generation(unrelated, cache_root)


def test_cache_validation_rejects_missing_generated_files(tmp_path):
    cache_file = tmp_path / 'field.npy'
    np.save(cache_file, np.ones(1, dtype=np.float32))
    data_module = type('CachedModule', (), {})()
    data_module.cache_fingerprint = 'expected'
    data_module.cache_files = [str(cache_file)]
    data_module.cache_reload_token = 'this-run'

    assert _data_cache_is_usable(data_module, 'expected')
    assert _data_cache_is_usable(data_module, 'expected', reload_token='this-run')
    assert not _data_cache_is_usable(data_module, 'expected', reload_token='another-run')
    cache_file.unlink()
    assert not _data_cache_is_usable(data_module, 'expected')


def test_trainer_device_config_has_valid_cpu_fallback():
    assert trainer_device_config(0) == ('cpu', 1)
    assert trainer_device_config(2) == ('gpu', 2)


def test_reload_token_is_stable_in_inherited_environment(monkeypatch):
    monkeypatch.delenv('SUNERF_CACHE_RELOAD_TOKEN', raising=False)
    monkeypatch.delenv('TORCHELASTIC_RUN_ID', raising=False)
    first = cache_reload_token(True)
    second = cache_reload_token(True)

    assert first == second
    assert cache_reload_token(False) is None


def _observation(index, rows=6, invalid=()):
    value = np.full((rows, 1), float(index), dtype=np.float32)
    pixel = np.arange(rows, dtype=np.float32)[:, None]
    for row in invalid:
        value[row] = np.nan
    return {'value': value, 'pixel': pixel}


def _add(writer, observation, seed, filter_rows=True):
    writer.add(observation, shuffled_valid_rows(
        observation, filter_rows=filter_rows, rng=np.random.default_rng(seed),
    ))


def test_stratified_cache_interleaves_every_observation_in_each_batch(tmp_path):
    n_observations, rows = 4, 6
    writer = ObservationCacheWriter(tmp_path, n_observations, ds_name='obs', shuffle_seed=3)
    for index in range(n_observations):
        _add(writer, _observation(index, rows), seed=index)
    paths, n_rows = writer.finalize(
        constant_tensors={'time': np.arange(n_observations, dtype=np.float32)[:, None] * 10}
    )

    assert n_rows == n_observations * rows
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        os.path.basename(path) for path in paths.values()
    )
    value = np.load(paths['value'])[:, 0]
    pixel = np.load(paths['pixel'])[:, 0]
    time = np.load(paths['time'])[:, 0]
    # Every group of n_observations consecutive rows holds each observation once.
    groups = value.reshape(rows, n_observations)
    assert all(sorted(group) == list(range(n_observations)) for group in groups)
    np.testing.assert_array_equal(time, value * 10)
    # The observation order is shuffled once, then shared by every slot.
    assert not np.array_equal(groups[0], np.arange(n_observations))
    assert all(np.array_equal(group, groups[0]) for group in groups)
    # Each observation visits each of its pixels exactly once, in shuffled order.
    for index in range(n_observations):
        own = pixel[value == index]
        assert sorted(own) == list(range(rows))
    assert any(not np.array_equal(pixel[value == i], np.arange(rows)) for i in range(n_observations))


def test_stratified_cache_drops_padding_of_short_observations(tmp_path):
    writer = ObservationCacheWriter(tmp_path, 3, shuffle_seed=0, block_rows=4)
    _add(writer, _observation(0, rows=6), seed=0)
    _add(writer, _observation(1, rows=6, invalid=(1, 4)), seed=1)
    _add(writer, {'value': np.full((6, 1), np.nan, dtype=np.float32),
                  'pixel': np.arange(6, dtype=np.float32)[:, None]}, seed=2)
    paths, n_rows = writer.finalize(constant_tensors={'time': np.array([[5.0], [7.0], [9.0]])})

    value = np.load(paths['value'])[:, 0]
    pixel = np.load(paths['pixel'])[:, 0]
    time = np.load(paths['time'])[:, 0]
    # The all-invalid observation is dropped; the short one keeps only its
    # valid rows, so the tail of the cache holds the long observation alone.
    assert n_rows == 6 + 4
    assert np.isfinite(value).all()
    assert sorted(pixel[value == 1]) == [0, 2, 3, 5]
    assert sorted(pixel[value == 0]) == list(range(6))
    np.testing.assert_array_equal(time, np.where(value == 0, 5.0, 7.0))
    assert sorted(value[:8].reshape(4, 2).tolist()) == [[0, 1], [0, 1], [0, 1], [0, 1]] or \
        sorted(value[:8].reshape(4, 2).tolist()) == [[1, 0], [1, 0], [1, 0], [1, 0]]
    np.testing.assert_array_equal(value[8:], [0, 0])


def test_sequential_cache_keeps_observation_order_and_drops_invalid_rows(tmp_path):
    writer = ObservationCacheWriter(tmp_path, 2, shuffle=False)
    for index, invalid in ((0, (1,)), (1, ())):
        observation = _observation(index, rows=3, invalid=invalid)
        writer.add(observation, shuffled_valid_rows(observation))
    paths, n_rows = writer.finalize(constant_tensors={'time': np.array([[5.0], [7.0]])})
    assert n_rows == 5
    np.testing.assert_array_equal(np.load(paths['value'])[:, 0], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(np.load(paths['pixel'])[:, 0], [0, 2, 0, 1, 2])
    np.testing.assert_array_equal(np.load(paths['time'])[:, 0], [5, 5, 7, 7, 7])


def test_unfiltered_cache_keeps_every_row_in_order(tmp_path):
    writer = ObservationCacheWriter(tmp_path, 2, shuffle=False)
    writer.add(_observation(0, rows=3, invalid=(1,)))
    writer.add(_observation(1, rows=3))
    paths, n_rows = writer.finalize(constant_tensors={'time': np.array([[5.0], [7.0]])})
    assert n_rows == 6
    value = np.load(paths['value'])[:, 0]
    assert np.isnan(value[1]) and value[3] == 1
    np.testing.assert_array_equal(np.load(paths['time'])[:, 0], [5, 5, 5, 7, 7, 7])


def test_cache_writer_removes_every_file_on_failure(tmp_path):
    writer = ObservationCacheWriter(tmp_path, 2, shuffle_seed=0)
    _add(writer, _observation(0), seed=0)
    with pytest.raises(ValueError, match='announced'):
        writer.finalize()
    assert not list(tmp_path.iterdir())

    writer = ObservationCacheWriter(tmp_path, 2, shuffle_seed=0)
    _add(writer, _observation(0), seed=0)
    with pytest.raises(ValueError, match='same keys'):
        writer.add({'value': np.zeros((6, 2))})
    assert not list(tmp_path.iterdir())


def test_block_size_does_not_change_the_cache(tmp_path):
    rng = np.random.default_rng(1)
    observations = []
    for index in range(5):
        value = rng.normal(size=(40, 3)).astype(np.float32)
        value[rng.integers(0, 40, size=index * 3)] = np.nan
        observations.append({'value': value, 'index': np.full((40, 1), index, np.float32)})
    caches = []
    for block_rows in (7, 10 ** 6):
        directory = tmp_path / str(block_rows)
        writer = ObservationCacheWriter(directory, 5, shuffle_seed=9, block_rows=block_rows)
        for index, observation in enumerate(observations):
            _add(writer, observation, seed=index)
        paths, n_rows = writer.finalize(constant_tensors={'time': np.arange(5.0)[:, None]})
        assert n_rows == sum(int(np.isfinite(o['value']).all(axis=1).sum()) for o in observations)
        caches.append({key: np.load(path) for key, path in paths.items()})
    for key in caches[0]:
        np.testing.assert_array_equal(caches[0][key], caches[1][key])
    assert np.isfinite(caches[0]['value']).all()


def test_thomson_cache_provenance_includes_matching_review_masks(tmp_path):
    from sunerf.data.review_masks import mask_path, write_mask

    image = tmp_path / 'observation.fits'
    image.touch()
    config = {'data_path_tB': str(tmp_path / '*.fits')}
    previous, _ = build_data_cache_fingerprint(config)
    target = mask_path(image)
    write_mask(target, np.eye(4, dtype=bool))
    current, records = build_data_cache_fingerprint(config)
    assert current != previous
    assert {r['path'] for r in records} == {str(image), str(target)}
    target.unlink()
    assert build_data_cache_fingerprint(config)[0] == previous
