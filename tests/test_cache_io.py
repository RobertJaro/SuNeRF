import os
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from sunerf.data.dataset import TensorsDataset
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


def test_loader_workers_are_bounded_and_generated_data_stays_in_process(tmp_path):
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

    mmap_kwargs = module._loader_kwargs(mmap_dataset, training=True)
    generated_kwargs = module._loader_kwargs(_GeneratedDataset(), training=True)
    assert mmap_kwargs['num_workers'] == len(mmap_dataset) == 3
    assert mmap_kwargs['prefetch_factor'] == 2
    assert generated_kwargs['num_workers'] == 0
    assert 'prefetch_factor' not in generated_kwargs
    mmap_dataset.clear()


def test_training_workers_are_a_per_rank_budget_not_a_per_loader_multiplier(tmp_path):
    datasets = {}
    for index in range(5):
        datasets[f'mmap_{index}'] = TensorsDataset(
            {'value': np.arange(12, dtype=np.float32)[:, None]},
            tmp_path,
            batch_size=2,
            filter_nans=False,
        )
    datasets['generated'] = _GeneratedDataset()
    module = BaseDataModule(
        datasets,
        {},
        Rs_per_ds=1,
        seconds_per_dt=1,
        ref_date=None,
        module_config={},
        num_workers=4,
    )

    allocations = module._training_worker_allocations()

    assert sum(allocations.values()) == 4
    assert allocations['generated'] == 0
    assert sorted(allocations[name] for name in datasets if name != 'generated') == [0, 1, 1, 1, 1]
    module.clear()


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
    assert len(records[0]['sha256']) == 64

    source.write_bytes(b'a different size')
    changed, _ = build_data_cache_fingerprint(config)
    assert changed != first

    changed_config, _ = build_data_cache_fingerprint({
        **config,
        'train_datasets': [{'data_path': str(source), 'scaling': 2}],
    })
    assert changed_config != changed


def test_cache_fingerprint_tracks_prepared_fits_by_content(tmp_path):
    prepared = tmp_path / 'prepared.fits'
    prepared.write_bytes(b'first-content')
    config = {'train_datasets': [{'data_path': str(prepared)}]}

    first, records = build_data_cache_fingerprint(config)
    original_stat = prepared.stat()
    prepared.write_bytes(b'other-content')
    os.utime(prepared, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second, _ = build_data_cache_fingerprint(config)

    assert {record['path'] for record in records} == {str(prepared)}
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
