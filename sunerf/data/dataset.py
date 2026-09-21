import hashlib
import os
import re
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset


class TensorsDataset(Dataset):
    """Cache aligned arrays on disk and lazily load batches as float32 tensors."""

    _WRITE_CHUNK_SIZE = 2 ** 20

    def __init__(self, tensors, work_directory, filter_nans=True, shuffle=True, ds_name=None,
                 valid_mask=None, shuffle_seed=None, batch_size=2 ** 13, **kwargs):
        self.batch_size = int(batch_size)
        self.addition_kwargs = kwargs
        self._mmap_arrays = None
        self._mmap_owner_pid = None
        os.makedirs(work_directory, exist_ok=True)
        if not tensors:
            raise ValueError('tensors must contain at least one array')
        n_samples = int(next(iter(tensors.values())).shape[0])
        if any(int(tensor.shape[0]) != n_samples for tensor in tensors.values()):
            raise ValueError('All cached tensors must have the same leading dimension.')

        # Filter in chunks to bound memory use. An explicit mask overrides
        # the default requirement that every tensor in a row is finite.
        keep_mask = None
        if filter_nans:
            if valid_mask is not None:
                keep_mask = np.asarray(valid_mask, dtype=bool)
                if keep_mask.shape != (n_samples,):
                    raise ValueError(
                        f'valid_mask must have shape ({n_samples},), got {keep_mask.shape}.'
                    )
            else:
                keep_mask = np.ones(n_samples, dtype=bool)
                for start in range(0, n_samples, self._WRITE_CHUNK_SIZE):
                    stop = min(start + self._WRITE_CHUNK_SIZE, n_samples)
                    valid_in_every_tensor = np.ones(stop - start, dtype=bool)
                    for tensor in tensors.values():
                        chunk = np.asarray(tensor[start:stop])
                        axes = tuple(range(1, chunk.ndim))
                        nonfinite = ~np.isfinite(chunk)
                        invalid = nonfinite if not axes else np.any(nonfinite, axis=axes)
                        valid_in_every_tensor &= ~invalid
                    keep_mask[start:stop] = valid_in_every_tensor
            filtered_count = int(n_samples - np.count_nonzero(keep_mask))
            if filtered_count:
                print(f'Filtering {filtered_count} non-finite entries')
        output_size = n_samples if keep_mask is None else int(np.count_nonzero(keep_mask))
        if output_size == 0:
            raise ValueError('No samples remain after filtering non-finite entries.')

        # One global row permutation mixes all frames before forming batches.
        # Reuse it for every tensor so images, coordinates and times stay paired.
        indices = np.arange(n_samples) if keep_mask is None else np.flatnonzero(keep_mask)
        del keep_mask
        if shuffle:
            np.random.default_rng(shuffle_seed).shuffle(indices)

        safe_name = 'dataset' if ds_name is None else re.sub(r'[^A-Za-z0-9_.-]+', '_', str(ds_name))[:64]
        cache_id = f'{safe_name}-{uuid.uuid4().hex}'
        batches_paths = {}
        created_paths = []
        outputs = {}
        try:
            for key, tensor in tensors.items():
                key_text = str(key)
                safe_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', key_text)[:64]
                key_digest = hashlib.sha1(key_text.encode()).hexdigest()[:8]
                safe_key = f'{safe_key}-{key_digest}'
                cache_path = os.path.join(work_directory, f'{cache_id}_{safe_key}.npy')
                output_shape = (output_size, *tensor.shape[1:])
                outputs[key] = np.lib.format.open_memmap(
                    cache_path, mode='w+', dtype=np.float32, shape=output_shape
                )
                created_paths.append(cache_path)
                batches_paths[key] = cache_path

            # Bound temporary tensor copies while writing the globally shuffled
            # rows sequentially. Training reads contiguous slices of this cache.
            for start in range(0, output_size, self._WRITE_CHUNK_SIZE):
                stop = min(start + self._WRITE_CHUNK_SIZE, output_size)
                for key, tensor in tensors.items():
                    outputs[key][start:stop] = np.asarray(tensor)[indices[start:stop]]

            for output in outputs.values():
                output.flush()
            outputs.clear()
        except Exception:
            outputs.clear()
            for cache_path in created_paths:
                try:
                    os.remove(cache_path)
                except FileNotFoundError:
                    pass
            raise
        self.batches_file_paths = batches_paths
        self._n_samples = output_size

    @staticmethod
    def _close_array(array):
        mmap = getattr(array, '_mmap', None)
        if mmap is not None:
            mmap.close()

    def _open_memmaps(self):
        pid = os.getpid()
        if self._mmap_arrays is None or self._mmap_owner_pid != pid:
            self.close()
            self._mmap_arrays = {
                key: np.load(path, mmap_mode='r')
                for key, path in self.batches_file_paths.items()
            }
            self._mmap_owner_pid = pid
        return self._mmap_arrays

    def __len__(self):
        if self._n_samples is None:
            ref_file = next(iter(self.batches_file_paths.values()))
            ref_array = np.load(ref_file, mmap_mode='r')
            try:
                self._n_samples = int(ref_array.shape[0])
            finally:
                self._close_array(ref_array)
        return (self._n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        start = idx * self.batch_size
        stop = min(start + self.batch_size, self._n_samples)
        arrays = self._open_memmaps()
        # A writable, contiguous copy is required before the tensor leaves a
        # worker process. torch.from_numpy then shares that one copy instead of
        # asking torch.tensor to allocate and convert a second buffer.
        data = {
            key: torch.from_numpy(np.array(array[start:stop], dtype=np.float32, copy=True, order='C'))
            for key, array in arrays.items()
        }
        data.update(self.addition_kwargs)
        return data

    def close(self):
        if self._mmap_arrays is not None:
            for array in self._mmap_arrays.values():
                self._close_array(array)
        self._mmap_arrays = None
        self._mmap_owner_pid = None

    def clear(self):
        self.close()
        for file_path in set(self.batches_file_paths.values()):
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        # mmap handles are process-local and should never enter data_module.pkl.
        state['_mmap_arrays'] = None
        state['_mmap_owner_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap_arrays = None
        self._mmap_owner_pid = None
        if not hasattr(self, '_n_samples'):
            self._n_samples = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ArrayDataset(Dataset):

    def __init__(self, array_dict, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.array_dict = array_dict
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_array = list(self.array_dict.values())[0]
        n_batches = np.ceil(ref_array.shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        data = {k: np.copy(v[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, v in self.array_dict.items()}
        return data

class IndexedDataset(Dataset):

    def __init__(self, dataset, key='dataset_idx'):
        """Data set wrapper to add an index to each data sample.

        :param dataset: base dataset.
        :param key: key to use for the index.
        """
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data[self.key] = torch.tensor([idx], dtype=torch.long)
        return data
