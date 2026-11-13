import hashlib
import os
import re
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset


def _safe_name(value, fallback):
    if value is None:
        return fallback
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))[:64]


def _remove_quietly(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def shuffled_valid_rows(tensors, valid=None, *, filter_rows=True, rng=None):
    """Return the row positions of one observation to cache, in cache order.

    ``valid`` marks the rows to keep; without it and with ``filter_rows`` the
    rows that are finite in every tensor are kept. ``rng`` shuffles the kept
    rows, so a loader worker can randomize its own observation before the
    rows are sent to the main process.
    """
    n_rows = int(np.shape(next(iter(tensors.values())))[0])
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (n_rows,):
            raise ValueError(f'valid_mask must have shape ({n_rows},), got {valid.shape}.')
    else:
        valid = np.ones(n_rows, dtype=bool)
        if filter_rows:
            for value in tensors.values():
                finite = np.isfinite(np.asarray(value))
                valid &= finite.reshape(n_rows, -1).all(axis=1)
    indices = np.flatnonzero(valid)
    if rng is not None:
        rng.shuffle(indices)
    return indices


class ObservationCacheWriter:
    """Assemble observation rows in memory and persist them once as float32 ``.npy``.

    An observation is a dict of aligned arrays whose leading axis enumerates
    its rows (pixels), added together with the positions of the rows to cache
    in cache order (see ``shuffled_valid_rows``). Observations are only
    referenced until ``finalize``; nothing is staged on disk.

    With ``shuffle`` the cache is the (observation, slot) table written
    transposed: ``finalize`` stacks every key as ``(n, width)``, with the
    observations in one random order and short observations filling their
    missing slots with a random subset of their own rows, and writes the table
    slot-major in blocks, so the final row ``slot * n + j`` holds slot ``slot``
    of the ``j``-th permuted observation. Every contiguous batch therefore
    draws the same number of random rows from every observation and spans the
    whole observation sequence. Without ``shuffle`` the selected rows are
    written in observation order.

    Keys whose value is constant per observation (such as the observation
    time) are passed to ``finalize`` as ``(n, ...)`` arrays and tiled on the
    fly, so they are never held per row.
    """

    def __init__(self, work_directory, n_observations, *, ds_name=None, shuffle=True,
                 shuffle_seed=None, block_rows=2 ** 22):
        self.work_directory = os.fspath(work_directory)
        os.makedirs(self.work_directory, exist_ok=True)
        self.n_observations = int(n_observations)
        if self.n_observations <= 0:
            raise ValueError('n_observations must be positive.')
        self.shuffle = bool(shuffle)
        self.block_rows = max(int(block_rows), 1)
        self.rng = np.random.default_rng(shuffle_seed)
        self.cache_id = f'{_safe_name(ds_name, "dataset")}-{uuid.uuid4().hex}'
        self._shapes = None  # key -> trailing shape, fixed by the first observation
        self._sources = []  # (tensors, indices) per observation
        self._outputs = {}
        self._final_paths = {}
        self._created = []
        self._filtered = 0
        self._finished = False

    # ------------------------------------------------------------------ files
    def _open(self, key, shape):
        key_text = str(key)
        safe_key = f'{_safe_name(key_text, "tensor")}-{hashlib.sha1(key_text.encode()).hexdigest()[:8]}'
        path = os.path.join(self.work_directory, f'{self.cache_id}_{safe_key}.npy')
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
        self._created.append(path)
        self._final_paths[key] = path
        self._outputs[key] = array

    # ------------------------------------------------------------------- add
    def add(self, tensors, indices=None):
        """Add one observation; ``indices`` are its rows to cache, in cache order."""
        if self._finished:
            raise RuntimeError('The cache writer has already been finalized.')
        try:
            if len(self._sources) >= self.n_observations:
                raise ValueError(f'More than {self.n_observations} observations were added.')
            if not tensors:
                raise ValueError('tensors must contain at least one array')
            rows = {key: int(np.shape(value)[0]) for key, value in tensors.items()}
            n_rows = next(iter(rows.values()))
            if any(value != n_rows for value in rows.values()):
                raise ValueError('All cached tensors must have the same leading dimension.')
            shapes = {key: tuple(np.shape(value)[1:]) for key, value in tensors.items()}
            if self._shapes is None:
                self._shapes = shapes
            elif shapes != self._shapes:
                raise ValueError('Every observation must provide the same keys and shapes.')
            if indices is not None:
                indices = np.asarray(indices, dtype=np.int64)
                if indices.ndim != 1 or (indices.size and (indices.min() < 0 or indices.max() >= n_rows)):
                    raise ValueError('indices must be row positions of the observation.')
                self._filtered += n_rows - indices.size
            self._sources.append((dict(tensors), indices))
        except Exception:
            self.abort()
            raise

    def _count(self, j):
        tensors, indices = self._sources[j]
        if indices is None:
            return int(np.shape(next(iter(tensors.values())))[0])
        return int(indices.size)

    def _rows(self, key, j, positions=slice(None)):
        tensors, indices = self._sources[j]
        value = np.asarray(tensors[key])
        rows = value[positions] if indices is None else value[indices[positions]]
        return np.asarray(rows, dtype=np.float32)

    # -------------------------------------------------------------- finalize
    def finalize(self, constant_tensors=None):
        """Write the final files and return ``(batches_file_paths, n_rows)``.

        ``constant_tensors`` maps keys to ``(n_observations, ...)`` arrays that
        are constant within an observation; they are tiled into the cache.
        """
        if self._finished:
            raise RuntimeError('The cache writer has already been finalized.')
        try:
            if len(self._sources) != self.n_observations:
                raise ValueError(
                    f'{len(self._sources)} observations were added, {self.n_observations} announced.'
                )
            constant_tensors = {} if constant_tensors is None else dict(constant_tensors)
            for key, value in constant_tensors.items():
                value = np.asarray(value, dtype=np.float32)
                if key in self._shapes or value.shape[:1] != (self.n_observations,):
                    raise ValueError(
                        f'Constant tensor {key!r} must be a new key with one entry per observation.'
                    )
                constant_tensors[key] = value
            if self._filtered:
                print(f'Filtering {self._filtered} non-finite entries')
            if self.shuffle:
                n_rows = self._finalize_stratified(constant_tensors)
            else:
                n_rows = self._finalize_sequential(constant_tensors)
            if n_rows == 0:
                raise ValueError('No samples remain after filtering non-finite entries.')
            for output in self._outputs.values():
                output.flush()
        except Exception:
            self.abort()
            raise
        self._outputs.clear()
        self._sources = []
        self._finished = True
        return dict(self._final_paths), int(n_rows)

    def _finalize_sequential(self, constant_tensors):
        counts = [self._count(j) for j in range(self.n_observations)]
        n_rows = int(sum(counts))
        for key, trailing in self._shapes.items():
            self._open(key, (n_rows, *trailing))
        for key, value in constant_tensors.items():
            self._open(key, (n_rows, *value.shape[1:]))
        start = 0
        for j, count in enumerate(counts):
            for key in self._shapes:
                self._outputs[key][start:start + count] = self._rows(key, j)
            for key, value in constant_tensors.items():
                self._outputs[key][start:start + count] = value[j]
            start += count
        return n_rows

    def _finalize_stratified(self, constant_tensors):
        counts = np.asarray([self._count(j) for j in range(self.n_observations)], dtype=np.int64)
        observations = np.flatnonzero(counts > 0)
        if observations.size == 0:
            return 0
        # One random observation order shared by every key keeps rows aligned.
        order = observations[self.rng.permutation(observations.size)]
        n = int(order.size)
        width = int(counts.max())
        ordered_counts = counts[order]
        n_rows = int(ordered_counts.sum())
        for key, trailing in self._shapes.items():
            self._open(key, (n_rows, *trailing))
        for key, value in constant_tensors.items():
            self._open(key, (n_rows, *value.shape[1:]))

        # The (observation, slot) table is NaN-padded for observations with
        # fewer rows than the widest one. Each block of slots is transposed to
        # slot-major order and its padded rows are dropped, so the output
        # holds exactly the selected rows while every contiguous run still
        # interleaves all observations.
        slots_per_block = max(1, self.block_rows // n)
        blocks = []
        start = 0
        for c0 in range(0, width, slots_per_block):
            c1 = min(c0 + slots_per_block, width)
            keep = (np.arange(c0, c1)[:, None] < ordered_counts[None, :]).reshape(-1)
            blocks.append((c0, c1, keep, start))
            start += int(np.count_nonzero(keep))

        for key, trailing in self._shapes.items():
            stack = np.full((n, width, *trailing), np.nan, dtype=np.float32)
            for out_j, j in enumerate(order):
                stack[out_j, :counts[j]] = self._rows(key, j)
            # The per-observation arrays of this key are no longer needed.
            for tensors, _ in self._sources:
                tensors.pop(key, None)
            for c0, c1, keep, row in blocks:
                block = stack[:, c0:c1].swapaxes(0, 1).reshape(-1, *trailing)[keep]
                self._outputs[key][row:row + len(block)] = block
            del stack
        for key, value in constant_tensors.items():
            ordered = value[order]
            for c0, c1, keep, row in blocks:
                block = np.broadcast_to(ordered[None], (c1 - c0, n, *value.shape[1:]))
                block = block.reshape(-1, *value.shape[1:])[keep]
                self._outputs[key][row:row + len(block)] = block
        return n_rows

    # --------------------------------------------------------------- cleanup
    @staticmethod
    def _close(array):
        mmap = getattr(array, '_mmap', None)
        if mmap is not None:
            mmap.close()

    def abort(self):
        """Remove every file this writer created."""
        for array in self._outputs.values():
            self._close(array)
        self._outputs.clear()
        self._sources = []
        for path in self._created:
            _remove_quietly(path)
        self._created = []
        self._final_paths = {}
        self._finished = True


class TensorsDataset(Dataset):
    """Cache aligned arrays on disk and lazily load batches as float32 tensors."""

    _WRITE_CHUNK_SIZE = 2 ** 22

    def __init__(self, tensors=None, work_directory=None, filter_nans=True, shuffle=True, ds_name=None,
                 valid_mask=None, shuffle_seed=None, batch_size=2 ** 13, cache=None, **kwargs):
        """Cache ``tensors`` (one in-memory observation) or adopt a finished ``cache``.

        ``cache`` is the ``(batches_file_paths, n_rows)`` pair returned by
        ``ObservationCacheWriter.finalize``; subclasses that stream their
        observations through a writer pass it instead of ``tensors``.
        """
        self.batch_size = int(batch_size)
        self.addition_kwargs = kwargs
        self._mmap_arrays = None
        self._mmap_owner_pid = None
        if cache is not None:
            if tensors is not None:
                raise ValueError('Pass either tensors or a finished cache, not both.')
            self.batches_file_paths, self._n_samples = dict(cache[0]), int(cache[1])
            return
        if not tensors:
            raise ValueError('tensors must contain at least one array')
        writer = ObservationCacheWriter(
            work_directory, 1, ds_name=ds_name, shuffle=shuffle,
            shuffle_seed=shuffle_seed, block_rows=self._WRITE_CHUNK_SIZE,
        )
        indices = shuffled_valid_rows(
            tensors, valid_mask if filter_nans else None, filter_rows=filter_nans,
            rng=np.random.default_rng(shuffle_seed) if shuffle else None,
        )
        writer.add(tensors, indices)
        self.batches_file_paths, self._n_samples = writer.finalize()

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
