import glob
import hashlib
import multiprocessing
import os
import re
import uuid

import numpy as np
import torch
from astropy import units as u
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sunerf.data.dataset import MmapDataset, IndexedDataset
from sunerf.data.ray_sampling import get_rays, hpc_impact_parameter
from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.train.coordinate_transformation import pose_spherical


class BaseDataModule(LightningDataModule):

    def __init__(self, training_datasets, validation_datasets,
                 Rs_per_ds, seconds_per_dt, ref_date,
                 module_config,
                 num_workers=None, **kwargs):
        super().__init__()
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_date = ref_date

        self.config = module_config
        self.validation_dataset_mapping = {i: name for i, name in enumerate(self.validation_datasets.keys())}
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def clear(self):
        """Remove only the mmap files explicitly owned by this data module."""
        cleared = set()
        # Training and validation commonly reuse the same dictionary key. Do not
        # iterate ``self.datasets`` here because its merged mapping necessarily
        # hides the training entry in that case.
        for dataset in (*self.training_datasets.values(), *self.validation_datasets.values()):
            # Validation datasets are commonly wrapped in RenderModeDataset.
            while hasattr(dataset, 'dataset'):
                dataset = dataset.dataset
            if isinstance(dataset, MmapDataset) and id(dataset) not in cleared:
                dataset.clear()
                cleared.add(id(dataset))

    @staticmethod
    def _mmap_worker_capacity(dataset):
        base_dataset = dataset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        if not isinstance(base_dataset, MmapDataset) or len(dataset) <= 1:
            return 0
        return len(dataset)

    def _loader_kwargs(self, dataset, *, training, worker_limit=None):
        base_dataset = dataset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        # Generated random-coordinate datasets do all their work in one vectorized
        # call. Spawning workers for them adds processes without adding parallelism.
        requested_workers = self.num_workers if worker_limit is None else worker_limit
        requested_workers = max(int(requested_workers or 0), 0)
        workers = min(requested_workers, self._mmap_worker_capacity(dataset))

        kwargs = {
            'batch_size': None,
            'num_workers': workers,
            'pin_memory': torch.cuda.is_available(),
            'shuffle': bool(
                training
                and len(dataset) > 1
                and getattr(base_dataset, 'randomize_batches', True)
            ),
        }
        if workers > 0:
            kwargs.update(persistent_workers=training, prefetch_factor=2)
        return kwargs

    def _training_worker_allocations(self):
        """Distribute the configured per-rank worker budget across train loaders.

        ``CombinedLoader`` keeps every training loader active at once. Treating
        ``num_workers`` as a per-loader value therefore multiplies the process
        count by both the number of data sets and the number of DDP ranks. A
        balanced budget retains prefetching for the largest mmap data sets while
        keeping the total at or below the configured value on each rank.
        """
        allocations = {name: 0 for name in self.training_datasets}
        remaining = max(int(self.num_workers or 0), 0)
        candidates = [
            (name, self._mmap_worker_capacity(dataset), order)
            for order, (name, dataset) in enumerate(self.training_datasets.items())
            if self._mmap_worker_capacity(dataset) > 0
        ]
        candidates.sort(key=lambda item: (-item[1], item[2]))

        while remaining > 0 and candidates:
            made_progress = False
            for name, capacity, _ in candidates:
                if allocations[name] >= capacity:
                    continue
                allocations[name] += 1
                remaining -= 1
                made_progress = True
                if remaining == 0:
                    break
            if not made_progress:
                break
        return allocations

    def train_dataloader(self):
        worker_allocations = self._training_worker_allocations()
        loaders = {name: DataLoader(
            ds,
            **self._loader_kwargs(
                ds, training=True, worker_limit=worker_allocations[name]
            ),
        )
                   for name, ds in self.training_datasets.items()}
        return CombinedLoader(loaders, 'max_size_cycle')

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            dataset = IndexedDataset(dataset)
            loader = DataLoader(dataset, **self._loader_kwargs(dataset, training=False))
            loaders.append(loader)
        return loaders


def get_data(data_path, Rs_per_ds, debug=False):
    files = sorted(glob.glob(data_path))
    if debug:
        files = files[::10]

    with multiprocessing.Pool(os.cpu_count()) as p:
        loader = MapDataLoader(Rs_per_ds=Rs_per_ds)
        data = [v for v in
                tqdm(p.imap(loader.load, files), total=len(files), desc='Loading data')]
    data_dict = {}
    for k in data[0].keys():
        data_dict[k] = np.stack([d[k] for d in data], axis=0)

    ref_map = Map(files[0])
    data_dict['resolution'] = ref_map.data.shape
    data_dict['wcs'] = ref_map.wcs
    data_dict['wavelength'] = ref_map.wavelength

    return data_dict


class MapDataLoader:

    def __init__(self, Rs_per_ds, reference_frame='carrington', max_radius=None, azimuthal_equidistant=False):
        self.Rs_per_ds = Rs_per_ds
        self.reference_frame = reference_frame
        self.max_radius = max_radius
        self.azimuthal_equidistant = azimuthal_equidistant

    def load(self, map_path):
        s_map = Map(map_path)
        time = s_map.date.datetime

        if self.reference_frame == 'carrington':
            pose = pose_spherical(s_map.carrington_longitude.to(u.rad).value,
                                  s_map.carrington_latitude.to(u.rad).value,
                                  s_map.dsun.to_value(u.solRad) / self.Rs_per_ds)
            observer = {'radius': s_map.dsun.to(u.solRad),
                        'latitude': s_map.carrington_latitude.to(u.deg),
                        'longitude': s_map.carrington_longitude.to(u.deg),
                        'time': time}
        elif self.reference_frame == 'heliographic':
            pose = pose_spherical(s_map.heliographic_longitude.to(u.rad).value,
                                  s_map.heliographic_latitude.to(u.rad).value,
                                  s_map.dsun.to_value(u.solRad) / self.Rs_per_ds)
            observer = {'radius': s_map.dsun.to(u.solRad),
                        'latitude': s_map.heliographic_latitude.to(u.deg),
                        'longitude': s_map.heliographic_longitude.to(u.deg),
                        'time': time}
        elif self.reference_frame == 'inertial':
            obs_coord = s_map.observer_coordinate.transform_to(frames.HeliocentricInertial)
            pose = pose_spherical(obs_coord.lon.to_value(u.rad),
                                  obs_coord.lat.to_value(u.rad),
                                  obs_coord.distance.to_value(u.solRad) / self.Rs_per_ds)
            observer = {'radius': obs_coord.distance.to(u.solRad),
                        'latitude': obs_coord.lat.to(u.deg),
                        'longitude': obs_coord.lon.to(u.deg),
                        'time': time}
        else:
            raise ValueError('reference_frame must be "heliographic" or "carrington"')

        image = s_map.data.astype(np.float32)

        if self.azimuthal_equidistant:
            img_coords = get_azimuthal_equidistant_coordinates(s_map)
            x = img_coords[..., 0]
            y = img_coords[..., 1]
        else:
            coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
            x = coords.Tx
            y = coords.Ty

        # Use the exact line-of-sight impact parameter.  The previous
        # hypot(Tx, Ty) / angular_solar_radius expression is a small-angle
        # approximation and clips the outer part of wide-field PUNCH images.
        projected_radius = hpc_impact_parameter(x, y, s_map.dsun).to_value(u.R_sun)

        all_rays = np.stack(get_rays(x, y, pose), -2).astype(np.float32, copy=False)

        distance = np.ones_like(x.to_value(u.arcsec)) * s_map.dsun.to_value(u.solRad)
        hpc_coords = np.stack(
            [x.to_value(u.arcsec), y.to_value(u.arcsec), distance], -1
        ).astype(np.float32, copy=False)

        if self.max_radius is not None:
            mask = projected_radius > self.max_radius
            # apply mask
            all_rays[mask] = np.nan
            image[mask] = np.nan

        return {
            'image': image,
            'pose': pose,
            'rays': all_rays,
            'time': time,
            'observer': observer,
            'hpc_coords': hpc_coords,
            'projected_radius': projected_radius.astype(np.float32),
        }


class BatchesDataset(MmapDataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)
        self.addition_kwargs = kwargs
        self._mmap_arrays = None
        self._mmap_owner_pid = None
        self._n_samples = None

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


class TensorsDataset(BatchesDataset):

    _WRITE_CHUNK_SIZE = 2 ** 20

    def __init__(self, tensors, work_directory, filter_nans=True, shuffle=True, ds_name=None,
                 valid_mask=None, shuffle_seed=None, **kwargs):
        os.makedirs(work_directory, exist_ok=True)
        if not tensors:
            raise ValueError('tensors must contain at least one array')
        n_samples = int(next(iter(tensors.values())).shape[0])
        if any(int(tensor.shape[0]) != n_samples for tensor in tensors.values()):
            raise ValueError('All cached tensors must have the same leading dimension.')

        # Build the filter a chunk at a time. This keeps peak memory bounded for
        # multi-billion-pixel image stacks and preserves the historical rule:
        # remove a row only when every tensor has at least one NaN in that row.
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
                    invalid_in_every_tensor = np.ones(stop - start, dtype=bool)
                    for tensor in tensors.values():
                        chunk = np.asarray(tensor[start:stop])
                        axes = tuple(range(1, chunk.ndim))
                        nonfinite = ~np.isfinite(chunk)
                        invalid = nonfinite if not axes else np.any(nonfinite, axis=axes)
                        invalid_in_every_tensor &= invalid
                    keep_mask[start:stop] = ~invalid_in_every_tensor
            filtered_count = int(n_samples - np.count_nonzero(keep_mask))
            if filtered_count:
                print(f'Filtering {filtered_count} nan entries')
        output_size = n_samples if keep_mask is None else int(np.count_nonzero(keep_mask))
        if output_size == 0:
            raise ValueError('No samples remain after filtering NaN entries.')

        # Shuffle bounded source chunks and the rows within each chunk while
        # writing the cache. This restores row-level mixing without allocating a
        # permutation proportional to the full data set. DataLoader also changes
        # the order of the contiguous on-disk batches each epoch.
        self.randomize_batches = bool(shuffle)

        chunk_bounds = [
            (start, min(start + self._WRITE_CHUNK_SIZE, n_samples))
            for start in range(0, n_samples, self._WRITE_CHUNK_SIZE)
        ]
        chunk_order = np.arange(len(chunk_bounds))
        if shuffle:
            if shuffle_seed is None:
                shuffle_seed = uuid.uuid4().int & ((1 << 63) - 1)
            shuffle_rng = np.random.default_rng(shuffle_seed)
            shuffle_rng.shuffle(chunk_order)
            chunk_seeds = shuffle_rng.integers(
                0, np.iinfo(np.int64).max, size=len(chunk_bounds), dtype=np.int64
            )
        else:
            chunk_seeds = None

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

            write_offset = 0
            for chunk_index in chunk_order:
                start, stop = chunk_bounds[int(chunk_index)]
                if keep_mask is None:
                    local_indices = np.arange(stop - start)
                else:
                    local_indices = np.flatnonzero(keep_mask[start:stop])
                if shuffle and local_indices.size > 1:
                    local_rng = np.random.default_rng(int(chunk_seeds[chunk_index]))
                    local_rng.shuffle(local_indices)

                next_offset = write_offset + local_indices.size
                for key, tensor in tensors.items():
                    source_chunk = np.asarray(tensor[start:stop])
                    outputs[key][write_offset:next_offset] = source_chunk[local_indices]
                write_offset = next_offset

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
        super().__init__(batches_paths, **kwargs)
        self._n_samples = output_size
