import glob
import multiprocessing
import os

import numpy as np
import torch
from astropy import units as u
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from sunerf.data.dataset import IndexedDataset, TensorsDataset
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
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 0)

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
            if isinstance(dataset, TensorsDataset) and id(dataset) not in cleared:
                dataset.clear()
                cleared.add(id(dataset))

    @staticmethod
    def _mmap_worker_capacity(dataset):
        base_dataset = dataset
        while hasattr(base_dataset, 'dataset'):
            base_dataset = base_dataset.dataset

        if not isinstance(base_dataset, TensorsDataset) or len(dataset) <= 1:
            return 0
        return len(dataset)

    def _loader_kwargs(self, dataset, *, training, worker_limit=None):
        # Generated random-coordinate datasets do all their work in one vectorized
        # call. Spawning workers for them adds processes without adding parallelism.
        requested_workers = self.num_workers if worker_limit is None else worker_limit
        requested_workers = max(int(requested_workers or 0), 0)
        workers = min(requested_workers, self._mmap_worker_capacity(dataset))

        kwargs = {
            'batch_size': None,
            'num_workers': workers,
            'pin_memory': torch.cuda.is_available(),
            # Dataset indices identify whole contiguous batches in the cache.
            'shuffle': bool(training and len(dataset) > 1),
        }
        # Distribute and shuffle batch indices, never individual cached rows.
        trainer = getattr(self, 'trainer', None)
        if training and trainer is not None and trainer.world_size > 1:
            kwargs['shuffle'] = False  # The explicit sampler owns shuffling.
            kwargs['sampler'] = DistributedSampler(
                dataset, num_replicas=trainer.world_size,
                rank=trainer.global_rank, shuffle=True,
            )
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

    if not files:
        raise ValueError(f'No map files found for {data_path!r}.')

    loader = MapDataLoader(Rs_per_ds=Rs_per_ds)
    workers = min(os.cpu_count() or 1, len(files))
    if workers == 1:
        data = [loader.load(path) for path in tqdm(files, desc='Loading data')]
    else:
        with multiprocessing.Pool(workers) as pool:
            data = [
                value for value in
                tqdm(pool.imap(loader.load, files), total=len(files), desc='Loading data')
            ]
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
        # Accept an already-open map so a multi-channel loader can validate WCS
        # and construct rays without reading the reference FITS file twice.
        s_map = map_path if hasattr(map_path, 'coordinate_frame') else Map(map_path)
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
            raise ValueError(
                'reference_frame must be "heliographic", "carrington", or "inertial"'
            )

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
