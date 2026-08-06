import glob
import multiprocessing
import os
import uuid

import numpy as np
import torch
from astropy import units as u
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm

from sunerf.data.dataset import MmapDataset, IndexedDataset
from sunerf.data.ray_sampling import get_rays
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
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, MmapDataset)]

    def train_dataloader(self):
        loaders = {name: DataLoader(ds, batch_size=None, num_workers=self.num_workers,
                                    pin_memory=False, shuffle=True, persistent_workers=True, prefetch_factor=5)
                   for name, ds in self.training_datasets.items()}
        return CombinedLoader(loaders, 'max_size_cycle')

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            dataset = IndexedDataset(dataset)
            loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=False,
                                shuffle=False)
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
            projected_radius = np.sqrt(x ** 2 + y ** 2) / s_map.rsun_obs.to_value(u.arcsec)
        else:
            coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
            x = coords.Tx
            y = coords.Ty
            projected_radius = (np.sqrt(x ** 2 + y ** 2) / s_map.rsun_obs).to_value(1)

        all_rays = np.stack(get_rays(x, y, pose), -2)

        distance = np.ones_like(x.to_value(u.arcsec)) * s_map.dsun.to_value(u.solRad)
        hpc_coords = np.stack([x.to_value(u.arcsec), y.to_value(u.arcsec), distance], -1)

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


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)
        self.addition_kwargs = kwargs

    def __len__(self):
        ref_file = list(self.batches_file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: torch.tensor(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size], dtype=torch.float32)
                for k, bf in self.batches_file_paths.items()}
        data.update(self.addition_kwargs)
        return data

    def clear(self):
        [os.remove(f) for f in self.batches_file_paths.values()]


class TensorsDataset(BatchesDataset):

    def __init__(self, tensors, work_directory, filter_nans=True, shuffle=True, ds_name=None, **kwargs):
        os.makedirs(work_directory, exist_ok=True)
        # filter nan entries
        nan_mask = np.all([np.any(np.isnan(t), axis=tuple(range(1, t.ndim))) for t in tensors.values()], axis=0)
        if nan_mask.sum() > 0 and filter_nans:
            print(f'Filtering {nan_mask.sum()} nan entries')
            tensors = {k: v[~nan_mask] for k, v in tensors.items()}

        # shuffle data
        if shuffle:
            r = np.random.permutation(list(tensors.values())[0].shape[0])
            tensors = {k: v[r] for k, v in tensors.items()}

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        batches_paths = {}
        for k, v in tensors.items():
            coords_npy_path = os.path.join(work_directory, f'{ds_name}_{k}.npy')
            np.save(coords_npy_path, v.astype(np.float32))
            batches_paths[k] = coords_npy_path
        super().__init__(batches_paths, **kwargs)
