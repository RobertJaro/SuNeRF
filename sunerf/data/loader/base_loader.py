import glob
import multiprocessing
import os
from itertools import repeat

import numpy as np
from astropy import units as u
from pytorch_lightning import LightningDataModule
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from sunerf.data.dataset import MmapDataset
from sunerf.data.ray_sampling import get_rays
from sunerf.train.coordinate_transformation import pose_spherical


class BaseDataModule(LightningDataModule):

    def __init__(self, training_datasets, validation_datasets,
                 Rs_per_ds, seconds_per_dt, ref_time,
                 module_config,
                 num_workers=None, **kwargs):
        super().__init__()
        self.training_datasets = training_datasets
        self.validation_datasets = validation_datasets
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.ref_time = ref_time

        self.config = module_config
        self.validation_dataset_mapping = {i: name for i, name in enumerate(self.validation_datasets.keys())}
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

    def clear(self):
        [ds.clear() for ds in self.datasets.values() if isinstance(ds, MmapDataset)]

    def train_dataloader(self):
        datasets = self.training_datasets

        # data loader with iterations based on the largest dataset
        ref_idx = np.argmax([len(ds) for ds in datasets.values()])
        ref_dataset_name, ref_dataset = list(datasets.items())[ref_idx]
        loaders = {ref_dataset_name: DataLoader(ref_dataset, batch_size=None, num_workers=self.num_workers,
                                                pin_memory=True, shuffle=True)}
        for i, (name, dataset) in enumerate(datasets.items()):
            if i == ref_idx:
                continue  # reference dataset already added
            sampler = RandomSampler(dataset, replacement=True, num_samples=len(ref_dataset))
            loaders[name] = DataLoader(dataset, batch_size=None, num_workers=self.num_workers,
                                       pin_memory=True, sampler=sampler)
        return loaders

    def val_dataloader(self):
        datasets = self.validation_datasets
        loaders = []
        for dataset in datasets.values():
            loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                                shuffle=False)
            loaders.append(loader)
        return loaders


def get_data(data_path, Rs_per_ds, debug=False):
    files = sorted(glob.glob(data_path))
    if debug:
        files = files[::10]

    with multiprocessing.Pool(os.cpu_count()) as p:
        data = [v for v in
                tqdm(p.imap(_load_map_data, zip(files, repeat(Rs_per_ds))), total=len(files), desc='Loading data')]
    data_dict = {}
    for k in data[0].keys():
        data_dict[k] = np.stack([d[k] for d in data], axis=0)

    ref_map = Map(files[0])
    data_dict['resolution'] = ref_map.data.shape
    data_dict['wcs'] = ref_map.wcs
    data_dict['wavelength'] = ref_map.wavelength

    return data_dict


def _load_map_data(data):
    map_path, Rs_per_ds = data

    s_map = Map(map_path)
    time = s_map.date.datetime

    pose = pose_spherical(-s_map.carrington_longitude.to(u.rad).value,
                          s_map.carrington_latitude.to(u.rad).value,
                          s_map.dsun.to_value(u.solRad) / Rs_per_ds).float().numpy()

    image = s_map.data.astype(np.float32)
    img_coords = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    all_rays = np.stack(get_rays(img_coords, pose), -2)

    all_rays = all_rays.reshape((-1, 2, 3))

    return {'image': image, 'pose': pose, 'all_rays': all_rays, 'time': time}
