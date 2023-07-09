import glob
import logging
import multiprocessing
import os
from datetime import datetime, timedelta

import numpy as np
import torch
from astropy import units as u
from pytorch_lightning import LightningDataModule
from sunpy.map import Map
from torch.utils.data import Dataset, DataLoader

from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.ray_sampling import get_rays


class NeRFDataModule(LightningDataModule):

    def __init__(self, hparams):
        super().__init__()

        logging.info('Load data')
        images, poses, rays, times, focal_lengths, wavelength = get_data(hparams)

        # fix time offset for transfer learning (TODO replace)
        if 'time_offset' in hparams:
            times = np.array(times) + hparams['time_offset']

        # keep data for logging
        self.images = images
        self.poses = np.array(poses)
        self.times = np.array(times)
        self.wavelength = wavelength

        self.working_dir = hparams['Training']['working_directory']
        os.makedirs(self.working_dir, exist_ok=True)

        N_GPUS = torch.cuda.device_count()
        self.batch_size = int(hparams['Training']['batch_size']) * N_GPUS

        num_workers = hparams['Training']['num_workers']
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2

        # use an image of the first instrument (SDO)
        test_idx = len(images) // 6

        valid_rays, valid_times, valid_images = self._flatten_data([v for i, v in enumerate(rays) if i == test_idx],
                                                                   [v for i, v in enumerate(times) if i == test_idx],
                                                                   [v for i, v in enumerate(images) if i == test_idx])
        # batching
        logging.info('Convert data to batches')
        n_batches = int(np.ceil(valid_rays.shape[0] / self.batch_size))
        valid_rays, valid_times, valid_images = np.array_split(valid_rays, n_batches), \
                                                np.array_split(valid_times, n_batches), \
                                                np.array_split(valid_images, n_batches)
        self.valid_rays, self.valid_times, self.valid_images = valid_rays, valid_times, valid_images
        self.test_kwargs = {'focal': focal_lengths[test_idx],
                            'resolution': images[test_idx].shape[0]}

        # load all training rays
        rays, times, images = self._flatten_data([v for i, v in enumerate(rays) if i != test_idx],
                                                 [v for i, v in enumerate(times) if i != test_idx],
                                                 [v for i, v in enumerate(images) if i != test_idx])
        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images = rays[r], times[r], images[r]
        # account for unequal length of last batch by padding with data from beginning
        pad = self.batch_size - rays.shape[0] % self.batch_size
        rays = np.concatenate([rays, rays[:pad]])
        times = np.concatenate([times, times[:pad]])
        images = np.concatenate([images, images[:pad]])
        # batching
        n_batches = rays.shape[0] // self.batch_size
        rays, times, images = np.array(np.split(rays, n_batches), dtype=np.float32), \
                              np.array(np.split(times, n_batches), dtype=np.float32), \
                              np.array(np.split(images, n_batches), dtype=np.float32)
        # save npy files
        # create file names
        logging.info('Save batches to disk')
        batch_file_rays = os.path.join(self.working_dir, 'rays_batches.npy')
        batch_file_times = os.path.join(self.working_dir, 'times_batches.npy')
        batch_file_images = os.path.join(self.working_dir, 'images_batches.npy')
        # save to disk
        np.save(batch_file_rays, rays)
        np.save(batch_file_times, times)
        np.save(batch_file_images, images)
        # to dataset
        self.train_rays, self.train_times, self.train_images = batch_file_rays, batch_file_times, batch_file_images

        logging.info('Create data sets')
        self.valid_data = BatchesDataset(self.valid_rays, self.valid_times, self.valid_images)
        self.train_data = NumpyFilesDataset(self.train_rays, self.train_times, self.train_images)

    def _flatten_data(self, rays, times, images):
        flat_rays = np.concatenate(rays)
        flat_images = np.concatenate([i.reshape((-1, 1)) for i in images])
        flat_time = np.concatenate([np.ones(i.shape[:3], dtype=np.float32).reshape((-1, 1)) * t
                                    for t, i in zip(times, images)])
        return flat_rays, flat_time, flat_images

    def train_dataloader(self):
        # handle batching manually
        return DataLoader(self.train_data, batch_size=None, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        # handle batching manually
        return DataLoader(self.valid_data, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                          shuffle=False)


class BatchesDataset(Dataset):

    def __init__(self, *batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches[0])

    def __getitem__(self, idx):
        return [b[idx] for b in self.batches]


class NumpyFilesDataset(Dataset):

    def __init__(self, *paths):
        self.paths = paths

    def __len__(self):
        return len(np.load(self.paths[0], mmap_mode='r'))

    def __getitem__(self, idx):
        return [np.copy(np.load(batch_file, mmap_mode='r')[idx]) for batch_file in self.paths]


def get_data(config_data):
    debug = config_data['Debug']

    data_path = config_data['data_path']

    s_maps = sorted(glob.glob(data_path))
    if debug:
        s_maps = s_maps[::10]

    with multiprocessing.Pool(os.cpu_count()) as p:
        data = p.starmap(_load_map_data, zip(s_maps))
    images, poses, rays, times, focal_lengths = list(map(list, zip(*data)))

    ref_wavelength = Map(s_maps[0]).wavelength.value
    return images, poses, rays, times, focal_lengths, ref_wavelength


def _load_map_data(map_path):
    s_map = Map(map_path)
    # compute focal length
    scale = s_map.scale[0]  # scale of pixels [arcsec/pixel]
    W = s_map.data.shape[0]  # number of pixels
    focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)

    time = normalize_datetime(s_map.date.datetime)

    pose = pose_spherical(-s_map.carrington_longitude.to(u.deg).value,
                          s_map.carrington_latitude.to(u.deg).value,
                          s_map.dsun.to(u.solRad).value).float().numpy()

    image = s_map.data.astype(np.float32)
    all_rays = np.stack(get_rays(image.shape[0], image.shape[1], s_map.reference_pixel, focal, pose), -2)

    # crop to square
    min_dim = min(image.shape[:2])
    image = image[:min_dim, :min_dim]
    all_rays = all_rays[:min_dim, :min_dim]

    all_rays = all_rays.reshape((-1, 2, 3))

    return image, pose, all_rays, time, focal


def normalize_datetime(date, max_time_range=timedelta(days=30)):
    """Normalizes datetime object for ML input.

    Time starts at 2010-01-01 with max time range == 2 pi
    Parameters
    ----------
    date: input date

    Returns
    -------
    normalized date
    """
    return (date - datetime(2010, 1, 1)) / max_time_range * (2 * np.pi)


def unnormalize_datetime(norm_date: float) -> datetime:
    """Computes the actual datetime from a normalized date.

    Parameters
    ----------
    norm_date: normalized date

    Returns
    -------
    real datetime
    """
    return norm_date * timedelta(days=30) / (2 * np.pi) + datetime(2010, 1, 1)
