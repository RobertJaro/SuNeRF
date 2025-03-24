import logging
import os

import numpy as np
import torch
from dateutil.parser import parse

from sunerf.data.dataset import MmapDataset, ArrayDataset
from sunerf.data.date_util import normalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule, get_data
from sunerf.train.callback import log_overview


class SingleChannelDataModule(BaseDataModule):

    def __init__(self, data_path, work_directory, Rs_per_ds=1, seconds_per_dt=86400, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, cmap='gray', **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        data_dict = get_data(data_path=data_path, Rs_per_ds=Rs_per_ds, debug=debug)

        o_times = data_dict['time']

        # normalize datetime
        ref_date = parse(ref_date) if ref_date is not None else min(o_times)
        times = np.array([normalize_datetime(t, seconds_per_dt, ref_date) for t in o_times], dtype=np.float32)

        # unpack data
        images = data_dict['image']
        rays = data_dict['rays']

        log_overview(images[..., None], data_dict['pose'], times, cmap, seconds_per_dt, ref_date)

        # select test image
        test_idx = len(images) // 6
        mask = np.ones(len(images), dtype=bool)
        mask[test_idx] = False

        valid_rays, valid_times, valid_images = rays[~mask], times[~mask], images[~mask]

        # load all training rays
        rays, times, images = rays[mask], times[mask], images[mask]

        # flatten rays
        rays = rays.reshape((-1, 2, 3))
        times = np.ones_like(images) * times[:, None, None]  # broadcast time to image shape
        times = times.reshape(-1, 1)
        images = images.reshape(-1, 1)

        # shuffle
        r = np.random.permutation(rays.shape[0])
        rays, times, images = rays[r], times[r], images[r]

        # save npy files
        # create file names
        logging.info('Save batches to disk')
        npy_rays = os.path.join(work_directory, 'rays_batches.npy')
        npy_times = os.path.join(work_directory, 'times_batches.npy')
        npy_images = os.path.join(work_directory, 'images_batches.npy')

        # save to disk
        np.save(npy_rays, rays)
        np.save(npy_times, times)
        np.save(npy_images, images)

        # adjust batch size
        N_GPUS = torch.cuda.device_count()
        batch_size = int(batch_size) * N_GPUS

        # init train dataset
        train_dataset = MmapDataset({'image': npy_images, 'rays': npy_rays, 'time': npy_times},
                                    batch_size=batch_size)

        valid_rays = valid_rays.reshape((-1, 2, 3))
        valid_times = np.ones_like(valid_images) * valid_times[:, None, None]
        valid_times = valid_times.reshape(-1, 1)
        valid_images = valid_images.reshape(-1, 1)

        valid_dataset = ArrayDataset({'image': valid_images, 'rays': valid_rays, 'time': valid_times},
                                     batch_size=batch_size)

        config = {'type': 'emission', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                  'wcs': data_dict['wcs'], 'resolution': data_dict['resolution'], 'wavelength': data_dict['wavelength'],
                  'times': o_times, 'cmap': cmap}
        super().__init__({'tracing': train_dataset}, {'test_image': valid_dataset},
                         start_time=o_times.min(), end_time=o_times.max(),
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=config, **kwargs)
