import glob
import os

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from sunerf.data.dataset import IndexedDataset
from sunerf.data.loader.base_loader import MapDataLoader


class ConditionedDataModule(LightningDataModule):

    def __init__(self, data_path, work_directory, patch_size=(256, 256), Rs_per_ds=1, batch_size=8, n_rays=256, cmap='gray', num_workers=None, **kwargs):
        self.Rs_per_ds = Rs_per_ds
        self.cmap = cmap
        self.num_workers = os.cpu_count() if num_workers is None else num_workers

        os.makedirs(work_directory, exist_ok=True)

        data_files = sorted(glob.glob(data_path))
        assert len(data_files) > 0, f"No files found for input pattern: {data_path}"

        # select test image
        test_idx = len(data_files) // 2
        mask = np.ones(len(data_files), dtype=bool)
        mask[test_idx] = False

        train_files = np.array(data_files)[mask].tolist()
        valid_file = data_files[test_idx]

        self.image_norm = 1e4
        self.arcsec_norm = 1e3
        self.train_dataset = ConditionedMapDataset(train_files, patch_size=patch_size, n_rays=n_rays,
                                                   image_norm=self.image_norm, arcsec_norm=self.arcsec_norm)
        self.valid_dataset = FullImageDataset(valid_file, patch_size=patch_size, batch_size=batch_size * n_rays,
                                              image_norm=self.image_norm, arcsec_norm=self.arcsec_norm)
        self.validation_dataset_mapping = {0: 'image'}

        self.config = {'type': 'conditioned', 'Rs_per_ds': Rs_per_ds, 'cmap': cmap, 'resolution': (256, 256),
                       'channels': self.train_dataset.channels}
        self.batch_size = batch_size
        super().__init__()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True,
                          prefetch_factor=5)

    def val_dataloader(self):
        dataset = self.valid_dataset
        dataset = IndexedDataset(dataset)
        loader = DataLoader(dataset, batch_size=None, num_workers=self.num_workers, pin_memory=True,
                            shuffle=False, persistent_workers=True, prefetch_factor=5)
        return [loader,]

class ConditionedMapDataset(Dataset):

    def __init__(self, data_files, patch_size=None, n_rays=32, image_norm=10000., arcsec_norm=1000., add_hpc=True):
        self.data_files = data_files
        self.patch_size = patch_size
        self.n_rays = n_rays
        self.scaling = image_norm
        self.arcsec_norm = arcsec_norm
        self.add_hpc = add_hpc
        self.channels = 3 if add_hpc else 1
        self.loader = MapDataLoader(Rs_per_ds=1, reference_frame="carrington", add_hpc=True)
        super().__init__()

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = self.loader.load(self.data_files[idx])
        image = data["image"]
        rays = data["rays"]
        hpc = data["hpc"]
        latitude = np.deg2rad(data["observer"]['latitude'])
        longitude = np.deg2rad(data["observer"]['longitude'])

        if self.patch_size is not None:
            # randomly sample a patch
            h, w = image.shape[:2]
            ph, pw = self.patch_size
            top = np.random.randint(0, h - ph)
            left = np.random.randint(0, w - pw)
            image = image[top:top + ph, left:left + pw]
            rays = rays[top:top + ph, left:left + pw]
            hpc = hpc[top:top + ph, left:left + pw, :]

        # normalize data
        image = image / self.scaling
        hpc = hpc / self.arcsec_norm

        # randomly sample n_rays
        h, w = image.shape[:2]
        possible_indices = np.arange(h*w)
        flat_image = image.reshape(-1)
        possible_indices = possible_indices[~np.isnan(flat_image)]
        # replace True to avoid error when most pixels are NaN
        indices = np.random.choice(possible_indices, size=self.n_rays, replace=True)
        target_image = flat_image[indices]
        rays = rays.reshape(-1, *rays.shape[-2:])[indices]

        # convert to channels first format
        image = image[None, :, :] # [C, H, W]
        image = np.nan_to_num(image, nan=0.0)

        # add HPC coordinate information
        if self.add_hpc:
            hpc = hpc.transpose(2, 0, 1)  # [2, H, W]
            input_image = np.concatenate([image, hpc], axis=0)  # [C, H, W]
        else:
            input_image = image  # [C, H, W]

        return {'target_image': torch.tensor(target_image, dtype=torch.float32),
                'rays': torch.tensor(rays, dtype=torch.float32),
                'input_image': torch.tensor(input_image, dtype=torch.float32),
                'longitude': torch.tensor(longitude, dtype=torch.float32),
                'latitude': torch.tensor(latitude, dtype=torch.float32)}


class FullImageDataset(Dataset):

    def __init__(self, data_file, patch_size=None, batch_size=32, image_norm=10000., arcsec_norm=1000., add_hpc=True):
        self.batch_size = batch_size
        self.add_hpc = add_hpc
        self.channels = 3 if add_hpc else 1

        loader = MapDataLoader(Rs_per_ds=1, reference_frame="carrington", add_hpc=True)
        # load full image and rays
        data_dict = loader.load(data_file)
        image = data_dict['image'] / image_norm
        rays = data_dict['rays']
        hpc = data_dict['hpc'] / arcsec_norm
        self.longitude = np.deg2rad(data_dict['observer']['longitude'])
        self.latitude = np.deg2rad(data_dict['observer']['latitude'])

        # extract patch
        if patch_size is not None:
            # randomly sample a patch
            h, w = image.shape[:2]
            ph, pw = patch_size
            top = np.random.randint(0, h - ph)
            left = np.random.randint(0, w - pw)
            image = image[top:top + ph, left:left + pw]
            rays = rays[top:top + ph, left:left + pw]
            hpc = hpc[top:top + ph, left:left + pw, :]

        # convert to channels first format
        image = image[None, :, :] # [C, H, W]
        image = np.nan_to_num(image, nan=0.0)

        # add HPC coordinate information
        if self.add_hpc:
            hpc = hpc.transpose(2, 0, 1)  # [2, H, W]
            input_image = np.concatenate([image, hpc], axis=0)  # [C, H, W]
        else:
            input_image = image  # [C, H, W]

        self.flat_image = image.reshape(-1, 1)
        self.flat_rays = rays.reshape(-1, *rays.shape[-2:])
        self.input_image = input_image
        self.n_pixels = input_image.shape[1] * input_image.shape[2]

        super().__init__()

    def __len__(self):
        return np.ceil(self.n_pixels / self.batch_size).astype(int)

    def __getitem__(self, idx):
        target_image = self.flat_image[idx * self.batch_size: (idx + 1) * self.batch_size]
        rays = self.flat_rays[idx * self.batch_size: (idx + 1) * self.batch_size]

        # expand dims to add batch dimension
        input_image = self.input_image[None, :, :, :] # [B, C, H, W]
        target_image = target_image[None, :, :]  # [B, N_rays, C]
        rays = rays[None, :, :]  # [B, N_rays, 2, 3]

        # handle NaNs
        input_image = np.nan_to_num(input_image, nan=0.0)

        return {'target_image': torch.tensor(target_image, dtype=torch.float32),
                'rays': torch.tensor(rays, dtype=torch.float32),
                'input_image': torch.tensor(input_image, dtype=torch.float32),
                'longitude': torch.tensor(self.longitude, dtype=torch.float32).reshape(1,),
                'latitude': torch.tensor(self.latitude, dtype=torch.float32).reshape(1,)}
