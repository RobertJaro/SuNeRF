import copy
import os

import numpy as np
import torch
from astropy import units as u
from dateutil.parser import parse

from sunerf.data.date_util import normalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule, TensorsDataset
import xarray as xr

class WaterVaporDataset(TensorsDataset):
    def __init__(self, data_path, work_directory, instrument_key, meters_per_ds, seconds_per_dt=1, ref_date=None,
                 batch_size=int(2 ** 10), test=False, **kwargs):
        data = np.load(data_path, allow_pickle=True)
        image = data['image']
        time = data['time']
        z = data['z']
        x_range = data['x_range']
        obs_angle = data['obs_angle']
        resolution = image.shape[1]

        # normalize datetime
        ref_date = ref_date if ref_date is not None else time
        normalized_time = normalize_datetime(time, seconds_per_dt, ref_date)

        # unpack data
        images = image  # (x, theta)

        angles = np.linspace(-obs_angle.to_value(u.rad) / 2, obs_angle.to_value(u.rad) / 2, resolution)
        rays_d = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=-1)  # (theta, 3)

        # get observer location
        x = x_range.to_value(u.m)
        rays_o = np.stack([x, np.zeros_like(x), np.ones_like(x) * z.to_value(u.m)], -1) / meters_per_ds  # (x, 3)

        rays_d = np.tile(rays_d[None, :, :], (len(x_range), 1, 1))  # (x, theta, 3)
        rays_o = np.tile(rays_o[:, None, :], (1, resolution, 1)) # (x, theta, 3)

        rays = np.stack([rays_o, rays_d], axis=-2)  # (x, theta, 2, 3)

        times = np.ones_like(image) * normalized_time

        data_dict = {'rays': rays, 'image': images, 'time': times}

        tensors = {k: v.reshape((-1, *v.shape[2:])) for k, v in data_dict.items() if k in ['image', 'rays', 'time']}

        self.ref_date = ref_date  # store reference date for normalization
        self.data_config = {'image_shape': image.shape[:2]}
        self.times = time
        self.image_shape = image.shape[:2]

        super().__init__(tensors=tensors, work_directory=work_directory, batch_size=batch_size,
                         shuffle=not test, filter_nans=not test, instrument=instrument_key)


class WaterVaporSliceDataset(TensorsDataset):
    def __init__(self, data_path, work_directory, meters_per_ds, batch_size=int(2 ** 10), test=False, **kwargs):
        file_path = os.path.join(data_path, 'qvapor_test.nc')
        z_file_path = os.path.join(data_path, 'z_test.nc')
        p_file_path = os.path.join(data_path, 'p_test.nc')

        # mixing ratio of water
        qvapor_data = xr.open_dataset(file_path)
        z_data = xr.open_dataset(z_file_path)
        p_data = xr.open_dataset(p_file_path)

        # z = z_data['Z'].values.T  # in meters
        # z = z_data['Z'].values.T  # in meters
        z = np.linspace(0, 1, z_data['Z'].shape[0]) * 15000  # in meters, assuming a fixed height for simplicity
        z = np.tile(z[None, :], (z_data['Z'].shape[1], 1))  # repeat for each longitude
        z = (z[:, 1:] + z[:, :-1]) / 2
        # water vapor density
        # rho_water = mixing ratio * rho_air = mixing ratio * p / (R * T) = C * mixing ratio * p
        rho_true_npy = qvapor_data['QVAPOR'].values.T * p_data['P'].values.T  # in kg/m^3
        rho_true_npy = np.log10(rho_true_npy)  # convert to log scale

        longitude = z_data['XLONG'].values
        x = (1 * u.R_earth).to_value(u.m) * np.cos(np.deg2rad(longitude))
        x = x - x.min()

        coords_npy = np.zeros((*rho_true_npy.shape, 4), dtype=np.float32)
        coords_npy[..., 2] = z
        coords_npy[..., 0] = x[:, None]

        coords_npy[z > 15e3] = np.nan # clip above observer height
        coords_npy = coords_npy / meters_per_ds  # convert to model units

        print('Coordinate range:')
        print(f'X: {np.nanmin(coords_npy[..., 0]):.2f} - {np.nanmax(coords_npy[..., 0]):.2f} ')
        print(f'Y: {np.nanmin(coords_npy[..., 1]):.2f} - {np.nanmax(coords_npy[..., 1]):.2f} ')
        print(f'Z: {np.nanmin(coords_npy[..., 2]):.2f} - {np.nanmax(coords_npy[..., 2]):.2f} ')
        print(f't: {np.nanmin(coords_npy[..., 3]):.2f} - {np.nanmax(coords_npy[..., 3]):.2f} ')

        self.cube_shape = coords_npy.shape[:2] # (x, z)

        data_dict = {'query_points': coords_npy, 'true_log10_rho': rho_true_npy[..., None]}

        tensors = {k: v.reshape((-1, *v.shape[2:])) for k, v in data_dict.items()}

        super().__init__(tensors=tensors, work_directory=work_directory, batch_size=batch_size,
                         shuffle=not test, filter_nans=not test)

class WaterVaporDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, meters_per_ds=1e4, seconds_per_dt=1,
                 ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False, **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        base_config = {'meters_per_ds': meters_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size}

        train_dict = self._load_dataset(train_datasets, base_config)
        ref_date = base_config['ref_date']  # update ref date if not specified

        module_config = {}
        for k, ref_ds in train_dict.items():
            dc = ref_ds.data_config
            module_config[k] = {'type': 'water', 'meters_per_ds': meters_per_ds, 'seconds_per_dt': seconds_per_dt,
                                'ref_date': ref_date, 'image_shape': dc['image_shape'], 'times': ref_ds.times}

        base_config['validation_batch_size'] = validation_batch_size
        valid_dict = self._load_dataset(valid_datasets, base_config, test_ds=True)

        self.meters_per_ds = meters_per_ds
        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=meters_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=module_config, **kwargs)

    def _load_dataset(self, data_config, base_config, test_ds=False):
        N_GPUS = torch.cuda.device_count()
        ref_date = None if 'ref_date' not in base_config else base_config['ref_date']
        data_config = copy.deepcopy(data_config)

        train_dict = {}
        for config in data_config:
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)
            # adjust batch size for multi-gpu
            ds_config['batch_size'] = ds_config['validation_batch_size'] if test_ds else ds_config['batch_size']
            ds_config['batch_size'] = ds_config['batch_size'] * N_GPUS if N_GPUS > 1 else ds_config['batch_size']
            if ds_type == 'QVAPOR':
                dataset = WaterVaporDataset(**ds_config, ds_key=ds_key, test=test_ds)
            elif ds_type == 'QVAPOR-slice':
                dataset = WaterVaporSliceDataset(**ds_config, ds_key=ds_key, test=test_ds)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_date is None:
                ref_date = dataset.ref_date
                base_config['ref_date'] = ref_date
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset
        return train_dict