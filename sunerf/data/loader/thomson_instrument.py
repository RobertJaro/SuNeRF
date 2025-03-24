import copy
import glob
import multiprocessing
import os
from datetime import timedelta
from itertools import repeat

import numpy as np
import scipy
import torch
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule, TensorsDataset, _load_map_data
from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset
from sunerf.train.callback import log_overview
from sunerf.train.coordinate_transformation import spherical_to_cartesian


class ThomsonDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, Rs_per_ds, seconds_per_dt, ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False,
                 **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        N_GPUS = torch.cuda.device_count() # adjust batch size for multi-gpu
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size * N_GPUS}

        train_dict = self._load_dataset(train_datasets, base_config)
        module_config = {}
        for k, ref_ds in train_dict.items():
            if not isinstance(ref_ds, GenericThomsonDataset):
                continue
            dc = ref_ds.data_config
            module_config[k] = {'type': 'thomson', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt,
                                'ref_date': ref_date, 'image_scaling': ref_ds.scaling,
                                'wcs': dc['wcs'], 'image_shape': dc['image_shape'], 'times': ref_ds.times}

        base_config['batch_size'] = validation_batch_size
        valid_dict = self._load_valid_dataset(valid_datasets, base_config)

        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=module_config, **kwargs)

    def _load_dataset(self, data_config, base_config):
        ref_date = None if 'ref_date' not in base_config else base_config['ref_date']
        data_config = copy.deepcopy(data_config)

        train_dict = {}
        for config in data_config:
            config = copy.deepcopy(config)
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)

            if ds_type.lower() == 'hao':
                dataset = HAOThomsonDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'random':
                assert len(train_dict) > 0, 'Specify at least one dataset for reference times. The random dataset configuration needs to be last in config file.'
                times = np.concatenate([dataset.normalized_times for dataset in train_dict.values()])
                time_range = [np.min(times), np.max(times)]
                radius_range = u.Quantity(ds_config.pop('radius_range'), unit=ds_config.pop('unit', 'AU'))
                dataset = RandomSphericalCoordinateDataset(time_range=time_range, radius_range=radius_range, **ds_config)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_date is None:
                ref_date = dataset.ref_date
                base_config['ref_date'] = ref_date
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset
        return train_dict

    def _load_valid_dataset(self, data_config, base_config):
        data_config = copy.deepcopy(data_config)

        valid_dict = {}
        for config in data_config:
            config = copy.deepcopy(config)
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)
            if ds_type.lower() == 'hao':
                dataset = HAOThomsonDataset(**ds_config, ds_key=ds_key, test=True)
            elif ds_type.lower() == 'reference_cube':
                dataset = ReferenceCubeDataset(**ds_config, ds_key=ds_key, shuffle=False, filter_nans=False)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            assert ds_key not in valid_dict, f'Duplicate dataset key {ds_key}'
            valid_dict[ds_key] = dataset
        return valid_dict


class GenericThomsonDataset(TensorsDataset):
    def __init__(self, data_path_pB, data_path_tB, scaling, ds_key, instrument_key,
                 Rs_per_ds, seconds_per_dt, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, test=False, **kwargs):
        self.scaling = scaling
        # select files with min diff in dates
        pB_files = sorted(glob.glob(data_path_pB))
        tB_files = sorted(glob.glob(data_path_tB))

        data_config = {}
        # load reference info
        ref_map = Map(pB_files[0])
        data_config['image_shape'] = ref_map.data.shape
        data_config['wcs'] = ref_map.wcs
        data_config['wavelength'] = ref_map.wavelength
        self.data_config = data_config

        if debug:
            sampling = len(pB_files) // 20
            pB_files = pB_files[::sampling]
            tB_files = tB_files[::sampling]
        if test:
            # select file at center of the list
            idx = len(pB_files) // 2
            pB_files = pB_files[idx:idx + 1]
            tB_files = tB_files[idx:idx + 1]

        # load rays
        data_dict = {}
        with multiprocessing.Pool(os.cpu_count()) as p:
            data = [v for v in
                    tqdm(p.imap(_load_map_data, zip(tB_files, repeat(Rs_per_ds), repeat('heliographic'))), total=len(tB_files),
                         desc=f'Loading tB + rays')]
        for k in data[0].keys():
            data_dict[k] = np.stack([d[k] for d in data], axis=0)

        # load remaining images
        with multiprocessing.Pool(os.cpu_count()) as p:
            pB_image_stack = [v for v in tqdm(p.imap(fits.getdata, pB_files), total=len(pB_files), desc=f'Loading pB')]
            pB_image_stack = np.stack(pB_image_stack, axis=0)

        image_stack = np.stack([data_dict['image'], pB_image_stack], axis=-1)
        image_stack[image_stack < 0] = np.nan  # remove negative values

        data_dict['image'] = image_stack / scaling

        # expand and normalize times
        times = data_dict['time']
        ref_date = min(times) if ref_date is None else ref_date
        self.ref_date = ref_date
        self.times = times
        times = np.array([normalize_datetime(t, seconds_per_dt, ref_date) for t in times])
        self.normalized_times = times
        times_arr = np.ones((*data_dict['image'].shape[:-1], 1), dtype=np.float32) * times[:, None, None, None]
        data_dict['time'] = times_arr

        if not test:
            cmap = cm.soholasco2.copy()
            cmap.set_bad(color='green')
            log_overview(data_dict["image"], data_dict['pose'], times, cmap, seconds_per_dt, ref_date, ds_key=ds_key)
            print('----- Data Overview -----')
            print(
                f'Image shape: {data_dict["image"].shape}; MIN: {np.nanmin(data_dict["image"])}; MAX: {np.nanmax(data_dict["image"])}')
            print(f'Time shape: {times_arr.shape}; MIN: {np.nanmin(times_arr)}; MAX: {np.nanmax(times_arr)}')

        tensors = {k: v.reshape((-1, *v.shape[3:])) for k, v in data_dict.items() if k in ['image', 'rays', 'time']}

        # info for plotting
        self.image_shape = image_stack.shape[1:3]

        super().__init__(tensors=tensors, batch_size=batch_size, shuffle=not test, filter_nans=not test, instrument=instrument_key, **kwargs)


class HAOThomsonDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        # super().__init__(scaling=5e-5, **kwargs)
        super().__init__(scaling=1.0, **kwargs)


class ReferenceCubeDataset(TensorsDataset):

    def __init__(self, data_path, ref_date, seconds_per_dt, Rs_per_ds, max_radius=130, **kwargs):

        o = scipy.io.readsav(data_path)
        date0 = parse("2010-04-03T09:04:00.000") # TODO check if times actually match
        time = date0 + timedelta(hours=float(o['this_time']))
        time = normalize_datetime(time, seconds_per_dt, ref_date)

        density = o['dens'].astype(np.float32).T
        ph = o['ph1d'].astype(np.float32)
        r = o['r1d'].astype(np.float32)
        th = o['th1d'].astype(np.float32) - np.pi / 2

        # clip radius to 100 Rsun
        mask = r < max_radius
        r = r[mask]
        density = density[mask]

        radius, theta, phi, t = np.meshgrid(r, th, ph, np.array([time]), indexing="ij")
        spherical_coords = np.stack([radius, theta, phi], axis=-1)

        cartesian_coords = spherical_to_cartesian(spherical_coords)
        cartesian_coords = cartesian_coords / Rs_per_ds
        x, y, z = cartesian_coords[..., 0], cartesian_coords[..., 1], cartesian_coords[..., 2]

        query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)
        query_points = query_points[:, :, :, 0] # squeeze time dimension

        print('Query points range: ', query_points.reshape(-1, 4).min(0), query_points.reshape(-1, 4).max(0))

        self.cube_shape = query_points.shape[:-1]

        query_points = query_points.reshape(-1, 4)
        density = density.reshape(-1)
        spherical_coords = spherical_coords.reshape(-1, 3)

        tensors = {'query_points': query_points,
                   'spherical_coords': spherical_coords,
                   'rho': density}
        super().__init__(tensors, **kwargs)