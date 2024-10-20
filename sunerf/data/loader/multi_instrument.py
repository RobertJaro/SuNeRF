import copy
import glob
import multiprocessing
import os
from itertools import repeat

import numpy as np
import torch
from astropy.io import fits
from dateutil.parser import parse
from sunpy.map import Map
from torch.utils.data import Dataset
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule, TensorsDataset, _load_map_data
from sunerf.train.callback import log_overview


class MultiInstrumentDataModule(BaseDataModule):

    def __init__(self, data_config, working_dir, Rs_per_ds=1, seconds_per_dt=86400, ref_time=None,
                 batch_size=int(2 ** 10), debug=False, **kwargs):
        os.makedirs(working_dir, exist_ok=True)
        # adjust batch size
        N_GPUS = torch.cuda.device_count()
        batch_size = int(batch_size) * N_GPUS

        data_train_config = copy.deepcopy(data_config)
        ref_time = parse(ref_time) if ref_time is not None else None  # parse ref time if specified
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_time': ref_time,
                       'debug': debug, 'working_dir': working_dir, 'batch_size': batch_size}

        train_dict = {}
        for config in data_train_config:
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            config.update(base_config)
            if ds_type == 'AIA':
                dataset = AIADataset(**config)
            elif ds_type == 'EUVI':
                dataset = EUVIDataset(**config)
            elif ds_type == 'PSI':
                dataset = PSIDataset(**config)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_time is None:
                base_config['ref_time'] = dataset.ref_time
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset

        valid_dict = {}
        data_valid_config = copy.deepcopy(data_config)
        for config in data_valid_config:
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            config.update(base_config)
            if ds_type == 'AIA':
                dataset = AIADataset(**config, test=True)
            elif ds_type == 'EUVI':
                dataset = EUVIDataset(**config, test=True)
            elif ds_type == 'PSI':
                dataset = PSIDataset(**config, test=True)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_time is None:
                ref_time = dataset.ref_time
                base_config['ref_time'] = ref_time
            assert ds_key not in valid_dict, f'Duplicate dataset key {ds_key}'
            valid_dict[ds_key] = dataset

        valid_dict['absorption'] = AbsorptionTestDataset(batch_size=batch_size)

        ref_ds = list(train_dict.values())[0]
        data_config = ref_ds.data_config
        config = {'type': 'plasma', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_time': ref_time,
                  'wcs': data_config['wcs'], 'image_shape': data_config['image_shape'], 'times': ref_ds.times,
                  'cmaps': data_config['cmaps']}
        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_time=ref_time,
                         module_config=config, **kwargs)


class GenericEUVDataset(TensorsDataset):
    def __init__(self, file_dict, working_dir, Rs_per_ds=1, seconds_per_dt=86400, ref_time=None,
                 batch_size=int(2 ** 10), debug=False, test=False, cmaps=None, scaling=1, static=True, **kwargs):
        data_config = {}
        wavelengths = sorted(list(file_dict.keys()))
        # load reference info
        ref_map = Map(file_dict[wavelengths[0]][0])
        data_config['image_shape'] = ref_map.data.shape
        data_config['wcs'] = ref_map.wcs
        data_config['wavelength'] = ref_map.wavelength
        data_config['cmaps'] = ['gray'] * len(wavelengths) if cmaps is None else cmaps
        self.data_config = data_config

        if debug:
            for k, v in file_dict.items():
                sampling = len(v) // 20
                file_dict[k] = v[::sampling]

        if test:
            for k, v in file_dict.items():
                # select file at center of the list
                file_dict[k] = [v[len(v) // 2]]

        # load rays
        data_dict = {}
        with multiprocessing.Pool(os.cpu_count()) as p:
            f = file_dict[wavelengths[0]]
            data = [v for v in
                    tqdm(p.imap(_load_map_data, zip(f, repeat(Rs_per_ds))), total=len(f),
                         desc=f'Loading {wavelengths[0]} + rays')]
        for k in data[0].keys():
            data_dict[k] = np.stack([d[k] for d in data], axis=0)

        # load images
        with multiprocessing.Pool(os.cpu_count()) as p:
            image_stack = []
            for wl in wavelengths[1:]:
                f = file_dict[wl]
                images = [v for v in tqdm(p.imap(fits.getdata, f), total=len(f), desc=f'Loading {wl}')]
                images = np.stack(images, axis=0)
                image_stack.append(images)

        image_stack = np.stack([data_dict['image'], *image_stack], axis=-1)
        image_stack[image_stack < 0] = 0  # remove negative values

        data_dict['image'] = image_stack / scaling

        # set to same time if static
        if static:
            times = data_dict['time']
            ref_time = min(times) if ref_time is None else ref_time
            data_dict['time'] = [ref_time] * len(times)

        # expand and normalize times
        times = data_dict['time']
        ref_time = min(times) if ref_time is None else ref_time
        self.ref_time = ref_time
        self.times = times
        times = np.array([normalize_datetime(t, seconds_per_dt, ref_time) for t in times])
        times_arr = np.ones((*data_dict['image'].shape[:-1], 1), dtype=np.float32) * times[:, None, None, None]
        data_dict['time'] = times_arr

        if not test:
            log_overview(data_dict["image"], data_dict['pose'], times, 'gray', seconds_per_dt, ref_time)

        tensors = {k: v.reshape((-1, *v.shape[3:])) for k, v in data_dict.items() if k in ['image', 'rays', 'time']}

        super().__init__(tensors=tensors, work_directory=working_dir, batch_size=batch_size,
                         shuffle=not test, filter_nans=not test)


class AIADataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, scaling=10000, **kwargs):
        wavelengths = [94, 131, 171, 193, 211, 304, 335] if wavelengths is None else wavelengths
        cmaps_dict = {94: 'sdoaia94', 131: 'sdoaia131', 171: 'sdoaia171', 193: 'sdoaia193',
                      211: 'sdoaia211', 304: 'sdoaia304', 335: 'sdoaia335'}
        cmaps = [cmaps_dict[wl] for wl in wavelengths]

        files = sorted(glob.glob(data_path))
        assert len(files) > 0, f'No files found in {data_path}'

        # group by wavelength
        file_dict = {wl: [] for wl in wavelengths}
        date_dict = {wl: [] for wl in wavelengths}
        for f in files:
            wl = int(fits.getheader(f)['WAVELNTH'])
            if wl not in wavelengths:
                continue
            date = parse(fits.getheader(f)['DATE-OBS'])
            file_dict[wl].append(f)
            date_dict[wl].append(date)

        # choose channel with smalest number of dates
        min_wl = min(date_dict, key=lambda k: len(date_dict[k]))
        ref_dates = date_dict[min_wl]
        # select files with min diff in dates
        for wl, f, dates in zip(file_dict.keys(), file_dict.values(), date_dict.values()):
            dates = np.array(dates)
            file_dict[wl] = [f[np.argmin(np.abs(dates - t), axis=0)] for t in ref_dates]

        super().__init__(file_dict,
                         cmaps=cmaps,
                         scaling=scaling,
                         **kwargs)


class EUVIDataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, scaling=7000, **kwargs):
        wavelengths = [171, 195, 284, 304] if wavelengths is None else wavelengths
        cmaps = {171: 'sdoaia171', 195: 'sdoaia193', 284: 'sdoaia211', 304: 'sdoaia304'}
        cmaps = [cmaps[wl] for wl in wavelengths]

        files = sorted(glob.glob(data_path))
        assert len(files) > 0, f'No files found in {data_path}'

        # group by wavelength
        file_dict = {wl: [] for wl in wavelengths}
        date_dict = {wl: [] for wl in wavelengths}
        for f in files:
            wl = int(fits.getheader(f)['WAVELNTH'])
            if wl not in wavelengths:
                continue
            date = parse(fits.getheader(f)['DATE-OBS'])
            file_dict[wl].append(f)
            date_dict[wl].append(date)

        # choose channel with smalest number of dates
        min_wl = min(date_dict, key=lambda k: len(date_dict[k]))
        ref_dates = date_dict[min_wl]
        # select files with min diff in dates
        for wl, f, dates in zip(file_dict.keys(), file_dict.values(), date_dict.values()):
            dates = np.array(dates)
            file_dict[wl] = [f[np.argmin(np.abs(dates - t), axis=0)] for t in ref_dates]

        super().__init__(file_dict,
                         cmaps=cmaps,
                         scaling=scaling,
                         **kwargs)


class PSIDataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, **kwargs):
        wavelengths = [171, 193, 211] if wavelengths is None else wavelengths

        file_dict = {wl: sorted(glob.glob(os.path.join(data_path, f'*_AIA_{wl}_*.fits'))) for wl in wavelengths}

        cmaps = {171: 'sdoaia171', 193: 'sdoaia193', 211: 'sdoaia211'}
        cmaps = [cmaps[wl] for wl in wavelengths]
        super().__init__(file_dict, cmaps=cmaps, **kwargs)


class AbsorptionTestDataset(Dataset):

    def __init__(self, n_logT=100, n_logNe=100, logT_range=(3.7, 8), logNe_range=(-3, 4), batch_size=1024):
        self.data = np.stack(np.meshgrid(np.linspace(*logT_range, n_logT),
                                         np.linspace(*logNe_range, n_logNe), indexing='ij'), -1)
        self.image_shape = (n_logT, n_logNe)
        self.data_tensor = torch.from_numpy(self.data).float().reshape(-1, 2)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(self.data_tensor.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, idx):
        data = self.data_tensor[idx * self.batch_size: (idx + 1) * self.batch_size]
        return {'log_T': data[:, 0:1], 'log_ne': data[:, 1:2]}
