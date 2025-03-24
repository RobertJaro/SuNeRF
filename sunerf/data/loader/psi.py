import glob
import os.path
from datetime import timedelta, datetime
from multiprocessing import Pool

import numpy as np

from sunerf.data.loader.base_loader import TensorsDataset
from sunerf.data.psi.read_psi import read_PSI


class PSICubeDataset(TensorsDataset):

    def __init__(self, data_path, Rs_per_ds, seconds_per_dt, sampling=8, ne_scaling = (1e-21 * 10000) ** 0.5, **kwargs):
        rho_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))[:sampling]
        t_files = sorted(glob.glob(os.path.join(data_path, 't', '*.h5')))[:sampling]

        with Pool(16) as p:
            data = p.starmap(read_PSI, zip(rho_files, t_files))

        log_rho = np.log10(np.array([d['rho'] * ne_scaling for d in data]))
        log_T = np.log10(np.array([d['T'] for d in data]))
        cartesian_coords = np.array([d['cartesian_coordinates'] for d in data]) / Rs_per_ds
        times = np.array([d['time'] for d in data]) * 3600 / seconds_per_dt
        del data # free memory

        # concatenate time as last dimension of coords
        coords = np.zeros((*cartesian_coords.shape[:-1], 4), dtype=np.float32)
        coords[..., :3] = cartesian_coords
        coords[..., 3] = times[:, None, None, None]

        print(f'RHO range: {log_rho.min(), log_rho.max()}')
        print(f'T range: {log_T.min(), log_T.max()}')
        print(f'Coords range x: {coords[..., 0].min(), coords[..., 0].max()}')
        print(f'Coords range y: {coords[..., 1].min(), coords[..., 1].max()}')
        print(f'Coords range z: {coords[..., 2].min(), coords[..., 2].max()}')
        print(f'Coords range t: {coords[..., 3].min(), coords[..., 3].max()}')
        radius = np.linalg.norm(coords[..., :3], axis=-1)
        print(f'Coords range r: {radius.min(), radius.max()}')

        log_rho = log_rho.reshape((-1, 1))
        log_T = log_T.reshape((-1, 1))
        coords = coords.reshape((-1, 4))

        nan_mask = np.isnan(log_rho).any(-1) | np.isnan(log_T).any(-1)
        log_rho = log_rho[~nan_mask]
        log_T = log_T[~nan_mask]
        coords = coords[~nan_mask]

        tensors = {'log_rho': log_rho, 'log_T': log_T, 'coords': coords}
        super().__init__(tensors, **kwargs)

        self.ref_date = datetime(2025, 1, 1)
        self.times = [self.ref_date + timedelta(seconds=float(t)) for t in times]