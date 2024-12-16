from torch.utils.data import Dataset
from sunerf.data.mhd.psi_io import rdhdf_3d
import glob
import os
import numpy as np

class mhdDataset(Dataset):

    def __init__(self, data_path:str, seconds_per_dt:int=86400, r_max=5, static:bool=False):       
        """
        Dataset to iterate over all PSI gridpoints

        Parameters
        ----------
        data_path : str
            Path containing a temperature (t) and density (rho) folders with the PSI simulation files
        seconds_per_dt : int, optional
            Normalization constant used to normalize time, by default 86400
        static : bool, optional
            Whether to set all timepoints to 0 as if there was no dynamic evolution, by default False
        """

        self.seconds_per_dt = seconds_per_dt
        self.r_max = r_max
        self.static = static

        self.density_files = sorted(glob.glob(os.path.join(data_path, 'rho', '*.h5')))
        self.temperature_files = sorted(glob.glob(os.path.join(data_path, 't', '*.h5')))
        self.ffirst = int(self.density_files[0].split('00')[1].split('.h5')[0])  # rho002531.h5
        self.flast = int(self.density_files[-1].split('00')[1].split('.h5')[0])

        ref_file = self.density_files[0]        # TODO: Keep in mind there may be a memory leak while working with PSI files
        r, _, _, data = rdhdf_3d(ref_file)
        print('r_init', r.shape, data.shape)
        data = data[:, :, r<self.r_max]
        self.n_files = len(self.density_files)
        self.n_gridpoints_per_file = data.reshape(-1).shape[0]
        self.n_gridpoints = self.n_files*self.n_gridpoints_per_file

    def __len__(self):
        return self.n_gridpoints

    def __getitem__(self, idx):
        # lazy load data
        file_idx = idx//self.n_gridpoints_per_file
        r, th, phi, density = rdhdf_3d(self.density_files[file_idx])
        density = density[:, :, r<self.r_max]
        _, _, _, temperature = rdhdf_3d(self.temperature_files[file_idx])
        temperature = temperature[:,:, r<self.r_max]
        r = r[r<self.r_max]

        print('r', np.min(r), np.max(r))
        print('th', np.min(th), np.max(th))
        print('phi', np.min(phi), np.max(phi))

        print('phi',phi.shape, 'th',th.shape, 'r', r.shape, 'density', density.shape)
        # TODO double check that psi's theta is colatitude and phi longitude 
        x = r[None, None, :]*np.sin(th[None, :, None])*np.cos(phi[:, None, None])
        y = r[None, None, :]*np.sin(th[None, :, None])*np.sin(phi[:, None, None])
        z = r[None, None, :]*np.cos(th[None, :, None])*(phi[:, None, None]*0+1)

        print(x.shape, y.shape, z.shape)

        # Pick the queried point
        gridpoint_idx = idx%self.n_gridpoints_per_file
        x = x.reshape(-1)[gridpoint_idx]
        y = y.reshape(-1)[gridpoint_idx]
        z = z.reshape(-1)[gridpoint_idx]
        # TODO Verify that we return the right time given the file index
        t = file_idx*60*60 / self.seconds_per_dt
        if self.static:
            t = 0
        density = density.reshape(-1)[gridpoint_idx]*1e8
        temperature = temperature.reshape(-1)[gridpoint_idx]*2.807066716734894e7

        return x, y, z, t, density, temperature


