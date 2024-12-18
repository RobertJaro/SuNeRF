from torch.utils.data import Dataset, DataLoader
from sunerf.data.mhd.psi_io import rdhdf_3d
import glob
import os
import numpy as np

from sunerf.data.loader.multi_instrument import MultiInstrumentDataModule

class PSIMHDDataModule(MultiInstrumentDataModule):
    def __init__(self, data_config, working_dir, Rs_per_ds=1, seconds_per_dt=86400,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False,
                 **kwargs):
        
        # Intialize the multi-insturment data module to retain the validation part
        super().__init__(data_config, working_dir,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt,
                         batch_size=batch_size, validation_batch_size=validation_batch_size,
                         debug=True, **kwargs)

        
        # Overwrite the training part for use in the direct training loop.
        mhd_data_config = kwargs['mhd_data_config']
        self.training_batch_size = batch_size

        train_dict = {}
        train_dict["psi"] = mhdDatasetFile(
            mhd_data_config['psi_data_path'],
            percentage_of_points = mhd_data_config['percentage_of_points'],
            seconds_per_dt = seconds_per_dt,
            r_max = mhd_data_config['r_max']
        )

        self.training_datasets = train_dict
        self.datasets = {**self.training_datasets, **self.validation_datasets}

        self.config['psi_mhd'] = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt,
                       'debug': debug, 'working_dir': working_dir, 'batch_size': batch_size,
                       'psi_data_path': mhd_data_config['psi_data_path'],
                       'percentage_of_points': mhd_data_config['percentage_of_points'],
                       'r_max':mhd_data_config['r_max']}


        # valid_dict = {}
        # valid_dict["psi"] = mhdDatasetFile(
        #     data_config['psi_data_path'],
        #     percentage_of_points = data_config['percentage_of_points'],
        #     seconds_per_dt = seconds_per_dt,
        #     r_max = data_config['r_max'],
        #     two_files=True
        # )        


class mhdDatasetFile(Dataset):

    def __init__(self, psi_data_path:str, percentage_of_points:float=0.3, seconds_per_dt:int=86400, r_max=5, static:bool=False, two_files=False):       
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
        self.percentage_of_points = percentage_of_points

        self.density_files = sorted(glob.glob(os.path.join(psi_data_path, 'rho', '*.h5')))
        self.temperature_files = sorted(glob.glob(os.path.join(psi_data_path, 't', '*.h5')))
        if two_files:
            self.density_files = self.density_files[0:2]
            self.temperature_files = self.temperature_files[0:2]

        self.ffirst = int(self.density_files[0].split('00')[1].split('.h5')[0])  # rho002531.h5
        self.flast = int(self.density_files[-1].split('00')[1].split('.h5')[0])

        self.n_files = len(self.density_files)

    def __len__(self):
        return self.n_files

    def __getitem__(self, idx):
        r, th, phi, density = rdhdf_3d(self.density_files[idx])
        density = density[:, :, r<self.r_max]
        fill_value = np.median(density[np.where(density > 0)])
        density[np.where(density < 0)] = fill_value

        _, _, _, temperature = rdhdf_3d(self.temperature_files[idx])
        temperature = temperature[:,:, r<self.r_max]
        fill_value = np.median(temperature[np.where(temperature > 0)])
        temperature[np.where(temperature < 0)] = fill_value

        n_gridpoints_in_cube = temperature.reshape(-1).shape[0]
        r = r[r<self.r_max]

        x = r[None, None, :]*np.sin(th[None, :, None])*np.cos(phi[:, None, None])
        y = r[None, None, :]*np.sin(th[None, :, None])*np.sin(phi[:, None, None])
        z = r[None, None, :]*np.cos(th[None, :, None])*(phi[:, None, None]*0+1)

        # Pick the queried points
        gridpoint_idxs = np.random.randint(low=0, high=n_gridpoints_in_cube-1, size=(int(n_gridpoints_in_cube*self.percentage_of_points)))
        x = x.reshape(-1)[gridpoint_idxs]
        y = y.reshape(-1)[gridpoint_idxs]
        z = z.reshape(-1)[gridpoint_idxs]

        # TODO Verify that we return the right time given the file index
        t = idx*60*60 / self.seconds_per_dt * (x*0+1)
        if self.static:
            t = t*0
        density = density.reshape(-1)[gridpoint_idxs]*1e8
        temperature = temperature.reshape(-1)[gridpoint_idxs]*2.807066716734894e7

        return np.stack([x, y, z, t],axis=-1), density, temperature


class mhdDatasetSinglePoints(Dataset):

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

        x = r[None, None, :]*np.sin(th[None, :, None])*np.cos(phi[:, None, None])
        y = r[None, None, :]*np.sin(th[None, :, None])*np.sin(phi[:, None, None])
        z = r[None, None, :]*np.cos(th[None, :, None])*(phi[:, None, None]*0+1)

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
