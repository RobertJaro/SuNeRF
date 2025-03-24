import numpy as np
import torch
from torch.utils.data import Dataset

from sunerf.train.coordinate_transformation import spherical_to_cartesian
from astropy import units as u

class RandomSphericalCoordinateDataset(Dataset):

    def __init__(self, radius_range, batch_size, time_range, Rs_per_ds=1,
                 latitude_range=(-np.pi/2, np.pi/2) * u.rad, longitude_range=(0, 2 * np.pi) * u.rad,
                 radial_weighted_sampling=False, latitude_weighted_sampling=False, **kwargs):
        self.radius_range = radius_range.to_value(u.Rsun)
        self.time_range = time_range
        self.Rs_per_ds = Rs_per_ds
        self.latitude_range = latitude_range.to_value(u.rad)
        self.longitude_range = longitude_range.to_value(u.rad)
        self.batch_size = batch_size
        self.float_tensor = torch.FloatTensor
        self.radial_weighted_sampling = radial_weighted_sampling
        self.latitude_weighted_sampling = latitude_weighted_sampling

    def __len__(self):
        return 1

    def __getitem__(self, item):
        random_coords = self.float_tensor(self.batch_size, 4).uniform_()
        # r [1, height]
        h_r = self.radius_range
        if self.radial_weighted_sampling:
            v_min, v_max = np.min(np.log(h_r)), np.max(np.log(h_r))
            random_coords[:, 0] = v_min + random_coords[:, 0] * (v_max - v_min)
            random_coords[:, 0] = torch.exp(random_coords[:, 0])
        else:
            random_coords[:, 0] = h_r[0] + random_coords[:, 0] * (h_r[1] - h_r[0])
        # theta [0, pi]
        if self.latitude_weighted_sampling:
            lat_r = self.latitude_range
            v_min, v_max = np.min(np.cos(lat_r)), np.max(np.cos(lat_r))
            random_coords[:, 1] = v_min + random_coords[:, 1] * (v_max - v_min)
            random_coords[:, 1] = torch.arccos(random_coords[:, 1])
        else:
            lat_r = self.latitude_range
            random_coords[:, 1] = lat_r[0] + random_coords[:, 1] * (lat_r[1] - lat_r[0])
        # phi [0, 2pi]
        lon_r = self.longitude_range
        random_coords[:, 2] = lon_r[0] + random_coords[:, 2] * (lon_r[1] - lon_r[0])
        # convert to cartesian
        cartesian_coords = spherical_to_cartesian(random_coords[:, :3], f=torch)
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # add time
        time_coords = self.time_range[0] + random_coords[:, 3:4] * (self.time_range[1] - self.time_range[0])
        random_coords = torch.cat([cartesian_coords, time_coords], dim=1)
        return {'coords': random_coords}