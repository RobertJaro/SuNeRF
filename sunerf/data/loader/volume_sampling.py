import numpy as np
import torch
from astropy import units as u
from torch.utils.data import Dataset

from sunerf.train.coordinate_transformation import spherical_to_cartesian


class RandomSphericalCoordinateDataset(Dataset):
    """Draw random points from a spherical volume.

    By default, points are uniform per unit physical volume. In spherical
    coordinates this requires sampling uniformly in ``r**3``, ``sin(latitude)``,
    and longitude, rather than uniformly in radius and latitude. Setting
    ``radial_sampling_exponent`` overrides only the radial distribution with
    ``r = r_min + (r_max - r_min) * u**exponent``. Exponents greater than one
    bias samples toward the lower radial boundary.
    """

    def __init__(self, radius_range, batch_size, time_range, Rs_per_ds=1,
                 latitude_range=(-np.pi/2, np.pi/2) * u.rad, longitude_range=(0, 2 * np.pi) * u.rad,
                 volume_uniform_sampling=True, radial_sampling_exponent=None, **kwargs):
        self.radius_range = radius_range.to_value(u.Rsun) if isinstance(radius_range, u.Quantity) else radius_range
        self.time_range = time_range
        self.Rs_per_ds = Rs_per_ds
        self.latitude_range = (latitude_range.to_value(u.rad)
                               if isinstance(latitude_range, u.Quantity) else np.asarray(latitude_range))
        self.longitude_range = (longitude_range.to_value(u.rad)
                                if isinstance(longitude_range, u.Quantity) else np.asarray(longitude_range))
        self.batch_size = batch_size
        self.volume_uniform_sampling = volume_uniform_sampling
        if radial_sampling_exponent is not None and radial_sampling_exponent <= 0:
            raise ValueError("radial_sampling_exponent must be greater than zero")
        self.radial_sampling_exponent = radial_sampling_exponent

    def __len__(self):
        return 1

    def _sample_unit_coordinates(self):
        return torch.rand(self.batch_size, 4, dtype=torch.float32)

    def __getitem__(self, item):
        random_coords = self._sample_unit_coordinates()
        # r [1, height]
        h_r = self.radius_range
        r_min, r_max = np.min(h_r), np.max(h_r)
        if self.radial_sampling_exponent is not None:
            radial_fraction = random_coords[:, 0].pow(self.radial_sampling_exponent)
            r = r_min + radial_fraction * (r_max - r_min)
        elif self.volume_uniform_sampling:
            r = (r_min ** 3 + random_coords[:, 0] * (r_max ** 3 - r_min ** 3)).pow(1.0 / 3.0)
        else:
            r = r_min + random_coords[:, 0] * (r_max - r_min)
        # latitude [-pi/2, pi/2]
        if self.volume_uniform_sampling:
            lat_r = self.latitude_range
            v_min, v_max = np.min(np.sin(lat_r)), np.max(np.sin(lat_r))
            sin_lat = v_min + random_coords[:, 1] * (v_max - v_min)
            lat = torch.arcsin(sin_lat)
        else:
            lat_r = self.latitude_range
            lat = lat_r[0] + random_coords[:, 1] * (lat_r[1] - lat_r[0])
        # phi [0, 2pi]
        lon_r = self.longitude_range
        lon = lon_r[0] + random_coords[:, 2] * (lon_r[1] - lon_r[0])
        # convert to cartesian
        spherical_coords = torch.stack([r, lat, lon], dim=1)
        cartesian_coords = spherical_to_cartesian(spherical_coords, f=torch)
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # add time
        time_coords = self.time_range[0] + random_coords[:, 3:4] * (self.time_range[1] - self.time_range[0])
        random_coords = torch.cat([cartesian_coords, time_coords], dim=1)
        return {'coords': random_coords}
