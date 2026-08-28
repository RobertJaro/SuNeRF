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

    ``radial_strata`` provides deterministic coverage of several radial shells
    while remaining uniform per unit volume *within* each shell. It is a
    mapping with ``edges`` and one ``fractions`` entry per adjacent edge pair,
    for example ``{'edges': [1.5, 10, 30, 110], 'fractions': [.5, .25, .25]}``.
    The largest-remainder allocation makes the shell counts sum exactly to the
    requested batch size.
    """

    def __init__(self, radius_range, batch_size, time_range, Rs_per_ds=1,
                 latitude_range=(-np.pi/2, np.pi/2) * u.rad, longitude_range=(0, 2 * np.pi) * u.rad,
                 volume_uniform_sampling=True, radial_sampling_exponent=None,
                 radial_strata=None, **kwargs):
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
        self.radial_strata = self._validate_radial_strata(radial_strata)

    def _validate_radial_strata(self, radial_strata):
        if radial_strata is None:
            return None
        if self.radial_sampling_exponent is not None:
            raise ValueError("radial_strata and radial_sampling_exponent are mutually exclusive")
        if not self.volume_uniform_sampling:
            raise ValueError("radial_strata requires volume_uniform_sampling=True")
        if not isinstance(radial_strata, dict) or set(radial_strata) != {'edges', 'fractions'}:
            raise ValueError("radial_strata must contain exactly 'edges' and 'fractions'")

        edges = np.asarray(radial_strata['edges'], dtype=np.float64)
        fractions = np.asarray(radial_strata['fractions'], dtype=np.float64)
        if edges.ndim != 1 or fractions.ndim != 1 or edges.size != fractions.size + 1:
            raise ValueError("radial_strata requires one fraction per adjacent pair of edges")
        if not np.isfinite(edges).all() or not np.isfinite(fractions).all():
            raise ValueError("radial_strata edges and fractions must be finite")
        if np.any(np.diff(edges) <= 0):
            raise ValueError("radial_strata edges must be strictly increasing")
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0):
            raise ValueError("radial_strata fractions must be non-negative and sum to one")

        r_min, r_max = np.min(self.radius_range), np.max(self.radius_range)
        if not np.isclose(edges[0], r_min) or not np.isclose(edges[-1], r_max):
            raise ValueError("radial_strata edges must span the configured radius_range")
        return {'edges': edges, 'fractions': fractions}

    def __len__(self):
        return 1

    def _sample_unit_coordinates(self):
        return torch.rand(self.batch_size, 4, dtype=torch.float32)

    def _stratum_counts(self):
        """Allocate an exact integer sample count to every radial shell."""
        expected = self.radial_strata['fractions'] * self.batch_size
        counts = np.floor(expected).astype(np.int64)
        remainder = self.batch_size - int(counts.sum())
        if remainder:
            fractional_parts = expected - counts
            # Stable sorting makes equal remainders deterministic.
            add_to = np.argsort(-fractional_parts, kind='stable')[:remainder]
            counts[add_to] += 1
        return counts

    def _sample_stratified_radius(self, unit_radius):
        edges = self.radial_strata['edges']
        counts = self._stratum_counts()
        radius = torch.empty_like(unit_radius)
        start = 0
        for count, lower, upper in zip(counts, edges[:-1], edges[1:]):
            end = start + int(count)
            u_radius = unit_radius[start:end]
            radius[start:end] = (
                lower ** 3 + u_radius * (upper ** 3 - lower ** 3)
            ).pow(1.0 / 3.0)
            start = end
        return radius

    def __getitem__(self, item):
        random_coords = self._sample_unit_coordinates()
        # r [1, height]
        h_r = self.radius_range
        r_min, r_max = np.min(h_r), np.max(h_r)
        if self.radial_strata is not None:
            r = self._sample_stratified_radius(random_coords[:, 0])
        elif self.radial_sampling_exponent is not None:
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
