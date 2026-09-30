"""Ground-truth density series (HAO IDL cubes, PSI/MAS HDF cubes) on a common spherical grid."""
import glob
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import scipy.io

from sunerf.data.psi.density_cube import MAS_RHO_TO_ELECTRON_CM3, dump_index, read_psi_density
from sunerf.data.psi.dump_times import load_dump_times
from sunerf.evaluation.cme.export_layout import coordinates_file, density_file
from sunerf.train.coordinate_transformation import spherical_to_cartesian

# Relaxed pre-eruption state of the 2021-10-28 PSI run followed by 5 min dumps.
PSI_FIRST_DUMP_TIME = datetime(2021, 10, 28, 15, 30)
PSI_DUMP_CADENCE = timedelta(minutes=5)


def longitude_mask(longitudes_deg, min_lon, max_lon):
    """Mask of longitudes inside ``[min_lon, max_lon]`` (degrees) with wrap-around."""
    lon = np.asarray(longitudes_deg) % 360
    min_lon, max_lon = min_lon % 360, max_lon % 360
    if min_lon <= max_lon:
        return (lon >= min_lon) & (lon <= max_lon)
    return (lon >= min_lon) | (lon <= max_lon)


@dataclass
class DensityFrame:
    """Density ``rho[r, lat, lon]`` in cm^-3 on a spherical grid (R_sun, rad, rad; Carrington frame)."""
    rho: np.ndarray
    radius: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    time: datetime

    def crop(self, min_radius=None, max_radius=None, min_latitude=None, max_latitude=None,
             min_longitude=None, max_longitude=None, stride=1):
        """Return a copy restricted to a radial / angular sector (degrees) with an angular stride."""
        mask_r = np.ones_like(self.radius, dtype=bool)
        if min_radius is not None:
            mask_r &= self.radius >= min_radius
        if max_radius is not None:
            mask_r &= self.radius <= max_radius
        mask_lat = np.ones_like(self.latitude, dtype=bool)
        if min_latitude is not None:
            mask_lat &= self.latitude >= np.deg2rad(min_latitude)
        if max_latitude is not None:
            mask_lat &= self.latitude <= np.deg2rad(max_latitude)
        mask_lon = np.ones_like(self.longitude, dtype=bool)
        if min_longitude is not None and max_longitude is not None:
            mask_lon &= longitude_mask(np.rad2deg(self.longitude), min_longitude, max_longitude)
        lat_idx = np.nonzero(mask_lat)[0][::stride]
        lon_idx = np.nonzero(mask_lon)[0][::stride]
        rho = self.rho[mask_r][:, lat_idx][:, :, lon_idx]
        return DensityFrame(rho, self.radius[mask_r], self.latitude[lat_idx], self.longitude[lon_idx], self.time)

    def query_points(self, Rs_per_ds, normalized_time):
        """Query points ``[r, lat, lon, 4]`` (Cartesian model units + normalized time) in the frame of the grid.

        Rotate the longitudes into the reference frame of the model first, see
        :func:`sunerf.evaluation.cme.export_density.model_query_points`.
        """
        radius, latitude, longitude = np.meshgrid(self.radius, self.latitude, self.longitude, indexing='ij')
        cartesian = spherical_to_cartesian(np.stack([radius, latitude, longitude], -1)) / Rs_per_ds
        t = np.full(cartesian.shape[:-1] + (1,), normalized_time, dtype=np.float32)
        return np.concatenate([cartesian.astype(np.float32), t], -1)


def sort_axes(frame):
    """Return the frame with ascending radius, latitude and longitude axes."""
    r_order = np.argsort(frame.radius)
    lat_order = np.argsort(frame.latitude)
    lon_order = np.argsort(frame.longitude)
    rho = frame.rho[r_order][:, lat_order][:, :, lon_order]
    return DensityFrame(rho, frame.radius[r_order], frame.latitude[lat_order], frame.longitude[lon_order],
                        frame.time)


def load_hao_frame(file, date0):
    """HAO IDL cube: ``dens[ph, th, r]`` in cm^-3, ``this_time`` in hours after ``date0``."""
    o = scipy.io.readsav(file)
    time = date0 + timedelta(hours=float(o['this_time']))
    rho = o['dens'].astype(np.float32).T  # (r, th, ph)
    radius = o['r1d'].astype(np.float64)
    latitude = o['th1d'].astype(np.float64) - np.pi / 2
    longitude = o['ph1d'].astype(np.float64) % (2 * np.pi)
    return sort_axes(DensityFrame(rho, radius, latitude, longitude, time))


def psi_dump_time(dump, dump_times=None):
    if dump_times is not None:
        return dump_times[dump].to_datetime()
    if dump == 1:
        return PSI_FIRST_DUMP_TIME
    return PSI_FIRST_DUMP_TIME + (dump - 1) * PSI_DUMP_CADENCE


def load_psi_frame(file, dump_times=None):
    """PSI/MAS cube: ``rho[ph, th, r]`` in code units (1e8 cm^-3), colatitude ``th``."""
    density, radius, theta, phi = read_psi_density(file)
    # drop the MAS ghost cells beyond the poles and the periodic seam
    inside_phi, inside_theta = (phi >= 0) & (phi < 2 * np.pi), (theta >= 0) & (theta <= np.pi)
    density, theta, phi = density[inside_phi][:, inside_theta], theta[inside_theta], phi[inside_phi]
    rho = (density * MAS_RHO_TO_ELECTRON_CM3).astype(np.float32).transpose(2, 1, 0)  # (r, th, ph)
    latitude = np.pi / 2 - theta
    longitude = phi % (2 * np.pi)
    time = psi_dump_time(dump_index(file), dump_times)
    return sort_axes(DensityFrame(rho, radius, latitude, longitude, time))


def load_export_frame(export_path, name):
    """Density of a SuNeRF export (``export_density``) on the grid of the PSI cube ``name``."""
    coordinates = np.load(coordinates_file(export_path, name))
    time = coordinates['time'].astype('datetime64[s]').item()
    return DensityFrame(np.load(density_file(export_path, name)), coordinates['radius'], coordinates['latitude'],
                        coordinates['longitude'], time)


def load_export_series(export_path, frames=None):
    """Yield the :class:`DensityFrame` objects of a SuNeRF export in the order of the dumps."""
    for file in list_series_files(density_file(export_path, '*'), frames):
        yield load_export_frame(export_path, Path(file).stem)


def list_series_files(pattern, frames=None):
    """Sorted files matching ``pattern`` restricted by a ``start:stop:step`` slice string."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No density cubes match {pattern}')
    if frames:
        parts = [int(p) if p else None for p in frames.split(':')]
        files = files[slice(*parts)]
    return files


def load_series(pattern, dataset, frames=None, date0=None, dump_times_path=None):
    """Yield :class:`DensityFrame` objects of a HAO (``dataset='hao'``) or PSI (``'psi'``) series."""
    files = list_series_files(pattern, frames)
    if dataset == 'hao':
        if date0 is None:
            raise ValueError('HAO cubes need --date0 (start of the simulation).')
        for file in files:
            yield load_hao_frame(file, date0)
    elif dataset == 'psi':
        dump_times = load_dump_times(dump_times_path) if dump_times_path else None
        for file in files:
            yield load_psi_frame(file, dump_times)
    else:
        raise ValueError(f'Unknown dataset {dataset}; use hao or psi.')


def synthetic_grid(times, min_radius, max_radius, n_radius, min_latitude, max_latitude, min_longitude,
                   max_longitude, angular_step):
    """Empty frames (``rho`` NaN) on a regular grid for model-only evaluation."""
    radius = np.linspace(min_radius, max_radius, n_radius)
    latitude = np.deg2rad(np.arange(min_latitude, max_latitude + 1e-6, angular_step))
    if max_longitude < min_longitude:
        max_longitude += 360
    longitude = np.deg2rad(np.arange(min_longitude, max_longitude + 1e-6, angular_step)) % (2 * np.pi)
    for time in times:
        rho = np.full((radius.size, latitude.size, longitude.size), np.nan, dtype=np.float32)
        yield DensityFrame(rho, radius, latitude, longitude, time)

