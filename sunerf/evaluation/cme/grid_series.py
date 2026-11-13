"""Read the spherical frames of ``export_density_grid``."""
import glob
import os

import numpy as np

CM_PER_RSUN = 6.957e10
# mass per electron of a plasma with 10 % helium, the convention of coronagraph CME masses
GRAMS_PER_ELECTRON = 1.97e-24


def load_series(export_path, keys=('density', 'background', 'velocity')):
    """All frames of an export: the fields ``keys`` (density, background [cm^-3], velocity [km/s]) as
    ``[time, latitude, longitude, radius]``, the axes radius [R_sun], latitude, longitude [rad], and times."""
    files = sorted(glob.glob(os.path.join(export_path, 'rho*.npz')))
    if not files:
        raise FileNotFoundError(f'No rho*.npz frames in {export_path}')
    frames = [np.load(file) for file in files]
    series = {key: np.stack([frame[key] for frame in frames]) for key in keys}
    series.update(radius=frames[0]['radius'], latitude=np.deg2rad(frames[0]['latitude']),
                  longitude=np.deg2rad(frames[0]['longitude']),
                  times=np.array([frame['time'] for frame in frames], dtype='datetime64[s]'))
    return series


def cell_volume(radius, latitude, longitude):
    """Volume of every grid cell ``[lat, lon, r]`` [cm^3]."""
    return (np.cos(latitude)[:, None, None] * np.gradient(latitude)[:, None, None]
            * np.gradient(longitude)[None, :, None] * (radius ** 2 * np.gradient(radius))[None, None, :]
            * CM_PER_RSUN ** 3)
