import os
import shutil
from datetime import datetime

import matplotlib.axes as maxes
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps, sdo_norms
from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian

chk_path = '/mnt/nerf-data/eruption/save_state.snf'
result_path = '/mnt/results/evaluation_eruption'

sdo_map = Map('/mnt/nerf-data/sdo_2012_08/1m_304/aia.lev1_euv_12s.2012-08-31T193009Z.304.image_lev1.fits')
stereo_map = Map('/mnt/nerf-data/stereo_2012_08_full_prep_converted_fov/304/2012-08-31T19:32.fits')

au = (1 * u.AU).to(u.solRad).value
distance = 0.8 * au

os.makedirs(result_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=1024)
cmap = sdo_cmaps[loader.wavelength]

# find center point on sphere
center = spherical_to_cartesian(1, 18 * np.pi / 180, 190 * np.pi / 180)


def _plot(time, outputs):
    # channel map
    header = _create_header(time)
    s_map = Map(outputs['channel_map'], header)
    fig = plt.figure(figsize=(4 * 3, 5))
    ax = plt.subplot(131, projection=s_map)
    ax.set_title(f'{time.isoformat(" ", timespec="minutes")}', fontsize='x-large')
    # ax.set_xlabel('Helioprojective Longitude [arcsec]', fontsize='medium')
    # ax.set_ylabel('Helioprojective Latitude [arcsec]', fontsize='medium')
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')

    sc = ax.imshow(sdo_norms[304].inverse(s_map.data),
                   norm=ImageNormalize(vmin=0, vmax=5000, stretch=AsinhStretch(0.005), clip=True),
                   cmap=cmap, origin='lower')
    ax.coords[0].set_ticks(spacing=400 * u.arcsec), ax.coords[1].set_ticks(spacing=400 * u.arcsec)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=maxes.Axes)
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.set_ylabel('Counts [DN/s]', rotation=270, labelpad=15)
    cb.ax.set_yticks([1000, 2000, 5000], ['1e3', '2e3', '5e3'])
    # height map
    s_map = Map(outputs['height_map'], header)
    ax = plt.subplot(132, projection=s_map)
    # ax.set_xlabel('Helioprojective Longitude [arcsec]', fontsize='large')
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    sc = ax.imshow(s_map.data, cmap='plasma', vmin=1, vmax=1.4, origin='lower')
    ax.coords[0].set_ticks(spacing=400 * u.arcsec), ax.coords[1].set_ticks(spacing=400 * u.arcsec)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=maxes.Axes)
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.set_ylabel('Height [solar radii]', rotation=270, labelpad=15)
    # absorption map
    s_map = Map(outputs['absorption_map'], header)
    ax = plt.subplot(133, projection=s_map)
    # ax.set_xlabel('Helioprojective Longitude [arcsec]', fontsize='large')
    ax.set_xlabel(' ')
    ax.set_ylabel(' ')
    sc = ax.imshow(s_map.data, cmap='viridis', vmin=1, vmax=6, origin='lower')
    ax.coords[0].set_ticks(spacing=400 * u.arcsec), ax.coords[1].set_ticks(spacing=400 * u.arcsec)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=maxes.Axes)
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.set_ylabel('Absorption', rotation=270, labelpad=15)
    fig.tight_layout(pad=8, w_pad=5)
    fig.savefig(os.path.join(result_path, f'{time.isoformat(timespec="minutes")}.jpg'), dpi=200, transparent=True)
    plt.close(fig)


def _create_header(time):
    observer = SkyCoord(sdo_map.heliographic_longitude, sdo_map.heliographic_latitude, distance * u.solRad,
                        obstime=time, frame='heliographic_stonyhurst')
    out_ref_coord = SkyCoord(-(512 * 1.2 + 50) * u.arcsec, -(512 * 1.2 - 100) * u.arcsec, frame=frames.Helioprojective,
                             observer=observer, obstime=time)
    out_header = make_fitswcs_header((1024, 1024), out_ref_coord,
                                     scale=u.Quantity((1.2, 1.2) * u.arcsec / u.pix),
                                     instrument='AIA', wavelength=loader.wavelength * u.AA)
    return out_header


# time = datetime(2012, 8, 31, 19, 0)
# outputs = loader.load_observer_image(sdo_map.carrington_latitude.value, -sdo_map.carrington_longitude.value, time, distance=distance, batch_size=4096, strides=1, center=center)
# _plot(time, outputs)

# time = datetime(2012, 8, 31, 19, 50)
# outputs = loader.load_observer_image(sdo_map.carrington_latitude.value, -sdo_map.carrington_longitude.value, time, distance=distance, batch_size=4096, strides=1, center=center)
# _plot(time, outputs)
#
# time = datetime(2012, 8, 31, 21, 0)
# outputs = loader.load_observer_image(sdo_map.carrington_latitude.value, -sdo_map.carrington_longitude.value, time, distance=distance, batch_size=4096, strides=1, center=center)
# _plot(time, outputs)

result_path = '/mnt/results/evaluation_eruption/video_2'
os.makedirs(result_path, exist_ok=True)
for time in tqdm(pd.date_range(datetime(2012, 8, 31, 19), datetime(2012, 8, 31, 23), 100)):
    outputs = loader.load_observer_image(sdo_map.carrington_latitude.value, -sdo_map.carrington_longitude.value,
                                         time.to_pydatetime(),
                                         distance=distance, batch_size=4096, strides=2, center=center)
    _plot(time, outputs)

shutil.make_archive(result_path, 'zip', result_path)
