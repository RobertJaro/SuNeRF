import datetime
import glob
import os
import shutil
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy import units as u
from iti.data.editor import AIAPrepEditor
from matplotlib import pyplot as plt
from pandas import Timestamp
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps, sdo_norms
from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.train.coordinate_transformation import spherical_to_cartesian

chk_path = '/mnt/nerf-data/eruption/save_state.snf'
video_path = '/mnt/results/video_304_eruption_v2'

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=1024)
cmap = sdo_cmaps[loader.wavelength]

prep_editor = AIAPrepEditor(calibration='auto')


def _load(f):
    return prep_editor.call(Map(f))


with Pool(os.cpu_count()) as p:
    aia_maps = p.map(_load, glob.glob('/mnt/nerf-data/sdo_2012_08/1m_304/*.fits'))

aia_maps = np.array([s_map for s_map in aia_maps if s_map.meta['QUALITY'] == 0])
aia_dates = np.array([s_map.date.datetime for s_map in aia_maps])

au = (1 * u.AU).to(u.solRad).value
n_points = 50
points_1 = zip(np.ones(n_points) * 0,
               np.ones(n_points) * 220,
               pd.date_range(datetime.datetime(2012, 8, 31, 19), datetime.datetime(2012, 8, 31, 21), n_points),
               np.ones(n_points) * 1.1 * au)

# points_2 = zip(np.linspace(0, 10, n_points),
#                np.linspace(220, 260, n_points),
#                [datetime.datetime(2012, 8, 31, 19)] * n_points,
#                np.ones(n_points) * 0.7 * au)
#
# points_3 = zip(np.ones(n_points) * 10,
#                np.ones(n_points) * 260,
#                pd.date_range(datetime.datetime(2012, 8, 31, 19), datetime.datetime(2012, 8, 31, 19, 45), n_points),
#                np.ones(n_points) * 0.7 * au)
#
# points_4 = zip(np.linspace(10, -10, n_points),
#                np.linspace(260, 300, n_points),
#                [datetime.datetime(2012, 8, 31, 19, 45)] * n_points,
#                np.ones(n_points) * 0.7 * au)
#
# points_5 = zip(np.ones(n_points) * -10,
#                np.ones(n_points) * 300,
#                pd.date_range(datetime.datetime(2012, 8, 31, 19, 45), datetime.datetime(2012, 8, 31, 21), n_points),
#                np.ones(n_points) * 0.7 * au)
#
# points_6 = zip(np.ones(n_points) * -10,
#                np.linspace(300, 220, n_points),
#                [datetime.datetime(2012, 8, 31, 21)] * n_points,
#                np.linspace(0.7 * au, 1. * au, n_points))
#
# points_7 = zip(np.ones(n_points) * -10,
#                np.ones(n_points) * 220,
#                pd.date_range(datetime.datetime(2012, 8, 31, 21), datetime.datetime(2012, 8, 31, 22), n_points),
#                np.ones(n_points) * 1. * au)

# combine coordinates
points = list(
    points_1)  # + list(points_2) + list(points_3) + list(points_4) + list(points_5)  + list(points_6) + list(points_7)

# find center point on sphere
center = spherical_to_cartesian(1, 18 * np.pi / 180, 190 * np.pi / 180)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    time = time.to_pydatetime() if isinstance(time, Timestamp) else time
    aia_map = aia_maps[np.argmin(np.abs(aia_dates - time))]
    aia_img = sdo_norms[int(aia_map.wavelength.value)](aia_map.data)

    outputs = loader.load_observer_image(lat, lon, time, distance=d, center=center, batch_size=4096, strides=1)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(outputs['channel_map'], cmap=cmap, vmin=0, vmax=1, origin='lower')
    axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.6, origin='lower')
    axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower', vmin=0, vmax=5)
    axs[3].imshow(aia_img, cmap=cmap, vmin=0, vmax=1, origin='lower')
    axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off(), axs[3].set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i))
    plt.close(fig)

# make video
# for i, time in tqdm(enumerate(pd.date_range(datetime(2012, 8, 31, 18), datetime(2012, 8, 31, 22), 100))):
#     outputs = loader.load_observer_image(sdo_map.carrington_latitude.value, -sdo_map.carrington_longitude.value, time.to_pydatetime(), distance=0.8 * au, batch_size=4096, strides=4, center=center)
#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     axs[0].imshow(outputs['channel_map'], cmap=cmap, vmin=0, vmax=1, origin='lower')
#     axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.6, origin='lower')
#     axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower', vmin=0, vmax=5)
#     axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off(), axs[3].set_axis_off()
#     plt.tight_layout(pad=0)
#     fig.savefig(os.path.join(video_path, '%03d.jpg' % i))
#     plt.close(fig)

shutil.make_archive(video_path, 'zip', video_path)
