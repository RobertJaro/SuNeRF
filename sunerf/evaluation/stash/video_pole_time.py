import datetime
import os
import shutil

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader

chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'
video_path = '/mnt/results/video_pole_time'

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=2048)
cmap = sdo_cmaps[loader.wavelength]

au = (1 * u.AU).to(u.solRad).value
n_points = 1000

points_1 = zip(-np.ones(n_points) * 90,
               np.ones(n_points) * 0,
               pd.date_range(datetime.datetime(2012, 8, 24), datetime.datetime(2012, 9, 7), n_points),
               np.ones(n_points) * 1 * au)

# combine coordinates
points = list(points_1)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * 1, strides=8)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(outputs['channel_map'], cmap=cmap, vmin=0, vmax=1, origin='lower')
    axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.2, origin='lower')
    axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower')
    axs[0].set_axis_off(), axs[1].set_axis_off(), axs[2].set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i))
    plt.close(fig)

shutil.make_archive(video_path, 'zip', video_path)
