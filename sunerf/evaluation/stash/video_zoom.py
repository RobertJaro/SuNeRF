import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader

chk_path = '/mnt/results/sunerf_v2_checkpoints/fov_211.snf'
video_path = '/mnt/results/video_211_fov'

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
cmap = sdo_cmaps[loader.wavelength]

au = (1 * u.AU).to(u.solRad).value
n_points = 50
points_1 = zip(np.linspace(-45, -60, n_points), np.linspace(-20, -40, n_points),
               pd.date_range(datetime(2012, 11, 10), datetime(2012, 11, 13), n_points),
               np.ones(n_points) * au)

# combine coordinates
points = list(points_1)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * 2, strides=2)
    img = outputs['channel_map']
    plt.imsave(os.path.join(video_path, '%03d.jpg' % i), img, cmap=cmap, vmin=0, vmax=1)
