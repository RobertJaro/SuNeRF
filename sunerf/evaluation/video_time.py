import os

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader

chk_path = '/mnt/results/2012_08_31_v1/save_state.snf'
video_path = '/mnt/results/video_304_eruption'

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path, resolution=3000)
cmap = sdo_cmaps[loader.wavelength]
subframe = loader.config['Subframe']

au = (1 * u.AU).to(u.solRad).value
points_1 = zip(np.ones(20) * subframe['hgc_lat'],
               np.ones(20) * subframe['hgc_lon'],
               pd.date_range(loader.start_time, loader.end_time, 20),
               np.ones(20) * au)

# combine coordinates
points = list(points_1)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat, lon, time, distance=d, batch_size=4096 * 2, strides=8)
    img = outputs['channel_map']
    plt.imsave(os.path.join(video_path, '%03d.jpg' % i), img, cmap=cmap, vmin=0, vmax=1)
