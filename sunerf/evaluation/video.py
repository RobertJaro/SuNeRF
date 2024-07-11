import argparse
import os

import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--video_path', type=str)
parser.add_argument('--resolution', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=4096)
args = parser.parse_args()

chk_path = args.chk_path
video_path = args.video_path
resolution = args.resolution
resolution = (resolution, resolution) * u.pix
batch_size = args.batch_size

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
cmap = sdo_cmaps[loader.wavelength.to_value(u.angstrom)]
avg_time = loader.start_time + (loader.end_time - loader.start_time) / 2

n_points = 20

points_1 = zip(np.ones(n_points) * 0,
               np.linspace(0, 360, n_points),
               [avg_time] * n_points,
               np.ones(n_points))

points_2 = zip(np.linspace(0, 360, n_points),
               np.ones(n_points) * 0,
               [avg_time] * n_points,
               np.ones(n_points))

points_3 = zip(np.linspace(0, 180, n_points),
               np.linspace(0, 360, n_points),
               [avg_time] * n_points,
               np.linspace(1, 0.2, n_points), )

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3)

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time, distance=d * u.AU, batch_size=batch_size, resolution=resolution)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(outputs['image'], cmap=cmap, norm=ImageNormalize(vmin=0, vmax=1, stretch=AsinhStretch(0.005)), origin='lower')
    # axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.2, origin='lower')
    # axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower')
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i), dpi=300)
    plt.close(fig)
