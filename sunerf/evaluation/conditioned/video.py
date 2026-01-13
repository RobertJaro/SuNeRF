import argparse
import os

import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ConditionedSuNeRFLoader

parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--ref_file', type=str)
parser.add_argument('--video_path', type=str, default=None, required=False)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=512)
args = parser.parse_args()

chk_path = args.chk_path
video_path = args.video_path if args.video_path is not None else os.path.join(os.path.dirname(chk_path), 'video')
resolution = args.resolution
resolution = (resolution, resolution) * u.pix
batch_size = args.batch_size

ref_file = args.ref_file
ref_map = Map(ref_file)
ref_map_lat = ref_map.carrington_latitude.to_value(u.deg)
ref_map_lon = ref_map.carrington_longitude.to_value(u.deg)
distance = ref_map.dsun.to_value(u.AU)

os.makedirs(video_path, exist_ok=True)

# plot params
img_norm = ImageNormalize(stretch=AsinhStretch(0.001))
cmap = cm.sdoaia193

# plot reference image
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
im = ax.imshow(ref_map.data, cmap=cmap, norm=img_norm, origin='lower')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
plt.tight_layout()
fig.savefig(os.path.join(video_path, f'ref_map.jpg'), dpi=300)
plt.close(fig)

# init loader
loader = ConditionedSuNeRFLoader(chk_path)

n_points = 10

points_1 = zip(np.ones(n_points) * ref_map_lat,
               np.linspace(ref_map_lon, ref_map_lon - 60, n_points),
               np.ones(n_points) * distance)

points_2 = zip(np.ones(n_points * 2) * ref_map_lat,
               np.linspace(ref_map_lon - 60, ref_map_lon + 60, n_points * 2),
               np.ones(n_points * 2) * distance)

points_3 = zip(np.ones(n_points) * ref_map_lat,
               np.linspace(ref_map_lon + 60, ref_map_lon, n_points),
               np.ones(n_points) * distance)

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3)

for i, (lat, lon, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_image(ref_file, lat * u.deg, lon * u.deg, distance=d * u.AU, batch_size=batch_size,
                                resolution=resolution)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    im = ax.imshow(outputs['image'][..., 0], cmap=cmap, norm=img_norm, origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    plt.tight_layout()
    fig.savefig(os.path.join(video_path, f'{i:03d}.jpg'), dpi=300)
    plt.close(fig)
