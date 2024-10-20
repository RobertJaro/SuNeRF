import argparse
import os

import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--video_path', type=str)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=int(2 ** 12))
args = parser.parse_args()

chk_path = args.chk_path
video_path = args.video_path
resolution = args.resolution
batch_size = args.batch_size

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
ref_time = loader.ref_time

central_longitude = loader.ref_map.carrington_longitude.to_value(u.deg)
central_latitude = loader.ref_map.carrington_latitude.to_value(u.deg)

shift = 70
n_points = 20

points_1 = zip(np.ones(n_points) * central_latitude,
               np.linspace(central_longitude, central_longitude + shift, n_points),
               [ref_time] * n_points,
               np.ones(n_points))

points_2 = zip(np.ones(n_points) * central_latitude,
               np.linspace(central_longitude + shift, central_longitude, n_points),
               [ref_time] * n_points,
               np.ones(n_points))

points_3 = zip(np.ones(n_points) * central_latitude,
               np.linspace(central_longitude, central_longitude - shift, n_points),
               [ref_time] * n_points,
               np.ones(n_points))

points_4 = zip(np.ones(n_points) * central_latitude,
               np.linspace(central_longitude - shift, central_longitude, n_points),
               [ref_time] * n_points,
               np.ones(n_points))

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3) + list(points_4)

ne_norm = LogNorm(vmin=1)
absorption_norm = LogNorm(vmin=1, vmax=100)
img_norm = ImageNormalize(vmin=0, vmax=0.7, stretch=AsinhStretch(0.005))
T_norm = ImageNormalize(vmin=4.5, vmax=7)

# cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]
cmaps = cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, cm.sdoaia335

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time, batch_size=batch_size,
                                resolution=(resolution, resolution) * u.pix,
                                # scale=[2400 / resolution, 2400 / resolution] * u.arcsec / u.pix
                                         )
    fig, axs = plt.subplots(2, len(cmaps), figsize=(len(cmaps) * 3, 5))

    for j, cmap in enumerate(cmaps):
        im = axs[0, j].imshow(outputs['image'][..., j], cmap=cmap, norm=img_norm, origin='lower')
        divider = make_axes_locatable(axs[0, j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    im = axs[1, 0].imshow(outputs['mean_T'], cmap='plasma', origin='lower', norm=T_norm)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[1, 1].imshow(outputs['total_ne'], cmap='viridis', origin='lower', norm=ne_norm)
    divider = make_axes_locatable(axs[1, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    im = axs[1, 2].imshow(outputs['mean_absorption'], cmap='cool', origin='lower', norm=absorption_norm)
    divider = make_axes_locatable(axs[1, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    [ax.axis('off') for ax in axs[1, 2:]]

    # axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.2, origin='lower')
    # axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower')

    axs[1, 0].set_title('Mean T [log K]')
    axs[1, 1].set_title('Total N$_e$ [cm$^{-2}$]')
    axs[1, 2].set_title('Integrated Absorption')

    [ax.axis('off') for ax in axs.ravel()]
    plt.tight_layout()
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i), dpi=300)
    plt.close(fig)
