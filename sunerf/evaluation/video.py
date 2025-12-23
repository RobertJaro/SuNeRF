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
parser.add_argument('--batch_size', type=int, default=512)
args = parser.parse_args()

chk_path = args.chk_path
video_path = args.video_path
resolution = args.resolution
resolution = (resolution, resolution) * u.pix
batch_size = args.batch_size

os.makedirs(video_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
# loader.rendering.rendering_modules['PSI'].absorption_model.coefficient = 0.5
avg_time = loader.start_time() + (loader.end_time() - loader.start_time()) / 2

n_points = 20

points_1 = zip(np.ones(n_points) * 0,
               np.linspace(0, 360, n_points),
               [avg_time] * n_points,
               np.ones(n_points))

points_2 = zip(np.linspace(0, 360, n_points),
               np.ones(n_points) * 0,
               [avg_time] * n_points,
               np.ones(n_points))

points_3 = zip(np.linspace(0, 45, n_points),
               np.linspace(0, 90, n_points),
               [avg_time] * n_points,
               np.linspace(1, 0.7, n_points), )

points_4 = zip(np.linspace(45, 45, n_points),
               np.linspace(90, 360, n_points),
               [avg_time] * n_points,
               np.linspace(0.7, 1.0, n_points), )

# combine coordinates
points = list(points_1) + list(points_2) + list(points_3) + list(points_4)

ne_norm = LogNorm(vmin=1)
absorption_norm = LogNorm()#LogNorm(vmin=1, vmax=100)

# cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]
cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, cm.sdoaia335]
# cmaps = [cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia335]
img_norms = [ImageNormalize(stretch=AsinhStretch(0.001)) for _ in cmaps]

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_image(lat * u.deg, lon * u.deg, time, distance=d * u.AU, batch_size=batch_size,
                                resolution=resolution)
    fig, axs = plt.subplots(2, len(cmaps), figsize=(len(cmaps) * 3, 5))

    for j, cmap in enumerate(cmaps):
        im = axs[0, j].imshow(outputs['image'][..., j], cmap=cmap, norm=img_norms[j], origin='lower')
        divider = make_axes_locatable(axs[0, j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    im = axs[1, 0].imshow(outputs['mean_T'], cmap='plasma', origin='lower', vmin=4, vmax=7)
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
