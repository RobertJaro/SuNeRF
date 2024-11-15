import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst
from sunpy.visualization.colormaps import cm

from sunerf.evaluation.loader import SuNeRFLoader

parser = argparse.ArgumentParser('Create video of ecliptic and polar views')
parser.add_argument('--chk_path', type=str)
parser.add_argument('--out_path', type=str)
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=int(2 ** 12))
args = parser.parse_args()

chk_path = args.chk_path
out_path = args.out_path
resolution = args.resolution
batch_size = args.batch_size

img_path = os.path.join(out_path, 'images')

os.makedirs(out_path, exist_ok=True)
os.makedirs(img_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(chk_path)
start_time = loader.start_time('AIA_FD')
end_time = loader.end_time('AIA_FD')

# sampling settings
planet_list = ['mars', 'earth', 'venus', 'mercury']
n_points = 5
distance = 1  # AU
log_T = np.linspace(4, 9, 101)

# plot settings
cmaps = cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, cm.sdoaia335
T_bins = list(reversed([(4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0)]))
ne_norm = LogNorm(vmin=1)
absorption_norm = LogNorm(vmin=1, vmax=100)
img_norm = [ImageNormalize(vmin=0, stretch=AsinhStretch(0.005)) for _ in cmaps]
T_norm = ImageNormalize(vmin=4.5, vmax=7)
em_norms = [LogNorm(vmin=10) for _ in T_bins]

start = datetime(start_time.year, start_time.month, start_time.day, 12)
days = (end_time - start_time).days
times = [start + timedelta(days=i) for i in range(days)]

for planet in planet_list:
    print(f'Loading {planet}...')
    for time in times:
        print(f'Processing {planet} at {time}...')
        carrington_frame = frames.HeliographicCarrington(observer=planet, obstime=time)
        planet_coord = get_body_heliographic_stonyhurst(planet, time=time).transform_to(carrington_frame)

        outputs = loader.load_observer_image(planet_coord.lat, planet_coord.lon, time, batch_size=batch_size,
                                             resolution=(resolution, resolution) * u.pix, distance=distance * u.AU,
                                             progress=False)

        # write image overview
        dem = outputs['dem']
        em_bins = [dem[..., (log_T > T_min) & (log_T < T_max)].sum(-1) for T_min, T_max in T_bins]

        fig, axs = plt.subplots(2, len(cmaps), figsize=(len(cmaps) * 3, 5))

        for j, cmap in enumerate(cmaps):
            im = axs[0, j].imshow(outputs['image'][..., j], cmap=cmap, norm=img_norm[j], origin='lower')
            divider = make_axes_locatable(axs[0, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        im = axs[1, 0].imshow(outputs['mean_T'], cmap='plasma', origin='lower', norm=T_norm)
        divider = make_axes_locatable(axs[1, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        for j in range(len(T_bins)):
            em_bin = em_bins[j]
            em_bin[em_bin < 1] = 1
            im = axs[1, j + 1].imshow(em_bin, cmap='jet', origin='lower', norm=em_norms[j])
            divider = make_axes_locatable(axs[1, j + 1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
            axs[1, j + 1].set_title(f'EM {T_bins[j][0]}--{T_bins[j][1]} [cm$^{-5}$]')

        im = axs[1, -1].imshow(outputs['mean_absorption'], cmap='cool', origin='lower', norm=absorption_norm)
        divider = make_axes_locatable(axs[1, -1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

        axs[1, 0].set_title('Mean T [log K]')
        axs[1, -1].set_title('Integrated Absorption')

        [ax.axis('off') for ax in axs.ravel()]
        plt.tight_layout()
        fig.savefig(os.path.join(img_path, f'{planet}_{time}.jpg'), dpi=300)
        plt.close(fig)

        # write fits files
        for c, s_map in outputs['maps'].items():
            s_map.save(os.path.join(out_path, f'{planet}_{time}_{c:03d}.fits'), overwrite=True)
