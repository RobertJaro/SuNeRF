import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
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
start_time = loader.start_time('AIA_FD')
end_time = loader.end_time('AIA_FD')

target_time = start_time + (end_time - start_time) / 2
target_time =  pd.Timestamp(target_time).round('H').to_pydatetime()

central_longitude = loader.ref_map().carrington_longitude.to_value(u.deg)
central_latitude = loader.ref_map().carrington_latitude.to_value(u.deg)

center_coord = loader.ref_map('AIA_FD').center.transform_to(frames.HeliographicCarrington)
target_longitude = center_coord.lon.to_value(u.deg)
target_latitude = center_coord.lat.to_value(u.deg)

n_points = 20

points_1 = zip(np.linspace(0, 0, n_points),
               np.linspace(0, 360, n_points),
               pd.date_range(start=target_time, end=target_time, periods=n_points),
               np.linspace(1.0, 1.0, n_points))

points_2 = zip(np.linspace(0, 360, n_points),
               np.linspace(0, 0, n_points),
               pd.date_range(start=target_time, end=target_time, periods=n_points),
               np.linspace(1.0, 1.0, n_points))

# combine coordinates
points = list(points_1) + list(points_2)

# cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]
cmaps = cm.sdoaia94, cm.sdoaia131, cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304, cm.sdoaia335
T_bins = list(reversed([(4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 6.5), (6.5, 7.0)]))

ne_norm = LogNorm(vmin=1)
absorption_norm = LogNorm(vmin=1, vmax=100)
img_norms = [ImageNormalize(vmin=0, stretch=AsinhStretch(0.005)) for _ in range(len(cmaps))]
T_norm = ImageNormalize(vmin=4.5, vmax=7)
em_norms = [LogNorm(vmin=10) for _ in T_bins]

for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time, batch_size=batch_size,
                                         resolution=(resolution, resolution) * u.pix, distance=d * u.AU)

    log_T = np.linspace(4, 9, 101)
    dem = outputs['dem']
    em_bins = [dem[..., (log_T > T_min) & (log_T < T_max)].sum(-1) for T_min, T_max in T_bins]

    fig, axs = plt.subplots(2, len(cmaps), figsize=(len(cmaps) * 3, 5))

    for j, (cmap, img_norm) in enumerate(zip(cmaps, img_norms)):
        im = axs[0, j].imshow(outputs['image'][..., j], cmap=cmap, norm=img_norm, origin='lower')
        divider = make_axes_locatable(axs[0, j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    im = axs[1, 0].imshow(outputs['mean_T'], cmap='plasma', origin='lower', norm=T_norm)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    for j in range(len(T_bins)):
        im = axs[1, j + 1].imshow(em_bins[j], cmap='jet', origin='lower', norm=em_norms[j])
        divider = make_axes_locatable(axs[1, j + 1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        axs[1, j + 1].set_title(f'EM {T_bins[j][0]}--{T_bins[j][1]} [cm$^{-5}$]')

    # im = axs[1, 1].imshow(outputs['total_ne'], cmap='viridis', origin='lower', norm=ne_norm)
    # divider = make_axes_locatable(axs[1, 1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # fig.colorbar(im, cax=cax)

    im = axs[1, -1].imshow(outputs['mean_absorption'], cmap='cool', origin='lower', norm=absorption_norm)
    divider = make_axes_locatable(axs[1, -1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    # [ax.axis('off') for ax in axs[1, 2:]]

    # axs[1].imshow(outputs['height_map'], cmap='plasma', vmin=1, vmax=1.2, origin='lower')
    # axs[2].imshow(outputs['absorption_map'], cmap='viridis', origin='lower')

    axs[1, 0].set_title('Mean T [log K]')
    # axs[1, 1].set_title('Total N$_e$ [cm$^{-2}$]')
    axs[1, -1].set_title('Integrated Absorption')

    [ax.axis('off') for ax in axs.ravel()]
    # add title with lat, lon, time
    plt.suptitle(f'Lat: {lat:.1f} Lon: {lon:.1f} Time: {time.strftime("%Y-%m-%d %H:%M:%S")} AU: {d:.2f}')
    plt.tight_layout()
    fig.savefig(os.path.join(video_path, '%03d.jpg' % i), dpi=300)
    plt.close(fig)
