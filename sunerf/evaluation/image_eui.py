import argparse
import os

import numpy as np
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
ds_key = 'EUI_FD'
start_time = loader.start_time(ds_key)
end_time = loader.end_time(ds_key)

target_time = start_time + (end_time - start_time) / 2

ref_map = loader.ref_map(ds_key)
target_longitude = ref_map.carrington_longitude.to_value(u.deg)
target_latitude = ref_map.carrington_latitude.to_value(u.deg)
target_time = ref_map.date.to_datetime()
distance = ref_map.dsun.to_value(u.AU)

# cmaps = [cm.sdoaia171, cm.sdoaia193, cm.sdoaia211, cm.sdoaia304]
cmaps = cm.sdoaia171, cm.sdoaia304
T_bins = list(reversed([(4, 6.0), (6.0, 6.5)]))

ne_norm = LogNorm(vmin=1)
absorption_norm = LogNorm(vmin=1, vmax=100)
img_norms = [ImageNormalize(vmin=0, stretch=AsinhStretch(0.005)) for _ in range(len(cmaps))]
T_norm = ImageNormalize(vmin=4.5, vmax=7)
em_norms = [LogNorm(vmin=10) for _ in T_bins]

n_points = 20
points_1 = zip(np.linspace(target_latitude, -30, n_points),
               np.linspace(target_longitude - 45, target_longitude - 45, n_points),
               [target_time] * n_points,
               np.ones(n_points) * (distance - 0.1))

# combine coordinates
points = list(points_1)

log_T = loader.log_T_range
for i, (lat, lon, time, d) in tqdm(list(enumerate(points)), total=len(points)):
    outputs = loader.load_observer_image(lat * u.deg, lon * u.deg, time,
                                         batch_size=batch_size, instrument_key=ds_key,
                                         resolution=(resolution, resolution) * u.pix, distance=d * u.AU,
                                         model_outputs=['image', 'mean_T', 'total_ne', 'mean_absorption', 'dem'])
    dem = outputs['dem']
    em_bins = [dem[..., (log_T > T_min) & (log_T < T_max)].sum(-1) for T_min, T_max in T_bins]

    fig, axs = plt.subplots(1, len(cmaps) + len(T_bins) , figsize=(10, 3))

    for j, (cmap, img_norm) in enumerate(zip(cmaps, img_norms)):
        im = axs[j].imshow(outputs['image'][..., j], cmap=cmap, norm=img_norm, origin='lower')
        divider = make_axes_locatable(axs[j])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)

    for j in range(len(T_bins)):
        ax = axs[j + len(cmaps)]
        im = ax.imshow(em_bins[j], cmap='jet', origin='lower', norm=em_norms[j])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title(f'EM {T_bins[j][0]}--{T_bins[j][1]} [cm$^{-5}$]')

    axs[0].set_ylabel(r'SuNeRF 174 $\text{\AA}$', fontsize=16)
    axs[1].set_ylabel(r'SuNeRF 304 $\text{\AA}$', fontsize=16)

    [ax.axis('off') for ax in axs.ravel()]
    plt.tight_layout()
    fig.savefig(os.path.join(video_path, f'image_eui_{i:03d}.jpg'), dpi=300)
    plt.close(fig)
