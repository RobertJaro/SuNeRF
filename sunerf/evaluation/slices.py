import argparse
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
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

ref_map = loader.ref_map
center = ref_map.center.transform_to(frames.HeliographicCarrington)
print(center) # --lat -20 --lon -135
# latitude_range = np.arange(center.lat.to_value(u.deg) - 10, center.lat.to_value(u.deg) + 10, .05) * u.deg
# center_lon = 180 - center.lon.to_value(u.deg) + 5
# longitude_range = np.arange(center_lon - 20, center_lon + 20, .05) * u.deg

radius = np.linspace(1, 1.3, 100) * u.solRad
longitude_range = np.linspace(0 * u.deg, 360 * u.deg, 360 * 2 + 1)
latitude_range = np.linspace(-90, 90, 180 * 2 + 1) * u.deg

# img_out  = loader.load_image(ref_map.carrington_latitude, -ref_map.carrington_longitude, ref_map.date.to_datetime())


out = loader.load_slice(longitude_range=longitude_range, radius_range=radius, latitude_range=latitude_range)

ne = 10 ** out['log_ne']
log_T = out['log_T']

total_ne = out['total_ne'][..., 0]


# plot integrated ne map
ne_map = np.sum(total_ne, axis=-1)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

im = ax.imshow(ne_map, cmap='viridis', origin='lower', norm='log')
ax.set_title('Total electron density')
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
plt.colorbar(im, cax=cax)

plt.savefig(os.path.join(video_path, f'total_ne_map.jpg'))
plt.close(fig)


T_norm = LogNorm(vmin=5.5, vmax=6.5)
ne_norm = LogNorm(vmin=0.1)

theta, r = np.meshgrid(latitude_range.to_value(u.rad), radius.to_value(u.solRad))

for i, l in enumerate(longitude_range[::10]):
    lon_ne = ne[:, i]
    lon_total_ne = np.sum(lon_ne, axis=-1)
    mean_log_T = (log_T[None, None, :] * lon_ne).sum(axis=-1) / lon_total_ne

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': 'polar'})

    ax = axs[0]
    im = ax.pcolormesh(theta, r, lon_total_ne.T, norm=ne_norm, cmap='viridis', edgecolors='face')
    ax.set_title('Total electron density')
    plt.colorbar(im, ax=ax)
    ax.set_rlim(2 - radius[-1].value, radius[-1].value)

    ax = axs[1]
    im = ax.pcolormesh(theta, r, mean_log_T.T, cmap='inferno', norm=T_norm, edgecolors='face')
    ax.set_title('Mean temperature')
    plt.colorbar(im, ax=ax)
    ax.set_rlim(2 - radius[-1].value, radius[-1].value)

    plt.suptitle(f'Longitude: {l}')

    plt.tight_layout()
    plt.savefig(os.path.join(video_path, f'lon_slice_{i:03d}.jpg'))
    plt.close(fig)

#
# for i, r in enumerate(radius):
#     r_ne = ne[..., i, :]
#     total_ne = np.sum(r_ne, axis=-1)
#
#     mean_log_T = (log_T[None, None, :] * r_ne).sum(axis=-1) / total_ne
#
#     fig, axs = plt.subplots(2, 1, figsize=(10, 10))
#
#     ax = axs[0]
#
#     im = ax.imshow(total_ne, norm=ne_norm, cmap='viridis', origin='lower')
#     ax.set_title('Total electron density')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     ax = axs[1]
#     im = ax.imshow(mean_log_T, cmap='inferno', norm=T_norm, origin='lower')
#     ax.set_title('Mean temperature')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     plt.colorbar(im, cax=cax)
#
#     plt.suptitle(f'Radius: {r:.2f} R_sun')
#     plt.savefig(os.path.join(video_path, f'{r:.2f}.jpg'))
#     plt.close(fig)


# plot example T distribution - bar plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(log_T, ne.sum((0, 1, 2)), '-o')
# ax.plot(log_T, ne[90, 40, 20], '-o')
ax.set_ylim(1e-3, None)
ax.set_yscale('log')
ax.set_xlabel('log T')
ax.set_ylabel('ne')
plt.savefig(os.path.join(video_path, f'mean_T_distribution.jpg'))
plt.close(fig)


