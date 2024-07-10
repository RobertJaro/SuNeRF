import os

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from sunpy.map import Map

from sunerf.data.utils import sdo_cmaps, sdo_norms, loadAIAMap
from sunerf.evaluation.loader import SuNeRFLoader

chk_path = '/mnt/results/sunerf_v2_checkpoints/2022_02.snf'  # '/mnt/nerf-data/transfer_runs/2022_02/save_state.snf'
video_path = '/mnt/results/comparison_solo'

os.makedirs(video_path, exist_ok=True)

stereo_a_map = Map('/mnt/nerf-data/stereo_2022_02_converted_fov/304/2022-02-18T00:00:00_A.fits')
sdo_map = loadAIAMap('/mnt/nerf-data/sdo_2022_02/304/aia.lev1_euv_12s.2022-02-18T000007Z.304.image_lev1.fits', 2048)
solo_map = Map('/mnt/nerf-data/so_2022_02/solo_l2_eui_fsi304_image_20220218t000015151_v01.fits')

###################################### PLOT oribts ########################################
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(projection='polar')
ax.plot(sdo_map.carrington_longitude.to('rad'), sdo_map.dsun.to(u.AU), 'o', label='SDO', color='tab:blue',
        markersize=10)
ax.plot(stereo_a_map.carrington_longitude.to('rad'), sdo_map.dsun.to(u.AU), 'o', label='STEREO A', color='tab:green',
        markersize=10)
ax.plot(solo_map.carrington_longitude.to('rad'), sdo_map.dsun.to(u.AU), 'o', label='Solar Orbiter', color='tab:orange',
        markersize=10)
ax.plot(0, 0, 'o', color='yellow', markersize=30)
ax.plot(0, 0, 'o', label='Sun', color='yellow', markersize=10)  # dummy for legend
ax.set_rticks([0.6, 1, 1.4])
ax.tick_params(labeltop=False, labelbottom=True, labelleft=False, labelright=True)
ax.set_thetamin(180)
ax.set_thetamax(270)
ax.legend(facecolor='white', framealpha=1, bbox_to_anchor=(0.25, 1.15), fontsize='large')
ax.text(np.radians(285), 1, 'Distance [AU]',
        rotation=270, ha='center', va='center', fontsize='x-large')
fig.savefig(os.path.join(video_path, 'orbits.png'), dpi=300, transparent=True)
plt.close()

###################################### PLOT map overview ########################################
# crop sdo
sr = sdo_map.rsun_obs
sdo_map = sdo_map.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=sdo_map.coordinate_frame),
                         top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=sdo_map.coordinate_frame))
# crop stereo
sr = stereo_a_map.rsun_obs
stereo_a_map = stereo_a_map.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=stereo_a_map.coordinate_frame),
                                   top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=stereo_a_map.coordinate_frame))
# crop solo
sr = solo_map.rsun_obs
solo_map = solo_map.rotate(recenter=True)
solo_map = solo_map.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=solo_map.coordinate_frame),
                           top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=solo_map.coordinate_frame))

# norm = sdo_norms[304]
norm = ImageNormalize(stretch=AsinhStretch(0.005), clip=True)
plt.imsave(os.path.join(video_path, 'stereo_a.jpg'), norm(stereo_a_map.data), cmap=sdo_cmaps[304], origin='lower')
plt.imsave(os.path.join(video_path, 'sdo.jpg'), norm(sdo_map.data), cmap=sdo_cmaps[304], origin='lower')
plt.imsave(os.path.join(video_path, 'solo.jpg'), norm(solo_map.data), cmap=sdo_cmaps[304], origin='lower')

n_gpus = torch.cuda.device_count()

# init loader
W = 2048

focal = (.5 * W) / np.arctan((1.1 * solo_map.rsun_obs).to(u.deg).value * np.pi / 180)
loader = SuNeRFLoader(chk_path, resolution=W, focal=focal)
cmap = sdo_cmaps[loader.wavelength]

time, d = solo_map.date.to_datetime(), solo_map.dsun.to(u.solRad).value
lat, lon = solo_map.carrington_latitude.to(u.deg).value, solo_map.carrington_longitude.to(u.deg).value

####################### PLOT SuNeRF #######################
outputs = loader.load_observer_image(lat, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
channel_mpb = ax.imshow(sdo_norms[304].inverse(outputs['channel_map']), norm=norm, cmap=cmap, origin='lower')
ax.set_axis_off()
plt.tight_layout(pad=0)
fig.savefig(os.path.join(video_path, f'sunerf.jpg'), dpi=300)
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(channel_mpb, ax=ax, )
# cbar.ax.set_yticks([0, 3e3, 6e3, 9e3], ['0', '3e3', '6e3', '9e3'])
ax.remove()
fig.savefig(os.path.join(video_path, 'data_colorbar.png'), dpi=300, transparent=True)
plt.close()
