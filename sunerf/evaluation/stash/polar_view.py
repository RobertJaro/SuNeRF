import datetime
import glob
import os

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import block_reduce
from astropy.visualization import ImageNormalize, AsinhStretch
from chronnos.evaluate.detect import CHRONNOSDetector
from matplotlib import pyplot as plt, cm
from matplotlib.colors import Normalize, SymLogNorm
from sunpy.map import Map
from tqdm import tqdm

from sunerf.baseline.reprojection import create_heliographic_map, create_new_observer
from sunerf.data.utils import sdo_cmaps, sdo_norms
from sunerf.evaluation.loader import SuNeRFLoader

chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'
video_path = '/mnt/results/polar_view'

os.makedirs(video_path, exist_ok=True)

stereo_a_map = Map('/mnt/nerf-data/prep_2012_08/193/2012-08-30T00:00:00_A.fits')
stereo_b_map = Map('/mnt/nerf-data/prep_2012_08/193/2012-08-30T00:00:00_B.fits')
sdo_map = Map('/mnt/nerf-data/prep_2012_08/193/aia.lev1_euv_12s.2012-08-30T000008Z.193.image_lev1.fits')

plt.imsave(os.path.join(video_path, 'stereo_a.jpg'), stereo_a_map.data, cmap=sdo_cmaps[193], vmin=0, vmax=1,
           origin='lower')
plt.imsave(os.path.join(video_path, 'stereo_b.jpg'), stereo_b_map.data, cmap=sdo_cmaps[193], vmin=0, vmax=1,
           origin='lower')
plt.imsave(os.path.join(video_path, 'sdo.jpg'), sdo_map.data, cmap=sdo_cmaps[193], vmin=0, vmax=1, origin='lower')

h_map = create_heliographic_map(sdo_map, stereo_a_map, stereo_b_map)

n_gpus = torch.cuda.device_count()

# init loader
W = 2048
scale = 2.2 * sdo_map.rsun_obs / (W * u.pix)  # frame fov width = 2.2 solar radii

focal = (.5 * W) / np.arctan((1.1 * sdo_map.rsun_obs).to(u.deg).value * np.pi / 180)
loader = SuNeRFLoader(chk_path, resolution=W, focal=focal)
cmap = sdo_cmaps[loader.wavelength]

time, d = sdo_map.date.to_datetime(), sdo_map.dsun.to(u.solRad).value
lon = sdo_map.carrington_longitude.value + 60

####################### PLOT SuNeRF #######################
lats = np.linspace(0, -90, 4).astype(int)
for lat in tqdm(lats, desc='Plot Latitudes'):
    outputs = loader.load_observer_image(lat, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    channel_mpb = ax.imshow(sdo_norms[193].inverse(outputs['channel_map']), cmap=cmap, norm=sdo_norms[193],
                            origin='lower')
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, f'sunerf_{lat}.jpg'), dpi=300)
    plt.close(fig)

    observer = create_new_observer(sdo_map, lat * u.deg, 60 * u.deg, sdo_map.dsun)
    sdo_new_view = h_map.reproject_to(observer)
    sr = sdo_new_view.rsun_obs
    sdo_new_view = sdo_new_view.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=sdo_new_view.coordinate_frame),
                                       top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=sdo_new_view.coordinate_frame))
    plt.imsave(os.path.join(video_path, f'baseline_{lat}.jpg'), np.nan_to_num(sdo_new_view.data, nan=0),
               cmap=sdo_cmaps[193], vmin=0, vmax=1, origin='lower')

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(channel_mpb, ax=ax, )
cbar.ax.set_yticks([0, 3e3, 6e3, 9e3], ['0', '3e3', '6e3', '9e3'])
ax.remove()
fig.savefig(os.path.join(video_path, 'data_colorbar.png'), dpi=300, transparent=True)
plt.close()

####################### PLOT Uncertainty #######################

ensemble_chks = sorted(glob.glob('/mnt/nerf-data/sunerf_ensemble/*/save_state.snf'))
uncertainty_imgs = []
for lat in tqdm(lats, desc='Ensemble'):
    ensemble_maps = []
    for chk in ensemble_chks:
        loader = SuNeRFLoader(chk, resolution=W, focal=focal)
        outputs = loader.load_observer_image(lat, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
        ensemble_maps += [outputs['channel_map']]
    uncertainty_imgs += [np.std(ensemble_maps, 0) * 100]

unc_norm = SymLogNorm(vmin=0, vmax=20, linthresh=5, clip=True)
for img, lat in zip(uncertainty_imgs, lats):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    unc_mpb = ax.imshow(img, norm=unc_norm, cmap='inferno', origin='lower')
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    fig.savefig(os.path.join(video_path, f'uncertainty_{lat}.jpg'), dpi=300)
    plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(unc_mpb, ax=ax, )
cbar.ax.set_yticks([1, 5, 10, 20], ['1%', '5%', '10%', '20%'])
ax.remove()
fig.savefig(os.path.join(video_path, 'unc_colorbar.png'), dpi=300, transparent=True)
plt.close()

####################### PLOT all SuNeRF wavelengths #######################

# 193
loader_193 = SuNeRFLoader(chk_path, resolution=W, focal=focal)
cmap = sdo_cmaps[loader_193.wavelength]
outputs_193 = loader_193.load_observer_image(-90, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
plt.imsave(os.path.join(video_path, f'sunerf_193.jpg'), outputs_193['channel_map'], cmap=cmap, vmin=0, vmax=1,
           origin='lower')
del loader_193

# 304
loader_304 = SuNeRFLoader('/mnt/nerf-data/transfer_runs/304/save_state.snf', resolution=W, focal=focal)
cmap = sdo_cmaps[loader_304.wavelength]
outputs_304 = loader_304.load_observer_image(-90, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
plt.imsave(os.path.join(video_path, f'sunerf_304.jpg'), outputs_304['channel_map'], cmap=cmap, vmin=0, vmax=.7,
           origin='lower')
del loader_304

# 171
loader_171 = SuNeRFLoader('/mnt/nerf-data/transfer_runs/171/save_state.snf', resolution=W, focal=focal)
cmap = sdo_cmaps[loader_171.wavelength]
outputs_171 = loader_171.load_observer_image(-90, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
plt.imsave(os.path.join(video_path, f'sunerf_171.jpg'), outputs_171['channel_map'], cmap=cmap, vmin=0, vmax=1,
           origin='lower')
del loader_171

# 211
loader_211 = SuNeRFLoader('/mnt/nerf-data/transfer_runs/211/save_state.snf', resolution=W, focal=focal)
cmap = sdo_cmaps[loader_211.wavelength]
outputs_211 = loader_211.load_observer_image(-90, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=1)
plt.imsave(os.path.join(video_path, f'sunerf_211.jpg'), outputs_211['channel_map'], cmap=cmap, vmin=0, vmax=1,
           origin='lower')
del loader_211

####################### Detect Coronal Hole #######################
chronnos_detector = CHRONNOSDetector(model_name='chronnos_euv_v1_0.pt')
sdo_ch_map = chronnos_detector.predict(
    [['/mnt/nerf-data/sdo_2012_08/1h_171/aia.lev1_euv_12s.2012-08-30T000012Z.171.image_lev1.fits'],
     ['/mnt/nerf-data/sdo_2012_08/1h_193/aia.lev1_euv_12s.2012-08-30T000008Z.193.image_lev1.fits'],
     ['/mnt/nerf-data/sdo_2012_08/1h_211/aia.lev1_euv_12s.2012-08-30T000001Z.211.image_lev1.fits'],
     ['/mnt/nerf-data/sdo_2012_08/1h_304/aia.lev1_euv_12s.2012-08-30T000009Z.304.image_lev1.fits']], reproject=True)[0]
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(sdo_map.data, extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
ax.contour(sdo_ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1, 1, -1, 1), linewidths=5)
ax.set_axis_off()
plt.tight_layout(pad=0)
f.savefig(os.path.join(video_path, f'sdo_ch.jpg'), dpi=300)
plt.close(f)

stereo_a_ch_map = chronnos_detector.predict([
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/171/2012-08-30T00:00:00_A.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/195/2012-08-30T00:00:00_A.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/284/2012-08-30T00:00:00_A.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/304/2012-08-30T00:00:00_A.fits']], calibrate=False, reproject=True)[0]
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(stereo_a_map.data, extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
ax.contour(stereo_a_ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1, 1, -1, 1), linewidths=5)
ax.set_axis_off()
plt.tight_layout(pad=0)
f.savefig(os.path.join(video_path, f'stereo_a_ch.jpg'), dpi=300)
plt.close(f)

stereo_b_ch_map = chronnos_detector.predict([
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/171/2012-08-30T00:00:00_B.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/195/2012-08-30T00:00:00_B.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/284/2012-08-30T00:00:00_B.fits'],
    ['/mnt/nerf-data/stereo_2012_08_converted_fov/304/2012-08-30T00:00:00_B.fits']], calibrate=False, reproject=True)[0]
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(stereo_b_map.data, extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
ax.contour(stereo_b_ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1, 1, -1, 1), linewidths=5)
ax.set_axis_off()
plt.tight_layout(pad=0)
f.savefig(os.path.join(video_path, f'stereo_b_ch.jpg'), dpi=300)
plt.close(f)

ch_map = create_heliographic_map(sdo_ch_map, stereo_a_ch_map, stereo_b_ch_map)
plt.imsave(os.path.join(video_path, f'ch_map.jpg'), np.nan_to_num(ch_map.data, nan=0))

observer = create_new_observer(sdo_map, -90 * u.deg, 60 * u.deg, sdo_map.dsun)
ch_map = ch_map.reproject_to(observer)
sr = ch_map.rsun_obs
ch_map = ch_map.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=ch_map.coordinate_frame),
                       top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=ch_map.coordinate_frame))

chronnos_norms = {
    171: ImageNormalize(vmin=0, vmax=6457.5, stretch=AsinhStretch(0.005), clip=True),  # 171
    193: ImageNormalize(vmin=0, vmax=7757.31, stretch=AsinhStretch(0.005), clip=True),  # 193
    211: ImageNormalize(vmin=0, vmax=6539.8, stretch=AsinhStretch(0.005), clip=True),  # 211
    304: ImageNormalize(vmin=0, vmax=3756, stretch=AsinhStretch(0.005), clip=True),  # 304
}

img_171 = block_reduce(outputs_171['channel_map'], (4, 4), func=np.mean)
img_171 = chronnos_norms[171](sdo_norms[171].inverse(img_171))

img_193 = block_reduce(outputs_193['channel_map'], (4, 4), func=np.mean)
img_193 = chronnos_norms[193](sdo_norms[193].inverse(img_193))

img_211 = block_reduce(outputs_211['channel_map'], (4, 4), func=np.mean)
img_211 = chronnos_norms[211](sdo_norms[211].inverse(img_211))

img_304 = block_reduce(outputs_304['channel_map'], (4, 4), func=np.mean)
img_304 = chronnos_norms[304](sdo_norms[304].inverse(img_304))

input_img = np.stack([img_171, img_193, img_211, img_304], 0).astype(np.float32)
input_img = torch.from_numpy(input_img)[None].cuda() * 2 - 1
with torch.no_grad():
    ch_detection = chronnos_detector.model(input_img).cpu()[0, 0]

f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(outputs_193['channel_map'], extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
sunerf_ch_line = ax.contour(ch_detection, levels=[0.3], colors=['red'], extent=(-1, 1, -1, 1), linestyles='dashed',
                            linewidths=4)
reprojection_ch_line = ax.contour(ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1, 1, -1, 1),
                                  linestyles='dashed', linewidths=4)
ax.set_axis_off()
plt.tight_layout(pad=0)
f.savefig(os.path.join(video_path, f'sunerf_ch.jpg'), dpi=300)
plt.close(f)

fig = plt.figure(figsize=(3, 2))
fig.legend(sunerf_ch_line.legend_elements()[0] + reprojection_ch_line.legend_elements()[0],
           ['SuNeRF', 'Reprojection'], loc='center', fancybox=True, shadow=True, borderpad=1)
fig.savefig(os.path.join(video_path, 'ch_legend.png'), dpi=300, transparent=True)
plt.close(fig)

####################### PLOT viewpoints #######################
sdo_files = glob.glob('/mnt/nerf-data/prep_2012_08/193/aia*')
sdo_maps = [Map(f) for f in sdo_files]

stereo_a_files = glob.glob('/mnt/nerf-data/prep_2012_08/193/*_A.fits')
stereo_a_maps = [Map(f) for f in stereo_a_files]

stereo_b_files = glob.glob('/mnt/nerf-data/prep_2012_08/193/*_B.fits')
stereo_b_maps = [Map(f) for f in stereo_b_files]

sdo_r = [m.dsun.to(u.solRad).value for m in sdo_maps]
sdo_theta = [m.carrington_longitude.value * np.pi / 180 for m in sdo_maps]
sdo_dates = np.array([m.date.to_datetime() for m in sdo_maps])

stereo_a_r = [m.dsun.to(u.solRad).value for m in stereo_a_maps]
stereo_a_theta = [m.carrington_longitude.value * np.pi / 180 for m in stereo_a_maps]
stereo_a_dates = np.array([m.date.to_datetime() for m in stereo_a_maps])

stereo_b_r = [m.dsun.to(u.solRad).value for m in stereo_b_maps]
stereo_b_theta = [m.carrington_longitude.value * np.pi / 180 for m in stereo_b_maps]
stereo_b_dates = np.array([m.date.to_datetime() for m in stereo_b_maps])

dates = np.concatenate([sdo_dates, stereo_a_dates, stereo_b_dates])
min_date = np.min(dates)
dates = (dates - min_date) / datetime.timedelta(days=1)

sdo_dates = (sdo_dates - min_date) / datetime.timedelta(days=1)
stereo_a_dates = (stereo_a_dates - min_date) / datetime.timedelta(days=1)
stereo_b_dates = (stereo_b_dates - min_date) / datetime.timedelta(days=1)
sdo_dates = sdo_dates.astype(np.float32)
stereo_a_dates = stereo_a_dates.astype(np.float32)
stereo_b_dates = stereo_b_dates.astype(np.float32)

norm = Normalize(vmin=np.min(dates), vmax=np.max(dates))

f = plt.figure(figsize=(20, 20))
ax = f.add_subplot(111, polar=True)

# SDO
sdo_cm = cm.get_cmap('Blues')
colors = sdo_cm(norm(sdo_dates))
cs = colors.tolist()
for c in colors:
    cs.append(c)
    cs.append(c)
ax.quiver(sdo_theta, sdo_r, -np.cos(sdo_theta), -np.sin(sdo_theta), pivot='tail', color=colors, scale=30,
          edgecolor='black', linewidth=1)
sdo_obs_theta = sdo_map.carrington_longitude.value * np.pi / 180
ax.quiver(sdo_obs_theta, 250, -np.cos(sdo_obs_theta), -np.sin(sdo_obs_theta), pivot='tail', color='blue', scale=20,
          edgecolor='black', linewidth=1)
# STEREO A
sdo_cm = cm.get_cmap('Greens')
colors = sdo_cm(norm(stereo_a_dates))
cs = colors.tolist()
for c in colors:
    cs.append(c)
    cs.append(c)
ax.quiver(stereo_a_theta, stereo_a_r, -np.cos(stereo_a_theta), -np.sin(stereo_a_theta), pivot='tail', color=colors,
          scale=30, edgecolor='black', linewidth=1)
stereo_a_obs_theta = stereo_a_map.carrington_longitude.value * np.pi / 180
ax.quiver(stereo_a_obs_theta, 250, -np.cos(stereo_a_obs_theta), -np.sin(stereo_a_obs_theta), pivot='tail',
          color='green', scale=20, edgecolor='black', linewidth=1)
# STEREO B
sdo_cm = cm.get_cmap('Reds')
colors = sdo_cm(norm(stereo_b_dates))
cs = colors.tolist()
for c in colors:
    cs.append(c)
    cs.append(c)
ax.quiver(stereo_b_theta, stereo_b_r, -np.cos(stereo_b_theta), -np.sin(stereo_b_theta), pivot='tail', color=colors,
          scale=30, edgecolor='black', linewidth=1)
stereo_b_obs_theta = stereo_b_map.carrington_longitude.value * np.pi / 180
ax.quiver(stereo_b_obs_theta, 250, -np.cos(stereo_b_obs_theta), -np.sin(stereo_b_obs_theta), pivot='tail', color='red',
          scale=20, edgecolor='black', linewidth=1)
obs_theta = lon * np.pi / 180
ax.quiver(obs_theta, 250, -np.cos(obs_theta), -np.sin(obs_theta), pivot='tail', color='white', scale=20,
          edgecolor='black', linewidth=1)

ax.scatter(0, 0, color='white')

ax.set_axis_off()
f.savefig(os.path.join(video_path, 'viewpoints.png'), dpi=300, transparent=True)
plt.close(f)

######################### TEST observed limb alignment #########################
# observer = create_new_observer(sdo_map, -90 * u.deg, 60 * u.deg, sdo_map.dsun)
# sdo_new_view = h_map.reproject_to(observer)
# sr = sdo_new_view.rsun_obs
# sdo_new_view = sdo_new_view.submap(bottom_left=SkyCoord(-sr * 1.1, -sr * 1.1, frame=sdo_new_view.coordinate_frame),
#                                    top_right=SkyCoord(sr * 1.1, sr * 1.1, frame=sdo_new_view.coordinate_frame))
#
# outputs_193 = loader_193.load_observer_image(-90, -lon, time, distance=d, batch_size=4096 * n_gpus, strides=4, ref_pixel=sdo_map.reference_pixel)
#
# f, ax = plt.subplots(2, 2, figsize=(10, 10))
# ax[0, 0].imshow(sdo_new_view.data, extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
# ax[0, 0].contour(ch_detection, levels=[0.3], colors=['red'], extent=(-.95, .95, -.95, .95), linestyles='dashed')
# ax[0, 1].imshow(sdo_new_view.data, extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
# ax[0, 1].contour(ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1, 1, -1, 1), linestyles='dashed')
#
# coords = all_coordinates_from_map(ch_map)
# r = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / ch_map.rsun_obs
# ax[1, 0].imshow(outputs_193['channel_map'], extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
# ax[1, 0].contour(ch_detection, levels=[0.3], colors=['red'], extent=(-1, 1, -1, 1), linestyles='dashed')
# ax[1, 1].imshow(outputs_193['channel_map'], extent=(-1, 1, -1, 1), origin='lower', vmin=0, vmax=1, cmap=sdo_cmaps[193])
# ax[1, 1].contour(r, levels=[1], colors=['red'], extent=(-1, 1, -1, 1), linestyles='dashed')
# ax[1, 1].contour(ch_map.data, levels=[0.3], colors=['tab:blue'], extent=(-1.0, 1.0, -1.0, 1.0), linestyles='dashed')
# # ax.set_axis_off()
# plt.tight_layout(pad=0)
# f.savefig(os.path.join(video_path, f'test.jpg'), dpi=300)
# plt.close(f)
