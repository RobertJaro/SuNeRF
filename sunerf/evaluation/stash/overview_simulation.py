import glob
import os

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from sunpy.map import Map
from tqdm import tqdm

from sunerf.baseline.reprojection import create_heliographic_map, create_new_observer
from sunerf.data.utils import sdo_cmaps, psi_norms
from sunerf.evaluation.loader import SuNeRFLoader

chk_paths = ['/mnt/results/psi_models/psi_193.snf',
             '/mnt/results/sunerf_v2_checkpoints/psi_ensemble_1.snf',
             '/mnt/results/sunerf_v2_checkpoints/psi_ensemble_2.snf',
             '/mnt/results/sunerf_v2_checkpoints/psi_ensemble_3.snf',
             '/mnt/results/sunerf_v2_checkpoints/psi_ensemble_4.snf']
result_path = '/mnt/results/comparison_simulation_193'
data_path = '/mnt/psi-data/PSI/AIA_193'

n_gpus = torch.cuda.device_count()

os.makedirs(result_path, exist_ok=True)

# init loader
ensemble_loaders = [SuNeRFLoader(p) for p in chk_paths]
loader = ensemble_loaders[0]
cmap = sdo_cmaps[loader.wavelength]

files = glob.glob(os.path.join(data_path, '*.fits'))

norm = psi_norms[loader.wavelength]

ref_maps = [Map(f) for f in files if Map(f).carrington_latitude.value <= 7]
h_map = create_heliographic_map(*ref_maps)

# plot image overview
plot_files = [f for f in files if Map(f).carrington_longitude.value > 345 and Map(f).carrington_latitude.value >= 0]

gt_imgs, pred_imgs, diff_imgs, uncertainty_imgs, reprojected_imgs = [], [], [], [], []

for f in tqdm(plot_files, desc='Load plot images'):
    s_map = Map(f)
    predictions = [loader.load_observer_image(s_map.carrington_latitude.value, -s_map.carrington_longitude.value,
                                              s_map.date.to_datetime(),
                                              distance=s_map.dsun.to(u.solRad).value, batch_size=4096 * n_gpus,
                                              strides=1)['channel_map']
                   for loader in ensemble_loaders]
    uncertainty_imgs += [np.std(predictions, 0) * 100]
    pred = predictions[0]
    gt = norm(s_map.data)
    gt_imgs += [gt]
    pred_imgs += [pred]
    diff_imgs += [(pred - gt) * 100]  # in percent of the image range
    # create stitched image
    observer = create_new_observer(s_map, s_map.heliographic_latitude, s_map.heliographic_longitude, s_map.dsun)
    sdo_new_view = h_map.reproject_to(observer)
    reprojected_imgs += [sdo_new_view.data]

latitudes = np.array([Map(f).carrington_latitude.value for f in plot_files])
# sort by latitude
sort_idx = np.argsort(latitudes)
latitudes = latitudes[sort_idx]
gt_imgs = np.array(gt_imgs)[sort_idx]
pred_imgs = np.array(pred_imgs)[sort_idx]
diff_imgs = np.array(diff_imgs)[sort_idx]
reprojected_imgs = np.array(reprojected_imgs)[sort_idx]
uncertainty_imgs = np.array(uncertainty_imgs)[sort_idx]

v_min_max = 20  # np.abs(diff_imgs).max()
diff_norm = SymLogNorm(vmin=-v_min_max, vmax=v_min_max, linthresh=2, clip=True)
# v_min_max = np.abs(uncertainty_imgs).max()
unc_norm = SymLogNorm(vmin=0, vmax=v_min_max, linthresh=2, clip=True)
data_norm = psi_norms[171]

fig, axs = plt.subplots(5, len(plot_files), figsize=(10, 8))
for ax, img in zip(axs[0], reprojected_imgs):
    ax.imshow(np.nan_to_num(img, nan=np.nanmin(img)), cmap=cmap, norm=data_norm)
for ax, img in zip(axs[1], gt_imgs):
    ax.imshow(norm.inverse(img), cmap=cmap, norm=data_norm)
for ax, img in zip(axs[2], pred_imgs):
    data_mpb = ax.imshow(norm.inverse(img), cmap=cmap, norm=data_norm)
for ax, img in zip(axs[3], diff_imgs):
    diff_mpb = ax.imshow(img, cmap='RdBu', norm=diff_norm)
for ax, img in zip(axs[4], uncertainty_imgs):
    unc_mpb = ax.imshow(img, cmap='inferno', norm=unc_norm)

for ax, c_lat in zip(axs[0], latitudes):
    ax.set_title(r'$%2d^\circ$' % c_lat)

axs[0, 0].set_ylabel('Baseline')
axs[1, 0].set_ylabel('Ground Truth')
axs[2, 0].set_ylabel('SuNeRF')
axs[3, 0].set_ylabel('SuNeRF - GT')
axs[4, 0].set_ylabel('Uncertainty')
[(ax.set_xticks([]), ax.set_yticks([])) for ax in np.ravel(axs)]

fig.tight_layout()
fig.savefig(os.path.join(result_path, 'overview_neurIPS.jpg'), dpi=300)
plt.close()

# save colorbars
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(diff_mpb, ax=ax, )
cbar.ax.set_yticks([-20, -10, -2, -1, 1, 2, 10, 20], ['-20%', '-10%', '-2%', '-1%', '1%', '2%', '10%', '20%'])
ax.remove()
fig.savefig(os.path.join(result_path, 'diff_colorbar.png'), dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(unc_mpb, ax=ax, )
cbar.ax.set_yticks([1, 2, 10, 20], ['1%', '2%', '10%', '20%'])
ax.remove()
fig.savefig(os.path.join(result_path, 'unc_colorbar_neurIPS.png'), dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
cbar = plt.colorbar(data_mpb, ax=ax, )
cbar.ax.set_yticks([10, 100, 1000, 10000], ['$10^{1}$', '$10^{2}$', '$10^{3}$', '$10^{4}$'])
ax.remove()
fig.savefig(os.path.join(result_path, 'data_colorbar.png'), dpi=300, transparent=True)
plt.close()
