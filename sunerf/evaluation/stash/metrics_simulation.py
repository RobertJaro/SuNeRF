import glob
import os
import pickle

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps, psi_norms
from sunerf.evaluation.loader import SuNeRFLoader

wl = 171
model_path = f'/mnt/results/psi_models/psi_{wl}.snf'
result_path = f'/mnt/results/comparison_simulation_{wl}'
data_path = f'/mnt/psi-data/PSI/AIA_{wl}'

n_gpus = torch.cuda.device_count()

os.makedirs(result_path, exist_ok=True)

# init loader
loader = SuNeRFLoader(model_path)
cmap = sdo_cmaps[loader.wavelength]

files = glob.glob(os.path.join(data_path, '*.fits'))

ssim = []
psnr = []
mae_percent = []
me_percent = []
lat, lon = [], []

norm = psi_norms[loader.wavelength]

for i, f in enumerate(tqdm(files)):
    s_map = Map(f)
    outputs = loader.load_observer_image(s_map.carrington_latitude.value, -s_map.carrington_longitude.value,
                                         s_map.date.to_datetime(),
                                         distance=s_map.dsun.to(u.solRad).value, batch_size=4096 * n_gpus, strides=1)
    pred = outputs['channel_map']
    gt = norm(s_map.data)
    diff = pred - gt
    ssim += [structural_similarity(pred, gt, data_range=1)]
    mse = ((pred - gt) ** 2).mean()
    mae_percent += [np.abs(pred - gt).mean() * 100]
    me_percent += [(pred - gt).mean() * 100]
    psnr += [-10. * np.log10(mse)]
    lat += [s_map.carrington_latitude.value]
    lon += [s_map.carrington_longitude.value]

lat = np.array(lat)
psnr = np.array(psnr)
ssim = np.array(ssim)
mae_percent = np.array(mae_percent)
me_percent = np.array(me_percent)

with open(os.path.join(result_path, 'evaluation.pickle'), 'wb') as f:
    pickle.dump({'lat': lat, 'psnr': psnr, 'ssim': ssim, 'mae_percent': mae_percent, 'me_percent': me_percent}, f)

with open(os.path.join(result_path, 'metrics.txt'), 'w') as f:
    print(f'ALL', file=f)
    print(f'PSNR; {np.mean(psnr)}', file=f)
    print(f'SSIM; {np.mean(ssim)}', file=f)
    print(f'MAE; {np.mean(mae_percent)}', file=f)
    print(f'ME; {np.mean(me_percent)}', file=f)
    print(f'TEST', file=f)
    test_cond = lat > 7
    print(f'PSNR; {np.mean(psnr[test_cond])}', file=f)
    print(f'SSIM; {np.mean(ssim[test_cond])}', file=f)
    print(f'MAE; {np.mean(mae_percent[test_cond])}', file=f)
    print(f'ME; {np.mean(me_percent[test_cond])}', file=f)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
sc1 = axs[0].scatter(lon, lat, c=psnr)
axs[0].set_title('PSNR')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
axs[0].axhline(7, linestyle='--', color='red')
axs[0].axhline(-7, linestyle='--', color='red')

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sc1, cax=cax, orientation='vertical')

sc2 = axs[1].scatter(lon, lat, c=ssim, cmap='plasma')
axs[1].set_title('SSIM')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
axs[1].axhline(7, linestyle='--', color='red')
axs[1].axhline(-7, linestyle='--', color='red')

divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(sc2, cax=cax, orientation='vertical')

fig.tight_layout()
fig.savefig(os.path.join(result_path, 'metrics.jpg'), dpi=300)
plt.close()
