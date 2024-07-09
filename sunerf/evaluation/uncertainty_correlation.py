import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units as u
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr
from sunpy.map import Map
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps, psi_norms
from sunerf.evaluation.loader import SuNeRFLoader

chk_paths = ['/mnt/nerf-data/sunerf_publication_results/psi_models/psi_193.snf',
             '/mnt/nerf-data/sunerf_publication_results/sunerf_v2_checkpoints/psi_ensemble_1.snf',
             '/mnt/nerf-data/sunerf_publication_results/sunerf_v2_checkpoints/psi_ensemble_2.snf',
             '/mnt/nerf-data/sunerf_publication_results/sunerf_v2_checkpoints/psi_ensemble_3.snf',
             '/mnt/nerf-data/sunerf_publication_results/sunerf_v2_checkpoints/psi_ensemble_4.snf']

result_path = f'/mnt/results/uncertainty_correlation'
data_path = f'/mnt/nerf-data/PSI/AIA_193'

batch_size = 4096
n_gpus = torch.cuda.device_count()
print('GPUs: ', [torch.cuda.get_device_name(i) for i in range(n_gpus)])

os.makedirs(result_path, exist_ok=True)

# init loader
ensemble_loaders = [SuNeRFLoader(p) for p in chk_paths]
loader = ensemble_loaders[0]
cmap = sdo_cmaps[loader.wavelength]

files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
files = [f for f in files if Map(f).carrington_latitude.value > 7]  # only consider test set

ssim = []
psnr = []
mae_percent = []
me_percent = []
lat, lon = [], []

norm = psi_norms[loader.wavelength]
strides = 2

preds = []
gts = []
uncs = []
errors = []

for i, f in enumerate(tqdm(files)):
    s_map = Map(f)

    predictions = [loader.load_observer_image(s_map.carrington_latitude.value, -s_map.carrington_longitude.value,
                                              s_map.date.to_datetime(),
                                              distance=s_map.dsun.to(u.solRad).value, batch_size=batch_size * n_gpus,
                                              strides=strides)['channel_map']
                   for loader in ensemble_loaders]
    pred = predictions[0]
    gt = norm(s_map.data)[::strides, ::strides]
    unc = (np.std(predictions, 0) * 100)

    mae = np.abs(pred - gt)
    # plt.imsave(os.path.join(result_path, f'pred_{i}.jpg'), pred)
    # plt.imsave(os.path.join(result_path, f'gt_{i}.jpg'), gt)
    # plt.imsave(os.path.join(result_path, f'mae_{i}.jpg'), mae)
    # plt.imsave(os.path.join(result_path, f'unc_{i}.jpg'), unc)

    preds += np.ravel(pred).tolist()
    gts += np.ravel(gt).tolist()
    uncs += np.ravel(unc).tolist()
    errors += np.ravel(mae).tolist()

pearson_corr = np.corrcoef(errors, uncs)[0, 1]
spearman_corr = spearmanr(errors, uncs).correlation

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

h = axs[0].hist2d(gts, preds, bins=64, norm=LogNorm(), cmap='cividis')
axs[0].set_xlabel('Ground-Truth [normalized counts]')
axs[0].set_ylabel('SuNeRF [normalized counts]')
cbar = plt.colorbar(mappable=h[3], ax=axs[0])
axs[0].set_xlim(0, .9)
axs[0].set_ylim(0, .9)
axs[0].set_xticks(np.arange(0, 1, 0.1))
axs[0].set_yticks(np.arange(0, 1, 0.1))
axs[0].set_aspect(1 / axs[0].get_data_ratio())

h = axs[1].hist2d(errors, uncs, bins=64, norm=LogNorm(), cmap='plasma')
axs[1].set_xlabel('Mean-Absolute-Error [normalized counts]')
axs[1].set_ylabel('Uncertainty Estimate')
cbar = plt.colorbar(mappable=h[3], ax=axs[1])
axs[1].set_xlim(0, None)
axs[1].set_ylim(0, None)
axs[1].set_aspect(1 / axs[1].get_data_ratio())

fig.savefig(os.path.join(result_path, f'correlation.jpg'), dpi=300)
plt.close()

print(f'CORRELATION: {pearson_corr}')
print(spearmanr(errors, uncs))
