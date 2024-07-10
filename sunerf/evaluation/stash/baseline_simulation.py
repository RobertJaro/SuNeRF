import glob
import os

import numpy as np
from skimage.metrics import structural_similarity
from sunpy.map import Map
from tqdm import tqdm

from sunerf.baseline.reprojection import create_heliographic_map, create_new_observer
from sunerf.data.utils import psi_norms

wl = 211
result_path = f'/mnt/results/comparison_simulation_{wl}'
data_path = f'/mnt/psi-data/PSI/AIA_{wl}'

os.makedirs(result_path, exist_ok=True)
files = glob.glob(os.path.join(data_path, '*.fits'))

ssim = []
psnr = []
mae_percent = []
me_percent = []
lat, lon = [], []

norm = psi_norms[193]

ref_maps = [Map(f) for f in files if Map(f).carrington_latitude.value <= 7]
h_map = create_heliographic_map(*ref_maps)

for i, f in enumerate(tqdm(files)):
    s_map = Map(f)
    observer = create_new_observer(s_map, s_map.heliographic_latitude, s_map.heliographic_longitude, s_map.dsun)
    sdo_new_view = h_map.reproject_to(observer)

    pred = norm(np.nan_to_num(sdo_new_view.data, nan=0))
    gt = norm(s_map.data)

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

with open(os.path.join(result_path, 'baseline_metrics.txt'), 'w') as f:
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
