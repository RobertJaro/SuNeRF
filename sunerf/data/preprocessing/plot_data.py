# quality check for NeRF input
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import block_reduce
from sunpy.map import Map
from tqdm import tqdm

file_path = '/mnt/nerf-data/prep_2012_08/193/*'
out_path = '/home/robert_jarolim/verification'
paths = sorted(glob.glob(file_path))

if os.path.exists(out_path):
    shutil.rmtree(out_path)

os.makedirs(out_path)

for path in tqdm(paths):
    s_map = Map(path)
    if np.std(s_map.data) == 0:
        print(path)
    bn = os.path.basename(path)
    plt.imsave(f'{out_path}/{bn}.jpg', block_reduce(s_map.data, (8, 8), np.mean), vmin=0, vmax=1,
               cmap=s_map.plot_settings['cmap'])

shutil.make_archive(out_path, 'zip', out_path)
