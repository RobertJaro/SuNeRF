import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sunerf.train.model import PositionalEncoder
from sunerf.utilities.data_loader import normalize_datetime
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from sunerf.baseline.reprojection import create_heliographic_map
from sunerf.data.utils import sdo_cmaps
from sunerf.rhoT_stash.volume_render import cumprod_exclusive

chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'
result_path = '/mnt/results/topo_map'

os.makedirs(result_path, exist_ok=True)
############################### Load NN ###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(chk_path)
encoder = PositionalEncoder(**state['encoder_kwargs'])
model = nn.DataParallel(state['fine_model']).to(device)

time = datetime(2012, 8, 30)
time = normalize_datetime(time)

target_lon = 302
n_points = 512
sampling_factor = 40
############################### compute map ###################################
longitude_slices = []
lons = np.linspace(-180, 180, 360 * sampling_factor + 1)
for target_lon in tqdm(lons):
    lat = np.linspace(90, 270, 180 * sampling_factor + 1, dtype=np.float32) * np.pi / 180
    lon = (np.ones_like(lat) * target_lon) * np.pi / 180

    lon -= np.pi / 2  # adjust 0 lon
    r = torch.linspace(1, 1.3, n_points, dtype=torch.float32)
    x = r[None] * (np.cos(lat) * np.cos(lon))[:, None]
    y = r[None] * (np.cos(lat) * np.sin(lon))[:, None]
    z = r[None] * np.sin(lat)[:, None]

    inp_coord = torch.stack([x, y, z, torch.ones_like(x) * time], -1)
    inp_tensor = inp_coord.reshape((-1, n_points, 4))
    emission = []
    bs = 4096 * 2
    dists = r[1] - r[0]
    with torch.no_grad():
        for i in range(inp_tensor.shape[0] // bs + 1):
            x = inp_tensor[i * bs: (i + 1) * bs].to(device)
            x = x.view(-1, 4)
            x = encoder(x)
            raw = model(x).view(-1, n_points, 2)

            intensity = torch.exp(raw[..., 0]) * dists
            absorption = torch.exp(-nn.functional.relu(raw[..., 1]) * dists)
            total_absorption = cumprod_exclusive(absorption + 1e-10)
            emerging_intensity = intensity * total_absorption
            pixel_intensity = emerging_intensity.sum(1)[:, None]

            pixel_intensity = torch.asinh(pixel_intensity / 0.005) / 5.991471
            emission += [pixel_intensity.detach().cpu().numpy()]

    emission = np.concatenate(emission)
    longitude_slices += [emission]

fig = plt.figure(figsize=(20, 10))
plt.imshow(np.stack(longitude_slices, 1)[..., 0], extent=(lons.min(), lons.max(), -90, 90), cmap=sdo_cmaps[193],
           origin='lower')
plt.axis('off')
fig.tight_layout(pad=0)
fig.savefig(os.path.join(result_path, 'sunerf_map_crop.jpg'), dpi=300)
plt.close(fig)

fig = plt.figure(figsize=(16, 8))
plt.imshow(np.stack(longitude_slices, 1)[..., 0], extent=(lons.min(), lons.max(), -90, 90), cmap=sdo_cmaps[193],
           origin='lower')
plt.xlabel('Carrington Longitude', fontsize='x-large')
plt.ylabel('Carrington Latitude', fontsize='x-large')
# plt.axvline(296, color='red')
# plt.axvline(260, color='blue')
fig.savefig(os.path.join(result_path, 'sunerf_map.jpg'), dpi=300)
plt.close(fig)

##################################### Create comparison synchronic map #####################################
stereo_a_map = Map('/mnt/nerf-data/prep_2012_08/193/2012-08-30T00:00:00_A.fits')
stereo_b_map = Map('/mnt/nerf-data/prep_2012_08/193/2012-08-30T00:00:00_B.fits')
sdo_map = Map('/mnt/nerf-data/prep_2012_08/193/aia.lev1_euv_12s.2012-08-30T000008Z.193.image_lev1.fits')

h_map = create_heliographic_map(sdo_map, stereo_a_map, stereo_b_map)

fig = plt.figure(figsize=(16, 8))
plt.imshow(h_map.data, extent=(-180, 180, -90, 90), cmap=sdo_cmaps[193], origin='lower')
plt.xlabel('Carrington Longitude', fontsize='x-large')
plt.ylabel('Carrington Latitude', fontsize='x-large')
fig.savefig(os.path.join(result_path, 'synchronic_map.jpg'), dpi=300)
plt.close(fig)
