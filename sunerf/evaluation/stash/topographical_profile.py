import os

import numpy as np
import torch
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from sunerf.train.model import PositionalEncoder
from sunerf.utilities.data_loader import normalize_datetime
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps

chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'
result_path = '/mnt/results/topo_map'
stereo_a_map = Map('/mnt/nerf-data/prep_2012_08/193/2012-08-30T00:00:00_A.fits')

os.makedirs(result_path, exist_ok=True)
############################### Load NN ###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(chk_path)
encoder = PositionalEncoder(**state['encoder_kwargs'])
model = nn.DataParallel(state['fine_model']).to(device)

time = stereo_a_map.date.to_datetime()
time = normalize_datetime(time)

target_lon = 300

############################ 2D profile #################################
lat = np.linspace(90, 270, int(1e5), dtype=np.float32) * np.pi / 180
lon = (np.ones_like(lat) * target_lon) * np.pi / 180

lon -= np.pi / 2  # adjust 0 lon
n_points = 1024
r = torch.linspace(1, 1.3, n_points, dtype=torch.float32)
x = r[None] * (np.cos(lat) * np.cos(lon))[:, None]
y = r[None] * (np.cos(lat) * np.sin(lon))[:, None]
z = r[None] * np.sin(lat)[:, None]
inp_coord = torch.stack([x, y, z, torch.ones_like(x) * time], -1)
inp_tensor = inp_coord.reshape((-1, n_points, 4))
emission_profile = []
height_profile = []
bs = 1024
r_t = r.to(device)
with torch.no_grad():
    for i in tqdm(range(inp_tensor.shape[0] // bs + 1)):
        x = inp_tensor[i * bs: (i + 1) * bs].to(device)
        x = x.view(-1, 4)
        x = encoder(x)
        x = torch.exp(model(x)[:, 0])
        x = x.reshape((-1, n_points))
        height = torch.sum(r_t[None] * x, 1) / torch.sum(x, 1)

        emission_profile += [x.detach().cpu().numpy()]
        height_profile += [height.detach().cpu().numpy()]

emission_profile = np.concatenate(emission_profile)
height_profile = np.concatenate(height_profile)
############################ Plot #################################


theta, rad = np.meshgrid(lat - np.pi, r)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar='True')
ax.pcolormesh(theta, rad, emission_profile.T, cmap=sdo_cmaps[193],
              norm=ImageNormalize(stretch=AsinhStretch(0.005)))  # X,Y & data2D must all be same dimensions
ch_cond = ((lat - np.pi) > np.radians(-43)) & ((lat - np.pi) < np.radians(-26))
fil_cond = ((lat - np.pi) > np.radians(-67)) & ((lat - np.pi) < np.radians(-57))
ax.plot(lat - np.pi, height_profile, color='black', linestyle='--')
ax.plot(lat[ch_cond] - np.pi, height_profile[ch_cond], color='blue', linestyle='--')
ax.plot(lat[fil_cond] - np.pi, height_profile[fil_cond], color='orange', linestyle='--')
ax.set_thetamin(90)
ax.set_thetamax(-90)
ax.set_rticks([1, 1.1, 1.2, 1.3])
ax.set_theta_zero_location("E")
ax.set_theta_direction(1)
# other half [0, 1] --> [-1, 1] (1:1)
# this half [1, 1.3] --> [0.7, 1.3]
ax.set_rlim(0.7, 1.3)
ax.text(-np.radians(100), 1.15, 'Height [solar radii]',
        rotation=90, ha='center', va='center')
ax.tick_params(labelleft=False, labelright=True)
fig.savefig(os.path.join(result_path, f'profile.png'), dpi=300, transparent=True)
plt.close(fig)

with open(os.path.join(result_path, f'height.txt'), 'w') as f:
    print(f'MEAN: {np.mean(height_profile)}', file=f)
    print(f'STD: {np.std(height_profile)}', file=f)
