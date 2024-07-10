import os
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sunerf.train.model import PositionalEncoder
from sunerf.utilities.data_loader import normalize_datetime
from torch import nn

from sunerf.data.utils import sdo_cmaps

chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'
result_path = '/mnt/results/voxel_volume'

os.makedirs(result_path, exist_ok=True)
############################### Load NN ###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(chk_path)
encoder = PositionalEncoder(**state['encoder_kwargs'])
model = nn.DataParallel(state['fine_model']).to(device)

time = datetime(2012, 8, 30)
time = normalize_datetime(time)

############################### compute map ###################################

n_points = 256
grid = np.stack(np.meshgrid(np.linspace(-1.3, 1.3, n_points, dtype=np.float32),
                            np.linspace(-1.3, 1.3, n_points, dtype=np.float32),
                            np.linspace(-1.3, 1.3, n_points, dtype=np.float32)), -1)
inp_coord = torch.from_numpy(grid)
inp_tensor = inp_coord.reshape((-1, 3))
intensity = []
bs = 8096 * 2
with torch.no_grad():
    for i in range(inp_tensor.shape[0] // bs + 1):
        x = inp_tensor[i * bs: (i + 1) * bs]
        x = torch.cat([x, torch.ones_like(x)[..., :1] * time], -1)  # add time
        x = x.to(device)
        x = x.view(-1, 4)
        x = encoder(x)
        raw = model(x)

        emission = torch.exp(raw[..., 0])

        pixel_intensity = torch.asinh(emission / 0.005)
        intensity += [pixel_intensity.detach().cpu().numpy()]

intensity = np.concatenate(intensity).reshape((n_points, n_points, n_points))
norm = Normalize(vmin=intensity.min(), vmax=intensity.max())
n_emission = norm(intensity)

cond = (np.sqrt((grid ** 2).sum(-1)) > 1) & (np.sqrt((grid ** 2).sum(-1)) < 1.3) & (n_emission > 0.2)

plot_grid = grid[cond]
colors = sdo_cmaps[193](n_emission)[cond][:, :3]
alpha = n_emission[cond]

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection='3d')
ax.set_axis_off()
ax.scatter(plot_grid[:, 0], plot_grid[:, 1], plot_grid[:, 2], c=colors, alpha=alpha, marker="o", s=.05)

# plot surface sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = 1 * np.outer(np.cos(u), np.sin(v))
y = 1 * np.outer(np.sin(u), np.sin(v))
z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='white', linewidth=1)
ax.view_init(elev=45., azim=0)

plt.tight_layout(pad=0)
fig.savefig(os.path.join(result_path, 'volume.jpg'), dpi=300, transparent=True)
plt.close(fig)

##################################### Create comparison synchronic map #####################################
