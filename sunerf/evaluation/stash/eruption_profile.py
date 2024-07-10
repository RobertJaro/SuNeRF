import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sunerf.train.model import PositionalEncoder
from sunerf.utilities.data_loader import normalize_datetime
from sunpy.coordinates import HeliographicCarrington
from sunpy.coordinates.utils import GreatArc
from sunpy.map import Map
from torch import nn
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps, sdo_norms

chk_path = '/mnt/nerf-data/eruption/save_state.snf'
result_path = '/mnt/results/evaluation_eruption_slice'

sdo_map = Map('/mnt/nerf-data/sdo_2012_08/1m_304/aia.lev1_euv_12s.2012-08-31T192009Z.304.image_lev1.fits')

os.makedirs(result_path, exist_ok=True)

############################### Load NN ###################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = torch.load(chk_path, map_location=device)
encoder = PositionalEncoder(**state['encoder_kwargs'])
model = nn.DataParallel(state['fine_model']).to(device)

n_gpus = torch.cuda.device_count()
bs = 1024 * n_gpus if n_gpus > 1 else 1024

target_lon = 95
max_height = 1.3
n_points = 512

# time = sdo_map.date.to_datetime()
e_norm = ImageNormalize(vmin=0, vmax=10, stretch=AsinhStretch(0.005))
a_norm = Normalize(vmin=0, vmax=0.011)  # 0.012

############################ Map overview #################################
start = SkyCoord(target_lon * u.deg, -5 * u.deg, frame=HeliographicCarrington,
                 obstime=sdo_map.date, observer=sdo_map.observer_coordinate)
end = SkyCoord(target_lon * u.deg, -35 * u.deg, frame=HeliographicCarrington,
               obstime=sdo_map.date, observer=sdo_map.observer_coordinate)
great_arc = GreatArc(start, end)

fig = plt.figure(figsize=(5, 5))
ax = plt.subplot(111, projection=sdo_map)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
data = sdo_map.data
ax.imshow(data, norm=sdo_norms[304], cmap=sdo_cmaps[304])
ax.plot_coord(great_arc.coordinates(), color='tab:blue', linewidth=3)
sdo_map.draw_grid(axes=ax, system='carrington', grid_spacing=30 * u.deg)
ax.set_xlabel('Helioprojective Longitude [arcsec]', fontsize=14)
ax.set_ylabel('Helioprojective Latitude [arcsec]', fontsize=14)
plt.tight_layout(pad=7)
fig.savefig(os.path.join(result_path, f'sdo.png'), dpi=300, transparent=True)
plt.close(fig)

for time in tqdm(pd.date_range(datetime(2012, 8, 31, 19), datetime(2012, 8, 31, 20, 30), 91)):
    event_datetime = time.to_pydatetime()
    time = normalize_datetime(event_datetime)

    lon = target_lon
    ############################ 2D profile #################################
    lat = (180 + np.linspace(-5, -35, int(1e4), dtype=np.float32)) * np.pi / 180
    lon = (np.ones_like(lat) * lon) * np.pi / 180

    lon -= np.pi / 2  # adjust 0 lon
    r = torch.linspace(1, max_height, n_points, dtype=torch.float32)
    x = r[None] * (np.cos(lat) * np.cos(lon))[:, None]
    y = r[None] * (np.cos(lat) * np.sin(lon))[:, None]
    z = r[None] * np.sin(lat)[:, None]
    inp_coord = torch.stack([x, y, z, torch.ones_like(x) * time], -1)
    inp_tensor = inp_coord.reshape((-1, n_points, 4))
    emission_profile = []
    absorption_profile = []
    r_t = r.to(device)
    dists = r_t[1] - r_t[0]
    with torch.no_grad():
        for i in range(inp_tensor.shape[0] // bs + 1):
            x = inp_tensor[i * bs: (i + 1) * bs].to(device)
            x = x.view(-1, 4)
            x = encoder(x)
            x = model(x)
            x = x.reshape((-1, n_points, 2))
            e = torch.exp(x[:, :, 0])
            a = 1 - torch.exp(-nn.functional.relu(x[:, :, 1]) * dists)

            emission_profile += [e.detach().cpu().numpy()]
            absorption_profile += [a.detach().cpu().numpy()]

    emission_profile = np.concatenate(emission_profile)
    absorption_profile = np.concatenate(absorption_profile)

    ############################ Plot #################################
    theta, rad = np.meshgrid(lat - np.pi, r)
    ############################ Emission #################################
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar='True')
    ax.set_title(f'{event_datetime.isoformat(" ", timespec="minutes")}', fontsize=20, y=0.75, x=0.3)
    e_img = ax.pcolormesh(theta, rad, emission_profile.T, cmap=sdo_cmaps[304], norm=e_norm)
    ax.plot(lat - np.pi, np.ones_like(lat) * 0.999, color='tab:blue', linestyle='-')
    ax.set_thetamin(np.degrees(theta.min()))
    ax.set_thetamax(np.degrees(theta.max()))
    ax.set_rticks(np.arange(1.1, max_height + 0.1, 0.1))
    ax.set_xticks(np.radians([-10., -20., -30.]))
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20, pad=15)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_rlim(1 - (max_height - 1), max_height)
    ax.text(np.radians(-43), 1.15, 'Height [solar radii]',
            rotation=35.5, ha='center', va='center', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'emission_{event_datetime.isoformat(timespec="minutes")}.png'),
                dpi=300, transparent=True)
    plt.close(fig)

    ############################ Absorption #################################
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar='True')
    a_img = ax.pcolormesh(theta, rad, absorption_profile.T, cmap='viridis', norm=a_norm)
    ax.plot(lat - np.pi, np.ones_like(lat) * 0.999, color='tab:blue', linestyle='-')
    ax.set_thetamin(np.degrees(theta.min()))
    ax.set_thetamax(np.degrees(theta.max()))
    ax.set_rticks(np.arange(1.1, max_height + 0.1, 0.1))
    ax.set_xticks(np.radians([-10., -20., -30.]))
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20, pad=15)
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)
    ax.set_rlim(1 - (max_height - 1), max_height)
    ax.text(np.radians(-43), 1.15, 'Height [solar radii]',
            rotation=35.5, ha='center', va='center', fontsize=20)
    fig.tight_layout()
    fig.savefig(os.path.join(result_path, f'absorption_{event_datetime.isoformat(timespec="minutes")}.png'),
                dpi=300, transparent=True)
    plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
cbar = plt.colorbar(e_img, ax=ax, )
cbar.ax.set_yticks([0, 3, 6, 9])
ax.remove()
fig.savefig(os.path.join(result_path, 'emission_colorbar.png'), dpi=300, transparent=True)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
cbar = plt.colorbar(a_img, ax=ax, )
ax.remove()
fig.savefig(os.path.join(result_path, 'absorption_colorbar.png'), dpi=300, transparent=True)
plt.close()
