import os

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from sunerf.train.model import PositionalEncoder
from sunerf.utilities.data_loader import normalize_datetime
from sunpy.coordinates import HeliographicCarrington
from sunpy.coordinates.utils import GreatArc
from sunpy.map import Map, all_coordinates_from_map
from torch import nn
from tqdm import tqdm

from sunerf.data.utils import sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader

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
############################ Plot #################################
target_lon = 300  # 296

start = SkyCoord(target_lon * u.deg, 90 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord(target_lon * u.deg, -90 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
great_arc = GreatArc(start, end)

start = SkyCoord(target_lon * u.deg, -26 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord(target_lon * u.deg, -43 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
ch_arc = GreatArc(start, end)

start = SkyCoord(target_lon * u.deg, -57 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord(target_lon * u.deg, -67 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
filament_arc = GreatArc(start, end)

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection=stereo_a_map)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
data = stereo_a_map.data
coords = all_coordinates_from_map(stereo_a_map)
r = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2) / stereo_a_map.rsun_obs
data[r.value > 1] = np.nan
ax.imshow(data, vmin=0, vmax=1, cmap=sdo_cmaps[193])
ax.plot_coord(great_arc.coordinates(), color='red', linewidth=5)
ax.plot_coord(ch_arc.coordinates(), color='blue', linewidth=8)
ax.plot_coord(filament_arc.coordinates(), color='orange', linewidth=8)
stereo_a_map.draw_grid(axes=ax, system='carrington', grid_spacing=30 * u.deg)
fig.tight_layout(pad=0)
fig.savefig(os.path.join(result_path, f'stereo_a.png'), dpi=300, transparent=True)
plt.close(fig)

############################ Plot SuNeRF rendered view #################################
scale = stereo_a_map.scale[0]  # scale of pixels [arcsec/pixel]
W = stereo_a_map.data.shape[0]  # number of pixels
focal = (.5 * W) / np.arctan(0.5 * (scale * W * u.pix).to(u.deg).value * np.pi / 180)

loader = SuNeRFLoader(chk_path, resolution=W, focal=focal)
outputs = loader.load_observer_image(0, -target_lon + 90, stereo_a_map.date.to_datetime(),
                                     stereo_a_map.dsun.to(u.solRad).value)

start = SkyCoord((target_lon + 90) * u.deg, 90 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord((target_lon + 90) * u.deg, -90 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
great_arc = GreatArc(start, end)

start = SkyCoord((target_lon + 90) * u.deg, -26 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord((target_lon + 90) * u.deg, -43 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
ch_arc = GreatArc(start, end)

start = SkyCoord((target_lon + 90) * u.deg, -57 * u.deg, frame=HeliographicCarrington,
                 obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
end = SkyCoord((target_lon + 90) * u.deg, -67 * u.deg, frame=HeliographicCarrington,
               obstime=stereo_a_map.date, observer=stereo_a_map.observer_coordinate)
filament_arc = GreatArc(start, end)

fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection=stereo_a_map)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
side_view = np.copy(outputs['channel_map'])
side_view[r.value > 1] = np.nan
channel_mpb = ax.imshow(side_view, cmap=sdo_cmaps[193], vmin=0, vmax=1, origin='lower')
ax.plot_coord(great_arc.coordinates(), color='red', linewidth=5)
ax.plot_coord(ch_arc.coordinates(), color='blue', linewidth=8)
ax.plot_coord(filament_arc.coordinates(), color='orange', linewidth=8)
stereo_a_map.draw_grid(axes=ax, system='carrington', grid_spacing=30 * u.deg)
plt.tight_layout(pad=0)
fig.savefig(os.path.join(result_path, f'sunerf_side.png'), transparent=True, dpi=300)
plt.close(fig)


def _get_emission(lat, lon):
    lon -= np.pi / 2  # adjust 0 lon
    r = torch.linspace(1, 1.3, 128, dtype=torch.float32)
    x = r[None] * (np.cos(lat) * np.cos(lon))[:, None]
    y = r[None] * (np.cos(lat) * np.sin(lon))[:, None]
    z = r[None] * np.sin(lat)[:, None]
    inp_coord = torch.stack([x, y, z, torch.ones_like(x) * time], -1)
    inp_tensor = inp_coord.reshape((-1, 128, 4))
    emission = []
    bs = 4096 * 2
    with torch.no_grad():
        for i in tqdm(range(inp_tensor.shape[0] // bs + 1)):
            x = inp_tensor[i * bs: (i + 1) * bs].to(device)
            x = x.view(-1, 4)
            x = encoder(x)
            x = torch.exp(model(x)[:, 0])
            x = x.reshape((-1, 128))
            x = x.sum(1)
            x = torch.asinh(x / 0.005)
            emission += [x.detach().cpu().numpy()]
    emission = np.concatenate(emission)
    return emission


############################ Main Arc #################################
main_lat = np.linspace(90, 270, int(1e6), dtype=np.float32) * np.pi / 180
lon = (np.ones_like(main_lat) * target_lon) * np.pi / 180
emission = _get_emission(main_lat, lon)

# Intensity - density ^ 2 * T_response --> use square root scaling for relative intensity to match density profile
norm = PowerNorm(gamma=0.5, vmin=emission.min(), vmax=emission.max())
n_main_emission = norm(emission)

############################ CH Arc #################################
ch_lat = np.linspace(90 + (90 - 43), 90 + (90 - 26), int(1e5), dtype=np.float32) * np.pi / 180
lon = (np.ones_like(ch_lat) * target_lon) * np.pi / 180

emission = _get_emission(ch_lat, lon)
n_ch_emission = norm(emission)
#
# ############################ Filament Arc #################################
fil_lat = np.linspace(90 + (90 - 67), 90 + (90 - 57), int(1e5), dtype=np.float32) * np.pi / 180
lon = (np.ones_like(fil_lat) * target_lon) * np.pi / 180

emission = _get_emission(fil_lat, lon)
n_fil_emission = norm(emission)

# ############################ Plot #################################
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(111, projection='polar')
ax.plot(main_lat - np.pi, n_main_emission, color='red')
ax.plot(ch_lat - np.pi, n_ch_emission, color='blue')
ax.plot(fil_lat - np.pi, n_fil_emission, color='orange')
ax.set_rlim(-1, 1)
ax.set_thetamin(90)
ax.set_thetamax(-90)
ax.set_theta_zero_location("E")
ax.set_rticks([0, 0.25, 0.5, 0.75, 1])
ax.text(-np.radians(100), .5, '(Relative Emission)$^{0.5}$',
        rotation=90, ha='center', va='center')
ax.tick_params(labelleft=False, labelright=True)
fig.savefig(os.path.join(result_path, f'topographic_{int(target_lon)}.jpg'), dpi=300)
plt.close(fig)
