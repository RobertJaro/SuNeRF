import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy import units as u
from iti.data.editor import AIAPrepEditor
from sunerf.utilities.data_loader import normalize_datetime
from sunpy.map import Map, all_coordinates_from_map
from sunpy.map.mapbase import PixelPair
from torch import nn

from sunerf.data.ray_sampling import get_rays
from sunerf.data.utils import sdo_norms, sdo_cmaps
from sunerf.evaluation.loader import SuNeRFLoader
from sunerf.train.coordinate_transformation import pose_spherical

results_path = '/mnt/results/method_overview'
os.makedirs(results_path, exist_ok=True)
sdo_map = Map('/mnt/nerf-data/sdo_2012_08/1h_193/aia.lev1_euv_12s.2012-08-30T000008Z.193.image_lev1.fits')
sdo_map = AIAPrepEditor(calibration='auto').call(sdo_map)

c = all_coordinates_from_map(sdo_map)
r = np.sqrt(c.Tx ** 2 + c.Ty ** 2) / sdo_map.rsun_obs

norm = sdo_norms[193]
cmap = sdo_cmaps[193]

# img = cmap(norm(sdo_map.data))
alpha = (2 - r.value) ** 8
alpha[alpha > 1] = 1
alpha[alpha < 0] = 0
# img[..., 3] =  alpha

fig = plt.figure(figsize=(4, 4))
plt.imshow(sdo_map.data, cmap=cmap, norm=norm, alpha=alpha)
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig(os.path.join(results_path, 'sdo.png'), dpi=300, transparent=True)
plt.close(fig)

###########################
chk_path = '/mnt/nerf-data/sunerf_ensemble/ensemble_4/save_state.snf'

os.makedirs(results_path, exist_ok=True)

W = 2048
scale = 2.2 * sdo_map.rsun_obs / (W * u.pix)  # frame fov width = 2.2 solar radii

focal = (.5 * W) / np.arctan((1.1 * sdo_map.rsun_obs).to(u.deg).value * np.pi / 180)
loader = SuNeRFLoader(chk_path, resolution=W, focal=focal)
cmap = sdo_cmaps[loader.wavelength]

# convert to pose
target_pose = pose_spherical(0, 0, (1 * u.AU).to(u.solRad).value).numpy()
# load rays
ref_pixel = PixelPair(x=(W - 1) / 2 * u.pix,
                      y=(W - 1) / 2 * u.pix)
rays_o, rays_d = get_rays(W, W, ref_pixel, focal, target_pose)
rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)

ray_o = rays_o[512, 512].reshape([-1, 3]).cuda()
ray_d = rays_d[512, 512].reshape([-1, 3]).cuda()

time = normalize_datetime(sdo_map.date.to_datetime())
t_time = torch.tensor(time)[None].cuda()

sampling_kwargs = loader.sampling_kwargs
query_points, z_vals = sampling_kwargs['sample_stratified'](
    ray_o, ray_d, sampling_kwargs['near'], sampling_kwargs['far'], n_samples=512, perturb=False)
# add time to query points
exp_times = t_time[:, None].repeat(1, query_points.shape[1], 1)
query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

enc_query_points = loader.encoding_fn(query_points_time.view(-1, 4))

raw = loader.fine_model(enc_query_points)
raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

dists = z_vals[..., 1:] - z_vals[..., :-1]
dists = torch.cat([dists[..., :1], dists], dim=-1)
dists = dists * torch.norm(ray_d[..., None, :], dim=-1)
emission = torch.exp(raw[..., 0]) * dists  # emission per sampled point [n_rays, n_samples]
absorption = torch.exp(-nn.functional.relu(raw[..., 1]) * dists)  # transmission per sampled point [n_rays, n_samples]

q = z_vals[0].detach().cpu().numpy()
e = emission[0].detach().cpu().numpy()
a = absorption[0].detach().cpu().numpy()
d_min, d_max = np.min(q) - 0.02, np.max(q)

plt.figure(figsize=(6, 1.5))
plt.hlines(1, d_min, d_max, color='black', linewidth=3)
plt.scatter(q[::16], np.ones_like(q)[::16], color='red', zorder=10)
plt.axis('off')
plt.savefig(os.path.join(results_path, 'sampling.png'), dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(6, 1.5))
h = np.min(e) - np.max(e) * 1e-1
plt.hlines(h, d_min, d_max, color='black', linewidth=3)
plt.scatter(q[::16], np.ones_like(q)[::16] * h, color='red', zorder=10)
plt.plot(q, e, color='tab:orange', zorder=10)
plt.axis('off')
plt.savefig(os.path.join(results_path, 'emission.png'), dpi=300, transparent=True)
plt.close()

plt.figure(figsize=(6, 1.5))
h = np.min(1 - a) - np.max(1 - a) * 1e-1
plt.hlines(h, d_min, d_max, color='black', linewidth=3)
plt.scatter(q[::16], np.ones_like(q)[::16] * h, color='red', zorder=10)
plt.plot(q, 1 - a, color='tab:blue', zorder=10)
plt.axis('off')
plt.savefig(os.path.join(results_path, 'absorption.png'), dpi=300, transparent=True)
plt.close()
