from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from astropy import units as u
from sunerf.train.ray_sampling import get_rays
from sunerf.train.volume_render import nerf_forward
from sunerf.utilities.data_loader import normalize_datetime, unnormalize_datetime
from sunpy.map.mapbase import PixelPair
from torch.nn import DataParallel

from sunerf.train.coordinate_transformation import pose_spherical


class SuNeRFLoader:

    def __init__(self, state_path, resolution=None, focal=None, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        state = torch.load(state_path)
        self.wavelength = state['wavelength']
        self.times = state['times']
        self.config = state['config']
        self.device = device
        self.model = state['model']

        self.model = DataParallel(self.model).to(device)
        self.rendering = state['rendering']
        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        self.ref_time = state['ref_time']

        self.resolution = state['data_config']['resolution']
        self.wcs = state['data_config']['wcs']

    @property
    def start_time(self):
        return self.unnormalize_datetime(np.min(self.times))

    @property
    def end_time(self):
        return self.unnormalize_datetime(np.max(self.times))

    def load_observer_image(self, lat: float, lon: float, time: datetime,
                            distance: float = (1 * u.AU).to(u.solRad).value,
                            center: Tuple[float, float, float] = None, ref_pixel: PixelPair = None,
                            strides: int = 1, batch_size: int = 4096):
        with torch.no_grad():
            # convert to pose
            target_pose = pose_spherical(-np.deg2rad(lon), np.deg2rad(lat), distance, center).numpy()
            # load rays

            mask = self.mask[::strides, ::strides]

            rays_o, rays_d = get_rays(self.img_coords, target_pose)
            rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
            img_shape = rays_o[::strides, ::strides].shape[:2]
            rays_o = rays_o[::strides, ::strides][mask].reshape([-1, 3]).to(self.device)
            rays_d = rays_d[::strides, ::strides][mask].reshape([-1, 3]).to(self.device)

            time = normalize_datetime(time, self.seconds_per_dt, self.ref_time)
            flat_time = torch.ones_like(rays_o[:, 0:1]) * time
            # make batches
            rays_o, rays_d, time = torch.split(rays_o, batch_size), \
                torch.split(rays_d, batch_size), \
                torch.split(flat_time, batch_size)

            outputs = {"tB": [], "pB": [], "density_map": []}
            for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
                b_outs = nerf_forward(b_rays_o, b_rays_d, b_time,
                                      model=self.model, sampler=self.sampler,
                                      hierarchical_sampler=self.hierarchical_sampler,
                                      rendering=self.rendering)
                outputs["tB"].append(b_outs["pixel_B"][..., 0].cpu())
                outputs["pB"].append(b_outs["pixel_B"][..., 1].cpu())
                outputs["density_map"].append(b_outs["density_map"].cpu())

            results = {}
            for key, val in outputs.items():
                val = torch.cat(val).numpy()
                img = np.ones(img_shape) * np.nan
                img[mask] = val
                results[key] = img
            return results

    def normalize_datetime(self, time):
        return normalize_datetime(time, self.seconds_per_dt, self.ref_time)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_time)

    def load_coords(self, query_points_npy):
        target_shape = query_points_npy.shape[:-1]
        query_points = torch.from_numpy(query_points_npy).float()

        flat_query_points = query_points.reshape(-1, 4)
        batch_size = 2048
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        density_list = []
        velocity_list = []
        for j in range(n_batches):
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)

            rho = 10 ** out[:, 0]
            velocity = out[:, 1:]

            density = rho * 1.12e6  # / (self.Mm_per_ds * 1e8) ** 3  # Ne/ds^3 --> Ne/cm^3
            velocity = velocity * (self.Mm_per_ds * 1e3) / self.seconds_per_dt  # km/s

            density_list.append(density.detach().cpu())
            velocity_list.append(velocity.detach().cpu())

        density = torch.cat(density_list, 0)
        velocity = torch.cat(velocity_list, 0)

        density = density.view(target_shape).numpy()
        velocity = velocity.view(target_shape + velocity.shape[-1:]).numpy()
        return {'density': density, 'velocity': velocity}
