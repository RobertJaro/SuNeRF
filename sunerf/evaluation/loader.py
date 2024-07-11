from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from astropy import units as u
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from torch import nn

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.ray_sampling import get_rays
from sunerf.train.coordinate_transformation import pose_spherical


class SuNeRFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        state = torch.load(state_path)
        data_config = state['data_config']
        self.config = data_config
        self.wavelength = data_config['wavelength']
        self.times = data_config['times']
        self.wcs = data_config['wcs']
        self.resolution = data_config['resolution']

        rendering = state['rendering']
        self.rendering = nn.DataParallel(rendering).to(device)
        model = rendering.fine_model
        self.model = nn.DataParallel(model).to(device)

        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        self.ref_time = state['ref_time']

        ref_map = Map(np.zeros(self.resolution), self.wcs)
        self.ref_map = ref_map

    @property
    def start_time(self):
        return np.min(self.times)

    @property
    def end_time(self):
        return np.max(self.times)

    @torch.no_grad()
    def load_observer_image(self, lat: u, lon: u, time: datetime,
                            distance = (1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            batch_size: int = 4096):
        # convert to pose
        target_pose = pose_spherical(-lon.to_value(u.rad), lat.to_value(u.rad), distance.to_value(u.solRad), center).numpy()
        # load rays
        if resolution is not None:
            ref_map = self.ref_map.resample(resolution)
            img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        else:
            img_coords = all_coordinates_from_map(self.ref_map).transform_to(frames.Helioprojective)

        rays_o, rays_d = get_rays(img_coords, target_pose)
        rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)

        img_shape = rays_o.shape[:2]
        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)

        time = normalize_datetime(time, self.seconds_per_dt, self.ref_time)
        flat_time = torch.ones_like(flat_rays_o[:, 0:1]) * time
        # make batches
        rays_o, rays_d, time = torch.split(flat_rays_o, batch_size), \
            torch.split(flat_rays_d, batch_size), \
            torch.split(flat_time, batch_size)

        outputs = {}
        for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
            b_outs = self.rendering(b_rays_o, b_rays_d, b_time)
            for k, v in b_outs.items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v)

        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).cpu().numpy() for k, v in outputs.items()}
        return results

    def normalize_datetime(self, time):
        return normalize_datetime(time, self.seconds_per_dt, self.ref_time)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_time)

    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048):
        target_shape = query_points_npy.shape[:-1]
        query_points = torch.from_numpy(query_points_npy).float()

        flat_query_points = query_points.reshape(-1, 4)
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        out_list = []
        for j in range(n_batches):
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)
            out_list.append(out.detach().cpu())

        output = torch.cat(out_list, 0).view(*target_shape, -1).numpy()
        return output
