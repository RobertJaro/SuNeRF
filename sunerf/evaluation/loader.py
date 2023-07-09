from typing import Tuple

import torch
from astropy import units as u
from datetime import datetime
from sunpy.map.mapbase import PixelPair
from torch import nn

from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.model import PositionalEncoder
from sunerf.train.ray_sampling import get_rays
from sunerf.train.volume_render import nerf_forward
from sunerf.utilities.data_loader import normalize_datetime


class SuNeRFLoader:

    def __init__(self, state_path, resolution=None, focal=None, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        state = torch.load(state_path)
        self.sampling_kwargs = state['sampling_kwargs']
        self.resolution = resolution if resolution is not None else state['test_kwargs']['resolution']
        self.focal = focal if focal is not None else state['test_kwargs']['focal']
        self.wavelength = state['wavelength']
        self.start_time = state['start_time']
        self.end_time = state['end_time']
        self.config = state['config']

        encoder = PositionalEncoder(**state['encoder_kwargs'])
        self.encoding_fn = lambda x: encoder(x)
        self.coarse_model = nn.DataParallel(state['coarse_model']).to(device)
        self.fine_model = nn.DataParallel(state['fine_model']).to(device)

        self.device = device

    def load_observer_image(self, lat: float, lon: float, time: datetime,
                            distance: float = (1 * u.AU).to(u.solRad).value,
                            center: Tuple[float, float, float] = None, ref_pixel: PixelPair = None,
                            strides: int = 1, batch_size: int = 4096):
        with torch.no_grad():
            # convert to pose
            target_pose = pose_spherical(lon, lat, distance, center).numpy()
            # load rays
            ref_pixel = PixelPair(x=(self.resolution - 1) / 2 * u.pix,
                                  y=(self.resolution - 1) / 2 * u.pix) if ref_pixel is None else ref_pixel
            rays_o, rays_d = get_rays(self.resolution, self.resolution, ref_pixel, self.focal, target_pose)
            rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
            img_shape = rays_o[::strides, ::strides].shape[:2]
            rays_o = rays_o[::strides, ::strides].reshape([-1, 3]).to(self.device)
            rays_d = rays_d[::strides, ::strides].reshape([-1, 3]).to(self.device)

            time = normalize_datetime(time)
            flat_time = (torch.ones(img_shape) * time).view((-1, 1)).to(self.device)
            # make batches
            rays_o, rays_d, time = torch.split(rays_o, batch_size), \
                                   torch.split(rays_d, batch_size), \
                                   torch.split(flat_time, batch_size)

            outputs = {'channel_map': [], 'height_map': [], 'absorption_map': []}
            for b_rays_o, b_rays_d, b_time in zip(rays_o, rays_d, time):
                b_outs = nerf_forward(b_rays_o, b_rays_d, b_time, self.coarse_model, self.fine_model,
                                      encoding_fn=self.encoding_fn,
                                      **self.sampling_kwargs)
                outputs['channel_map'] += [b_outs['channel_map'].cpu()]
                outputs['height_map'] += [b_outs['height_map'].cpu()]
                outputs['absorption_map'] += [b_outs['absorption_map'].cpu()]
            outputs = {k: torch.cat(v).view(img_shape).numpy() for k, v in
                       outputs.items()}
            return outputs
