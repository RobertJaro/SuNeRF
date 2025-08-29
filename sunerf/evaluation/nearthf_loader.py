import pickle
from datetime import datetime
from typing import Iterable

import numpy as np
import torch
from astropy import units as u
from matplotlib import pyplot as plt
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime


class NEarthFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        state = torch.load(state_path)
        # data_config = state['data_config']
        # self.ds_keys = list(data_config.keys())
        # self.config = data_config
        # self.observers = [o for k in data_config.keys() if 'observers' in data_config[k] for o in data_config[k]['observers']]

        rendering = state['rendering']
        self.rendering = rendering.to(device)
        model = rendering.model
        self.model = model.to(device)
        self.instrument_keys = list(self.rendering.rendering_modules.keys())

        self.seconds_per_dt = state['seconds_per_dt']
        self.meters_per_ds = state['meters_per_ds']
        self.ref_date = state['ref_date']

    @torch.no_grad()
    def load_observer_image(self, x: u, z: u, time: datetime, resolution=128, obs_angle: u = 45 * u.deg,
                            instrument_key=None, **kwargs):

        # get ray directions
        angles = np.linspace(-obs_angle.to_value(u.rad) / 2, obs_angle.to_value(u.rad) / 2, resolution)
        rays_d = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=-1)  # (128, 3)
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)  # normalize directions

        # get observer location
        rays_o = np.array([x.to_value(u.m), 0, z.to_value(u.m)]) / self.meters_per_ds  # (3,)
        rays_o = np.tile(rays_o, (resolution, 1))  # (128, 3)

        rays_o = torch.tensor(rays_o, dtype=torch.float32, device=self.device)  # (128, 3)
        rays_d = torch.tensor(rays_d, dtype=torch.float32, device=self.device)  # (128, 3)

        results = self.load_rays(rays_o, rays_d, time, instrument_key=instrument_key, **kwargs)
        results['angles'] = angles
        return results

    def load_rays(self, rays_o, rays_d, time, batch_size=int(2 ** 10), instrument_key=None, progress=True,
                  model_outputs=None):
        img_shape = rays_o.shape[:1]

        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)
        time = self.normalize_datetime(time)
        flat_time = torch.ones_like(flat_rays_o[:, 0:1]) * time

        # make batches
        rays_o, rays_d, time = torch.split(flat_rays_o, batch_size), \
            torch.split(flat_rays_d, batch_size), \
            torch.split(flat_time, batch_size)
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        outputs = {k: [] for k in model_outputs} if model_outputs is not None else {}
        iter = tqdm(zip(rays_o, rays_d, time), total=len(rays_o)) if progress else zip(rays_o, rays_d, time)
        for b_rays_o, b_rays_d, b_time in iter:
            b_rays = torch.stack([b_rays_o, b_rays_d], 1)
            batch = {instrument_key: {'rays': b_rays, 'time': b_time, 'instrument': instrument_key}}
            rendering_out = self.rendering(batch)
            for k, v in rendering_out['model_out'][instrument_key].items():
                if k not in outputs and model_outputs is None:
                    outputs[k] = []
                if k not in outputs:
                    continue
                outputs[k].append(v.detach().cpu())
        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).numpy() for k, v in outputs.items()}
        return results

    def normalize_datetime(self, time):
        if isinstance(time, Iterable):
            return [normalize_datetime(t, self.seconds_per_dt, self.ref_date) for t in time]
        return normalize_datetime(time, self.seconds_per_dt, self.ref_date)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_date)


if __name__ == '__main__':
    loader = NEarthFLoader('/glade/work/rjarolim/nearthfs/qvapor/save_state.nef')

    time = datetime(2025, 1, 1)
    z = 20e3 * u.m
    obs_angle = 45 * u.deg
    resolution = 1024
    x_buffer = z / np.cos(obs_angle.to_value(u.rad) / 2)
    x_range = np.linspace(x_buffer, 73785.75 * u.m - x_buffer, resolution)

    extent = [x_range.min(), x_range.max(), 0, z]

    images = []
    for x in tqdm(x_range, desc='Load observer images'):
        result = loader.load_observer_image(x, z, time, obs_angle=obs_angle)
        image = result['image']
        images.append(image)

    image = np.stack(images, axis=0)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    im = axs.imshow(image, cmap='viridis', origin='lower', aspect='auto', norm='log')
    axs.set_ylabel('Observer position (m)')
    axs.set_xlabel('Scan (deg)')
    plt.colorbar(im, ax=axs, label='Brightness (dB)')

    fig.savefig(f'/glade/work/rjarolim/nearthfs/qvapor/observer_image.jpg', bbox_inches='tight')
    plt.close(fig)

    out_file = '/glade/work/rjarolim/nearthfs/qvapor/observation.pickle'
    data = {'image':image, 'time':time, 'z':z, 'x_range':x_range,'obs_angle':obs_angle}
    # pickle
    with open(out_file, 'wb') as f:
        pickle.dump(data, f)