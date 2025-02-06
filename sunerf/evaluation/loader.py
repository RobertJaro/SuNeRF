from datetime import datetime
from typing import Tuple, Iterable

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.ray_sampling import get_rays
from sunerf.evaluation.util import convert_spherical_to_cartesian
from sunerf.train.coordinate_transformation import pose_spherical


class SuNeRFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        state = torch.load(state_path)
        data_config = state['data_config']
        self.instrument_keys = list(data_config.keys())
        self.config = data_config

        rendering = state['rendering']
        self.rendering = rendering.to(device)
        model = rendering.fine_model
        self.model = model.to(device)

        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        # self.ref_time = state['ref_time']
        self.ref_time = self.start_time()

        self.ref_maps = {k: Map(np.zeros(self.resolution(k)), self.wcs(k)) for k in self.instrument_keys}
        self.ne_scaling = (1e-21 * 10000) ** 0.5  #TODO read from config

    def start_time(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return np.min(self.config[instrument_key]['times'])

    def end_time(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return np.max(self.config[instrument_key]['times'])

    def times(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return self.config[instrument_key]['times']

    def wcs(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return self.config[instrument_key]['wcs']

    def resolution(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return self.config[instrument_key]['image_shape']

    def ref_map(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        return self.ref_maps[instrument_key]

    @torch.no_grad()
    def load_observer_image(self, lat: u, lon: u, time: datetime,
                            distance=(1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            instrument_key=None,
                            **kwargs):
        # convert to pose
        target_pose = pose_spherical(-lon.to_value(u.rad), lat.to_value(u.rad), distance.to_value(u.solRad),
                                     center).numpy()
        # load rays
        ref_map = self.ref_map(instrument_key)
        if resolution is not None:
            ref_map = ref_map.resample(resolution)
            img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        else:
            img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)

        pose_out = self.load_pose(img_coords, target_pose, time, **kwargs)
        scale = [ref_map.scale[0].to_value(u.arcsec / u.pix), ref_map.scale[1].to_value(u.arcsec / u.pix)] * u.arcsec / u.pix

        reference_coord = ref_map.reference_coordinate
        observer = SkyCoord(lat=lat, lon=lon, obstime=time, radius=distance, frame=frames.HeliographicCarrington, observer='earth')
        reference_coord = SkyCoord(reference_coord.Tx, reference_coord.Ty, observer=observer,
                                   frame=frames.Helioprojective)
        maps = self.get_maps(pose_out['image'], reference_coord, scale, instrument_key)
        pose_out['maps'] = maps
        return pose_out

    @torch.no_grad()
    def load_image(self, lat: u, lon: u,
                   time: datetime,
                   distance=(1 * u.AU).to(u.solRad),
                   hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                   resolution=(256, 256) * u.pix, scale=[2400 / 256, 2400 / 256] * u.arcsec / u.pix,
                   instrument_key=None,
                   **kwargs):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]

        obs = SkyCoord(0 * u.deg, 0 * u.deg, distance, frame=frames.HeliographicStonyhurst, obstime=time)
        reference_coord = SkyCoord(hpc_lat, hpc_lon, obstime=time, observer=obs,
                                   frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution])
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)

        # convert to pose
        target_pose = pose_spherical(-lon.to_value(u.rad), lat.to_value(u.rad), distance.to_value(u.solRad)).numpy()
        # load image coordinates
        img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)

        pose_out = self.load_pose(img_coords, target_pose, time, **kwargs)
        pose_out['maps'] = self.get_maps(pose_out['image'], reference_coord, scale, instrument_key)
        return pose_out

    def load_pose(self, img_coords, target_pose, time, batch_size=int(2 ** 10), instrument_key=None, progress=True):
        # load rays
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
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        outputs = {}
        iter = tqdm(zip(rays_o, rays_d, time), total=len(rays_o)) if progress else zip(rays_o, rays_d, time)
        for b_rays_o, b_rays_d, b_time in iter:
            b_rays = torch.stack([b_rays_o, b_rays_d], 1)
            batch = {instrument_key: {'rays': b_rays, 'time': b_time}}
            fine_out, _ = self.rendering(batch)
            for k, v in fine_out[instrument_key].items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v.detach().cpu())

        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).numpy() for k, v in outputs.items()}
        return results

    def normalize_datetime(self, time):
        return normalize_datetime(time, self.seconds_per_dt, self.ref_time)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_time)

    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048, progress=True):
        target_shape = query_points_npy.shape[:-1]
        query_points = torch.from_numpy(query_points_npy).float()

        flat_query_points = query_points.reshape(-1, 4)
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        out_dict = {'log_ne': [], 'total_ne': [], 'mean_log_T': [], 'total_log_ne': [], 'ne': []}
        iter = range(n_batches) if not progress else tqdm(range(n_batches))
        for j in iter:
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)
            for k, v in out.items():
                if k not in out_dict:
                    continue
                out_dict[k].append(v.detach().cpu())
            # set temperature from model

        output = {k: torch.cat(v).reshape(*target_shape, *v[0].shape[1:]).numpy() for k, v in out_dict.items()}
        output['log_T'] = out['log_T'].detach().cpu()  # TODO move

        # unnomalize rho
        output['ne'] = output['ne'] * self.ne_scaling
        output['total_ne'] = output['total_ne'] * self.ne_scaling
        output['total_log_ne'] = output['total_log_ne'] + np.log10(self.ne_scaling)

        return output

    def load_slice(self, latitude_range=None,
                   longitude_range=None,
                   time=None,
                   radius_range=None, **kwargs):
        latitude_range = np.arange(-90, 90, 1) * u.deg if latitude_range is None else latitude_range
        longitude_range = np.arange(0, 360, 1) * u.deg if longitude_range is None else longitude_range
        radius_range = np.linspace(1, 2, 10) * u.solRad if radius_range is None else radius_range

        time = self.ref_time if time is None else time
        time = [time] if not isinstance(time, Iterable) else time
        time = [self.normalize_datetime(t) for t in time]

        coords = np.stack(np.meshgrid(latitude_range.to_value(u.rad),
                                      longitude_range.to_value(u.rad),
                                      radius_range.to_value(u.solRad), time, indexing='ij'), -1)
        x, y, z = convert_spherical_to_cartesian(coords[..., 2], coords[..., 0], coords[..., 1])
        cartesian_coords = np.stack([x, y, z, coords[..., 3]], -1)
        return self.load_coords(cartesian_coords, **kwargs)

    def get_maps(self, channel_images, reference_coord, scale, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        channels = self.config[instrument_key]['wavelengths'] if instrument_key is not None else list(range(channel_images.shape[-1]))
        
        # if 'AIA' in instrument_key:
        #     channels = [94, 131, 171, 193, 211, 304, 335]
        # elif 'EUVI' in instrument_key:
        #     channels = [171, 195, 284, 304]
        # elif 'EUI' in instrument_key:
        #     channels = [174, 304]
        # else:
        #     channels = list(range(channel_images.shape[-1]))

        maps = {}
        for i, channel in enumerate(channels):
            img = channel_images[..., i].T
            header = make_fitswcs_header(img, reference_coord, scale=scale)
            maps[channel] = Map(img, header)
        return maps
