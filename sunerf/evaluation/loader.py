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
from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.evaluation.util import convert_spherical_to_cartesian
from sunerf.rendering.base_tracing import MultiResolutionRenderingModule
from sunerf.train.coordinate_transformation import pose_spherical


class SuNeRFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        state = torch.load(state_path)
        data_config = state['data_config']
        self.instrument_keys = list(data_config.keys())
        self.config = data_config
        self.observers = [o for k in data_config.keys() for o in data_config[k]['observers']]

        rendering = state['rendering']
        self.rendering = rendering.to(device)
        model = rendering.fine_model if isinstance(rendering, MultiResolutionRenderingModule) else rendering.model
        self.model = model.to(device)

        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        self.ref_date = state['ref_date']

        self.ref_maps = {k: Map(np.zeros(self.resolution(k)), self.wcs(k)) for k in self.instrument_keys}

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
        target_pose = pose_spherical(lon.to_value(u.rad), lat.to_value(u.rad),
                                     distance.to_value(u.solRad) / self.Rs_per_ds,
                                     center).numpy()
        # load rays
        ref_map = self.ref_map(instrument_key)
        if resolution is not None:
            ref_map = ref_map.resample(resolution)
            img_coords = get_azimuthal_equidistant_coordinates(ref_map)
        else:
            img_coords = get_azimuthal_equidistant_coordinates(ref_map)

        pose_out = self.load_pose(img_coords, target_pose, time, **kwargs)
        scale = [ref_map.scale[0].to_value(u.arcsec / u.pix),
                 ref_map.scale[1].to_value(u.arcsec / u.pix)] * u.arcsec / u.pix

        reference_coord = ref_map.reference_coordinate
        observer = SkyCoord(lat=lat, lon=lon, obstime=time, radius=distance, frame=frames.HeliographicCarrington,
                            observer='earth')
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
        rays_o, rays_d = get_rays(img_coords[..., 0], img_coords[..., 1], target_pose)
        rays_o, rays_d = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
        img_shape = rays_o.shape[:2]

        flat_rays_o = rays_o.reshape([-1, 3]).to(self.device)
        flat_rays_d = rays_d.reshape([-1, 3]).to(self.device)
        time = self.normalize_datetime(time)
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
            batch = {instrument_key: {'rays': b_rays, 'time': b_time, 'instrument': instrument_key}}
            rendering_out = self.rendering(batch)
            for k, v in rendering_out['model_out'][instrument_key].items():
                if k not in outputs:
                    outputs[k] = []
                outputs[k].append(v.detach().cpu())

        results = {k: torch.cat(v).view(*img_shape, *v[0].shape[1:]).numpy() for k, v in outputs.items()}
        return results

    def normalize_datetime(self, time):
        if isinstance(time, Iterable):
            return [normalize_datetime(t, self.seconds_per_dt, self.ref_date) for t in time]
        return normalize_datetime(time, self.seconds_per_dt, self.ref_date)

    def unnormalize_datetime(self, time):
        return unnormalize_datetime(time, self.seconds_per_dt, self.ref_date)

    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048, progress=True):
        target_shape = query_points_npy.shape[:-1]
        query_points = torch.from_numpy(query_points_npy).float()

        flat_query_points = query_points.reshape(-1, 4)
        n_batches = np.ceil(len(flat_query_points) / batch_size).astype(int)

        out_dict = {}
        iter = range(n_batches) if not progress else tqdm(range(n_batches))
        for j in iter:
            batch = flat_query_points[j * batch_size:(j + 1) * batch_size].to(self.device)
            out = self.model(batch)
            for k, v in out.items():
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v.detach().cpu())

        output = {k: torch.cat(v).reshape(*target_shape, *v[0].shape[1:]).numpy() for k, v in out_dict.items()}

        return output

    def load_slice(self, latitude_range=None,
                   longitude_range=None,
                   time=None,
                   radius_range=None, **kwargs):
        latitude_range = np.arange(-90, 90, 1) * u.deg if latitude_range is None else latitude_range
        longitude_range = np.arange(0, 360, 1) * u.deg if longitude_range is None else longitude_range
        radius_range = np.linspace(1, 2, 10) * u.solRad if radius_range is None else radius_range

        time = self.ref_date if time is None else time
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
        if 'AIA' in instrument_key:
            channels = [94, 131, 171, 193, 211, 304, 335]
        elif 'EUVI' in instrument_key:
            channels = [171, 195, 284, 304]
        elif 'EUI' in instrument_key:
            channels = [174, 304]
        else:
            channels = list(range(channel_images.shape[-1]))

        maps = {}
        for i, channel in enumerate(channels):
            img = channel_images[..., i]
            header = make_fitswcs_header(img, reference_coord, scale=scale)
            maps[channel] = Map(img, header)
        return maps


class ThomsonSuNeRFLoader(SuNeRFLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rho_scaling = 57.80811838603689  # from calibration

    def load_coords(self, *args, **kwargs):
        output = super().load_coords(*args, **kwargs)
        # unnormalize rho
        output['rho'] = output['rho'] * self.rho_scaling
        output['log_rho'] = output['log_rho'] * self.rho_scaling

        return output

    def load_cube(self, radius_range, time, pixel_per_Rs, **kwargs):
        max_radius = radius_range[1]
        #
        cartesian_coords = np.stack(np.meshgrid(
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        # only load the points in the radius range
        r = np.linalg.norm(cartesian_coords[..., :3], axis=-1)
        mask = (r >= radius_range[0]) & (r <= radius_range[1])
        sub_coords = cartesian_coords[mask]

        # normalize coordinates
        sub_coords[..., 0:3] = sub_coords[..., 0:3] / self.Rs_per_ds
        # load the coordinates
        model_out = self.load_coords(sub_coords, **kwargs)
        rho = model_out['rho']
        v = model_out['v']

        rho_cube = np.zeros((*cartesian_coords.shape[:-1],))
        rho_cube[mask] = rho.squeeze(-1)

        v_cube = np.zeros((*cartesian_coords.shape[:-1], 3))
        v_cube[mask] = v

        return {'rho': rho_cube, 'v': v_cube, 'cartesian_coords': cartesian_coords}
