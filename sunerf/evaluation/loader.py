from datetime import datetime
from typing import Tuple, Iterable

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from torch import nn
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.data.ray_sampling import get_rays, hpc_impact_parameter
from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.evaluation.util import convert_spherical_to_cartesian
from sunerf.rendering.base_tracing import MultiResolutionRenderingModule
from sunerf.train.coordinate_transformation import pose_spherical, spherical_to_cartesian


class SuNeRFLoader:

    def __init__(self, state_path, device=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.device = device

        state = torch.load(state_path, map_location=device, weights_only=False)
        self.state = state
        data_config = state['data_config']
        self.ds_keys = list(data_config.keys())
        self.config = data_config
        self.observers = [o for k in data_config.keys() if 'observers' in data_config[k] for o in
                          data_config[k]['observers']]

        rendering = state['rendering']
        self.rendering = rendering.to(device).eval()
        model = rendering.fine_model if isinstance(rendering, MultiResolutionRenderingModule) else rendering.model
        self.model = nn.DataParallel(model).to(device).eval()  # wrap model for multi-gpu inference
        self.instrument_keys = list(self.rendering.rendering_modules.keys())

        self.seconds_per_dt = state['seconds_per_dt']
        self.Rs_per_ds = state['Rs_per_ds']
        self.Mm_per_ds = self.Rs_per_ds * (1 * u.R_sun).to_value(u.Mm)
        self.ref_date = state['ref_date']

        self.ref_maps = {k: Map(np.zeros(self.resolution(k)), self.wcs(k)) for k in self.ds_keys}

    def instrument_key(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key].get('instrument_key', ds_key)

    def start_time(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return np.min(self.config[ds_key]['times'])

    def end_time(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return np.max(self.config[ds_key]['times'])

    def times(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['times']

    def wcs(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['wcs']

    def resolution(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.config[ds_key]['image_shape']

    def ref_map(self, ds_key=None):
        ds_key = ds_key if ds_key is not None else self.ds_keys[0]
        return self.ref_maps[ds_key]

    @torch.no_grad()
    def load_observer_image(self, lat: u, lon: u, time: datetime,
                            distance=(1 * u.AU).to(u.solRad),
                            center: Tuple[float, float, float] = None, resolution=None,
                            instrument_key=None,
                            **kwargs):
        # convert to pose
        if center is not None:
            raise NotImplementedError(
                "Offset look-at centers are not supported by pose_spherical."
            )
        target_pose = pose_spherical(lon.to_value(u.rad), lat.to_value(u.rad),
                                     distance.to_value(u.solRad) / self.Rs_per_ds)
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
                   instrument_key=None, **kwargs):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]

        obs = SkyCoord(0 * u.deg, 0 * u.deg, distance, frame=frames.HeliographicStonyhurst, obstime=time)
        reference_coord = SkyCoord(hpc_lat, hpc_lon, obstime=time, observer=obs,
                                   frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution])
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)

        # convert to pose
        target_pose = pose_spherical(lon.to_value(u.rad), lat.to_value(u.rad),
                                     distance.to_value(u.solRad) / self.Rs_per_ds)
        # load image coordinates
        img_coords = get_azimuthal_equidistant_coordinates(ref_map)

        pose_out = self.load_pose(img_coords, target_pose, time, **kwargs)
        pose_out['maps'] = self.get_maps(pose_out['image'], reference_coord, scale, instrument_key)
        return pose_out

    @torch.no_grad()
    def load_pose(self, img_coords, target_pose, time, batch_size=int(2 ** 10), instrument_key=None, progress=True,
                  model_outputs=['image', 'mean_T', 'total_ne', 'mean_absorption']):
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

    @torch.no_grad()
    def load_coords(self, query_points_npy, batch_size=2048, progress=False):
        query_points = torch.from_numpy(query_points_npy).float()

        nan_mask = ~torch.isnan(query_points).any(-1)
        flat_query_points = query_points[nan_mask]
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

        output = {}
        for k in out_dict.keys():
            v = out_dict[k][0]
            out_v = torch.ones(query_points.shape[:-1] + v.shape[1:], dtype=v.dtype) * torch.nan
            out_v[nan_mask] = torch.cat(out_dict[k])
            output[k] = out_v.numpy()

        return output

    def load_spherical(self, latitude_range=None, longitude_range=None, time=None, radius_range=None, **kwargs):
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
        if 'thomson_normalization' not in self.state:
            raise KeyError("Missing 'thomson_normalization' in saved Thomson state.")

        normalization = self.state['thomson_normalization']
        required_keys = ['msb', 'sigma_ne', 'msb_norm', 'drho_cm3']
        missing_keys = [k for k in required_keys if k not in normalization]
        if missing_keys:
            raise KeyError(
                f"Saved Thomson state is missing normalization keys: {', '.join(missing_keys)}"
            )

        self.msb = normalization['msb']  # ph/cm2/s/sr
        self.sigma_ne = normalization['sigma_ne']
        self.msb_norm = normalization['msb_norm']
        self.drho_cm3 = normalization['drho_cm3']
        self.correction_modules = self.state.get('correction_modules', nn.ModuleDict()).to(self.device)
        self.calibration_modules = self.state.get('calibration_modules', nn.ModuleDict()).to(self.device)
        self.correction_modules.eval()
        self.calibration_modules.eval()

    def _get_correction_norms(self, instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        matching_ds_keys = [
            ds_key for ds_key in self.ds_keys
            if self.instrument_key(ds_key) == instrument_key
        ]
        if not matching_ds_keys:
            raise ValueError(
                f"No dataset config maps to instrument_key '{instrument_key}'. "
                f"Available instrument keys: {', '.join(self.instrument_keys)}."
            )

        norms = {
            (
                float(self.config[ds_key].get('image_norm', 512.0)),
                float(self.config[ds_key].get('hpc_norm', 1e4)),
            )
            for ds_key in matching_ds_keys
        }
        if len(norms) != 1:
            raise ValueError(
                f"Dataset configs for instrument_key '{instrument_key}' use inconsistent correction norms: "
                f"{sorted(norms)}."
            )
        return next(iter(norms))

    @staticmethod
    def _get_image_coords(shape, image_norm):
        ny, nx = shape
        image_coords = np.stack(np.mgrid[:ny, :nx], axis=-1).astype(np.float32)
        image_coords[..., 0] -= 0.5 * (ny - 1)
        image_coords[..., 1] -= 0.5 * (nx - 1)
        image_coords /= float(image_norm)
        return image_coords

    def _get_hpc_coords(self, ref_map, hpc_norm):
        coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)
        x = coords.Tx.to_value(u.arcsec)
        y = coords.Ty.to_value(u.arcsec)
        distance = np.ones_like(x, dtype=np.float32) * ref_map.dsun.to_value(u.solRad)
        hpc_coords = np.stack([x, y, distance], axis=-1).astype(np.float32)
        hpc_coords[..., :2] /= float(hpc_norm)
        hpc_coords[..., 2] /= float(self.Rs_per_ds)
        return hpc_coords

    def _build_correction_inputs(self, ref_map, instrument_key=None):
        image_norm, hpc_norm = self._get_correction_norms(instrument_key)
        image_coords = self._get_image_coords(ref_map.data.shape, image_norm)
        hpc_coords = self._get_hpc_coords(ref_map, hpc_norm)
        time_value = self.normalize_datetime(ref_map.date.datetime)
        time = np.full(ref_map.data.shape + (1,), time_value, dtype=np.float32)
        return image_coords, hpc_coords, time

    @staticmethod
    def _mask_invalid_coords(output, ref_map):
        finite_mask = np.isfinite(ref_map.data)
        for key, value in output.items():
            if value.shape[:2] != finite_mask.shape:
                continue
            value = value.copy()
            value[~finite_mask] = np.nan
            output[key] = value
        return output

    @torch.no_grad()
    def load_correction_masks(self, ref_map, instrument_key=None, apply_valid_mask=True):
        ref_map = Map(ref_map)
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        image_coords, hpc_coords, time = self._build_correction_inputs(ref_map, instrument_key)

        outputs = {}
        correction_module = self.correction_modules[instrument_key] if instrument_key in self.correction_modules else None
        calibration_module = self.calibration_modules[instrument_key] if instrument_key in self.calibration_modules else None

        if correction_module is not None:
            zero_image = torch.zeros(ref_map.data.shape + (2,), dtype=torch.float32, device=self.device)
            _, corrections = correction_module(
                zero_image,
                torch.from_numpy(image_coords).to(self.device),
                torch.from_numpy(hpc_coords).to(self.device),
                torch.from_numpy(time).to(self.device),
            )
            outputs.update({k: v.detach().cpu().numpy()[..., 0] for k, v in corrections.items()})

        if calibration_module is not None:
            calibration_scalar = torch.exp(calibration_module.calibration.detach()).item()
            outputs['instrument_calibration'] = np.full(ref_map.data.shape, calibration_scalar, dtype=np.float32)

        if apply_valid_mask:
            outputs = self._mask_invalid_coords(outputs, ref_map)

        return {k: Map(v.astype(np.float32), ref_map.meta) for k, v in outputs.items()}

    @torch.no_grad()
    def load_correction_image(self, lat: u, lon: u, time: datetime,
                              distance=(1 * u.AU).to(u.solRad),
                              hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                              resolution=(256, 256) * u.pix, scale=None,
                              instrument_key=None):
        instrument_key = instrument_key if instrument_key is not None else self.instrument_keys[0]
        if scale is None:
            scale = [2400 / resolution[0].to_value(u.pix), 2400 / resolution[1].to_value(u.pix)] * u.arcsec / u.pix

        obs = SkyCoord(lat=lat, lon=lon, distance=distance, frame=frames.HeliocentricInertial, obstime=time)
        reference_coord = SkyCoord(hpc_lat, hpc_lon, obstime=time, observer=obs, frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution], dtype=np.float32)
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)
        return self.load_correction_masks(ref_map, instrument_key=instrument_key, apply_valid_mask=False)

    def convert_rho(self, model_rho):
        # convert to electron density in cm^-3
        physical_rho = model_rho * self.drho_cm3
        return physical_rho

    def convert_column_density(self, model_column_density):
        """Convert ``integral rho_model d(model distance)`` to electrons cm^-2."""
        ds_cm = float(self.Rs_per_ds) * (1 * u.R_sun).to_value(u.cm)
        return model_column_density * self.drho_cm3 * ds_cm

    @torch.no_grad()
    def load_image(self, lat: u, lon: u,
                   time: datetime,
                   distance=(1 * u.AU).to(u.solRad),
                   hpc_lat: u = 0 * u.arcsec, hpc_lon: u = 0 * u.arcsec,
                   resolution=(256, 256) * u.pix, scale=None,
                   occ_min=None, occ_max=None,
                   instrument_key=None, **kwargs):
        if scale is None:
            if occ_max is not None:
                scale = _get_scale_from_occ_max(occ_max, distance, resolution)
            else:
                scale = [2400 / 256, 2400 / 256] * u.arcsec / u.pix

        obs = SkyCoord(lat=lat, lon=lon, distance=distance, frame=frames.HeliocentricInertial, obstime=time)
        reference_coord = SkyCoord(hpc_lat, hpc_lon, obstime=time, observer=obs,
                                   frame=frames.Helioprojective)
        mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution])
        header = make_fitswcs_header(mock_data, reference_coord, scale=scale)
        ref_map = Map(mock_data, header)

        # apply occulter mask
        mask = _get_mask(ref_map, occ_min, occ_max)
        ref_map.data[mask] = np.nan

        return self.load_map(ref_map, **kwargs)

    @torch.no_grad()
    def load_map(self, ref_map, filter_occ=True, **kwargs):
        map_loader = MapDataLoader(self.Rs_per_ds, 'inertial', azimuthal_equidistant=False)
        map_data = map_loader.load(ref_map)  # image, pose, rays, time, observer
        # convert to pose
        target_pose = pose_spherical(map_data['observer']['longitude'].to_value(u.rad),
                                     map_data['observer']['latitude'].to_value(u.rad),
                                     map_data['observer']['radius'].to_value(u.solRad) / self.Rs_per_ds)
        # load image coordinates
        img_coords = all_coordinates_from_map(ref_map)
        img_coords = np.stack([img_coords.Tx, img_coords.Ty], -1)

        # occulter mask
        if filter_occ:
            mask = np.isnan(ref_map.data)
            img_coords[mask] = np.nan

        pose_out = self.load_pose(img_coords, target_pose, map_data['observer']['time'],
                                  model_outputs=['image', 'density'], **kwargs)

        # create maps
        tB_map = Map(pose_out['image'][..., 0], ref_map.meta)
        pB_map = Map(pose_out['image'][..., 1], ref_map.meta)
        density_map = Map(pose_out['density'], ref_map.meta)
        #
        return {'tB_map': tB_map, 'pB_map': pB_map, 'density_map': density_map}

    def load_spherical_cube(self, radius, latitude, longitude, time, **kwargs):
        spherical_coords = np.stack(np.meshgrid(
            radius.to_value(u.R_sun),
            latitude.to_value(u.rad),
            longitude.to_value(u.rad),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_latitude(self, radius_range, time, latitude, Nr=128, Nphi=128, longitude_range=None, **kwargs):
        longitude_range = [0, 2 * np.pi] * u.rad if longitude_range is None else longitude_range
        spherical_coords = np.stack(np.meshgrid(
            np.linspace(radius_range[0].to_value(u.R_sun), radius_range[1].to_value(u.R_sun), Nr),
            latitude.to_value(u.rad),
            np.linspace(longitude_range[0].to_value(u.rad), longitude_range[1].to_value(u.rad), Nphi, endpoint=False),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_longitude(self, radius_range, time, longitude, latitude_range=None, Nr=128, Ntheta=128, **kwargs):
        latitude_range = [0, 2 * np.pi] * u.rad if latitude_range is None else latitude_range
        spherical_coords = np.stack(np.meshgrid(
            np.linspace(radius_range[0].to_value(u.R_sun), radius_range[1].to_value(u.R_sun), Nr),
            np.linspace(latitude_range[0].to_value(u.rad), latitude_range[1].to_value(u.rad), Ntheta, endpoint=False),
            longitude.to_value(u.rad),
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        cartesian_coords = spherical_to_cartesian(spherical_coords[..., :3], np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        query_points = np.concatenate([cartesian_coords, spherical_coords[..., 3:4]], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {'rho': rho, 'v': v, 'spherical_coords': spherical_coords}

    def load_radius(self, radius, time, Ntheta=128, Nphi=256, projection='lat', **kwargs):
        if projection == 'lat':
            latitude = np.linspace(-np.pi / 2, np.pi / 2, Ntheta, endpoint=False)
            latitude_axis = np.rad2deg(latitude)
        elif projection == 'sinlat':
            latitude_axis = np.linspace(-1.0, 1.0, Ntheta)
            latitude = np.arcsin(latitude_axis)
        else:
            raise ValueError(f"Unsupported projection '{projection}'. Use 'lat' or 'sinlat'.")

        coords = np.stack(np.meshgrid(
            radius.to_value(u.R_sun),
            latitude,
            np.linspace(0, 2 * np.pi, Nphi, endpoint=False),
            [1],
            indexing='ij'
        ), -1)
        sky_coords = SkyCoord(radius=coords[..., 0] * u.R_sun,
                              lat=coords[..., 1] * u.rad,
                              lon=coords[..., 2] * u.rad,
                              frame=frames.HeliographicCarrington, obstime=time,
                              observer='self')
        sky_coords = sky_coords.transform_to(frames.HeliocentricInertial)

        spherical_coords = np.stack([sky_coords.distance.to_value(u.R_sun),
                                     sky_coords.lat.to_value(u.rad),
                                     sky_coords.lon.to_value(u.rad)], axis=-1)
        cartesian_coords = spherical_to_cartesian(spherical_coords, np)
        # normalize coordinates
        cartesian_coords = cartesian_coords / self.Rs_per_ds
        # append time
        time_coords = np.ones_like(cartesian_coords[..., 0:1]) * self.normalize_datetime(time)
        query_points = np.concatenate([cartesian_coords, time_coords], axis=-1)
        # load the coordinates
        model_out = self.load_coords(query_points, **kwargs)
        rho = model_out['rho']
        v = model_out['v']
        return {
            'rho': rho,
            'v': v,
            'spherical_coords': spherical_coords,
            'projection': projection,
            'latitude_axis': latitude_axis
        }

    def load_coords(self, *args, **kwargs):
        output = super().load_coords(*args, **kwargs)
        # convert rho to physical units
        output['rho'] = self.convert_rho(output['rho'])
        output['log_rho'] = np.log(output['rho'])
        output['v'] = output['v'] * (self.Mm_per_ds / self.seconds_per_dt) * 1e3  # convert to km/s

        return output

    def load_cube(self, radius_range: u.solRad, time, pixel_per_Rs, **kwargs):
        radius_range = radius_range.to_value(u.R_sun)
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

    def load_slice(self, radius_range, time, z, pixel_per_Rs, **kwargs):
        max_radius = radius_range[1].to_value(u.R_sun)
        #
        cartesian_coords = np.stack(np.meshgrid(
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            np.linspace(-max_radius, max_radius, int((2 * max_radius + 1) * pixel_per_Rs)),
            z,
            self.normalize_datetime(time),
            indexing='ij'
        ), -1)
        # only load the points in the radius range
        r = np.linalg.norm(cartesian_coords[..., :3], axis=-1)
        mask = (r >= radius_range[0].to_value(u.R_sun)) & (r <= radius_range[1].to_value(u.R_sun))
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

    def load_pose(self, *args, **kwargs):
        output = super().load_pose(*args, **kwargs)
        # convert image
        output['image'] = output['image'] * self.msb_norm
        if 'density' in output:
            output['density'] = self.convert_column_density(output['density'])
        return output


class PlasmaSuNeRFLoader(SuNeRFLoader):

    def __init__(self, state_path, *args, **kwargs):
        state = torch.load(state_path, map_location=device, weights_only=False)
        self.log_T_range = state['log_T_range']
        super().__init__(state_path, *args, **kwargs)


def _get_scale_from_occ_max(occ_max, distance, resolution):
    occ_max = u.Quantity(occ_max).to(u.R_sun)
    distance = u.Quantity(distance).to(u.R_sun)
    # NumPy/SunPy image shapes are ordered (y, x), while WCS scale is (x, y).
    ny = int(resolution[0].to_value(u.pix))
    nx = int(resolution[1].to_value(u.pix))

    impact_ratio = (occ_max / distance).to_value(u.one)
    if not 0 < impact_ratio < 1:
        raise ValueError("occ_max must be positive and smaller than the observer distance.")

    # FITS TAN uses a gnomonic projection-plane coordinate.  Convert the
    # desired physical impact parameter to its exact sky angle and then to the
    # corresponding tangent-plane radius at the outermost pixel centers.
    sky_radius = np.arcsin(impact_ratio)
    tan_plane_radius = (np.tan(sky_radius) * u.rad).to(u.arcsec)
    x_half_span = max(nx - 1, 1) / 2
    y_half_span = max(ny - 1, 1) / 2

    return u.Quantity(
        [tan_plane_radius / x_half_span, tan_plane_radius / y_half_span]
    ) / u.pix


def _get_mask(s_map, occ_min, occ_max):
    # mask occultor
    img_coords = all_coordinates_from_map(s_map)
    x = img_coords.Tx
    y = img_coords.Ty
    impact_radius = hpc_impact_parameter(x, y, s_map.dsun).to(u.R_sun)

    mask = np.zeros(x.shape, dtype=bool)
    if occ_min is not None:
        occ_min_cond = impact_radius < u.Quantity(occ_min).to(u.R_sun)
        mask[occ_min_cond] = True
    if occ_max is not None:
        occ_max_cond = impact_radius > u.Quantity(occ_max).to(u.R_sun)
        mask[occ_max_cond] = True
    return mask
