import copy
import glob
import multiprocessing
import os
from datetime import timedelta, datetime

import numpy as np
import scipy
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from dateutil.parser import parse
from skimage.morphology import remove_small_objects, binary_opening, disk
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header, all_coordinates_from_map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime, unnormalize_datetime
from sunerf.data.loader.base_loader import BaseDataModule, TensorsDataset, MapDataLoader
from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset
from sunerf.data.ray_sampling import get_rays
from sunerf.train.callback import log_overview
from sunerf.train.coordinate_transformation import spherical_to_cartesian, pose_spherical
from sunerf.train.render_mode import RenderModeDataset, RenderMode


def create_scaling_mask(projected_radius, scaling_mask_config):
    tB_coeffs = np.load(scaling_mask_config['tB_coeffs_file'])
    pB_coeffs = np.load(scaling_mask_config['pB_coeffs_file'])

    tB_fit = np.exp(np.polyval(tB_coeffs, projected_radius))
    pB_fit = np.exp(np.polyval(pB_coeffs, projected_radius))
    mask = np.stack([tB_fit, pB_fit], axis=-1)
    return np.clip(mask, 1e-12, None).astype(np.float32)


class ThomsonDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, Rs_per_ds, seconds_per_dt, ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False,
                 **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size}

        train_dict, ref_date = self._load_dataset(train_datasets, base_config)

        module_config = {}
        for k, train_ds in train_dict.items():
            if not isinstance(train_ds, GenericThomsonDataset):
                continue
            dc = train_ds.data_config
            module_config[k] = {'type': 'thomson', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt,
                                'ref_date': ref_date, 'image_scaling': train_ds.scaling,
                                'wcs': dc['wcs'], 'image_shape': dc['image_shape'], 'times': train_ds.times,
                                'observers': dc['observers']}

        base_config['batch_size'] = validation_batch_size
        times = np.concatenate(
            [dataset.normalized_times for dataset in train_dict.values() if isinstance(dataset, GenericThomsonDataset)])
        time_range = [np.min(times), np.max(times)]
        valid_dict = self._load_valid_dataset(valid_datasets, base_config, time_range=time_range,
                                              seconds_per_dt=seconds_per_dt, ref_date=ref_date)

        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=module_config, **kwargs)

    def _load_dataset(self, data_config, base_config):
        ref_date = None if 'ref_date' not in base_config else base_config['ref_date']
        data_config = copy.deepcopy(data_config)

        train_dict = {}
        for config in data_config:
            config = copy.deepcopy(config)
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)

            if ds_type.lower() == 'hao':
                dataset = HAOThomsonDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'cor':
                dataset = COR2Dataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'lasco':
                dataset = LASCOC2Dataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'metis':
                dataset = MetisDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'ccor':
                dataset = CCORDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() in {'punchwfi', 'punch_wfi', 'punch'}:
                dataset = PunchWFIDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'psi_cme':
                dataset = PSICMEDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'random':
                assert len(
                    train_dict) > 0, 'Specify at least one dataset for reference times. The random dataset configuration needs to be last in config file.'
                times = np.concatenate([dataset.normalized_times for dataset in train_dict.values() if
                                        isinstance(dataset, GenericThomsonDataset)])
                time_range = [np.min(times), np.max(times)]
                radius_range = u.Quantity(ds_config.pop('radius_range'), unit=ds_config.pop('unit', 'AU'))
                dataset = RandomSphericalCoordinateDataset(time_range=time_range, radius_range=radius_range,
                                                           **ds_config)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_date is None:
                ref_date = dataset.ref_date
                base_config['ref_date'] = ref_date
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset
        return train_dict, ref_date

    def _load_valid_dataset(self, data_config, base_config, time_range, seconds_per_dt, ref_date):
        data_config = copy.deepcopy(data_config)

        valid_dict = {}
        for config in data_config:
            config = copy.deepcopy(config)
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)
            if ds_type.lower() == 'hao':
                dataset = HAOThomsonDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'cor':
                dataset = COR2Dataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'lasco':
                dataset = LASCOC2Dataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'metis':
                dataset = MetisDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'ccor':
                dataset = CCORDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() in {'punchwfi', 'punch_wfi', 'punch'}:
                dataset = PunchWFIDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'psi_cme':
                dataset = PSICMEDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.INSTRUMENT)
            elif ds_type.lower() == 'reference_cube':
                dataset = ReferenceCubeDataset(**ds_config, ds_key=ds_key, shuffle=False, filter_nans=False)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.REFERENCE)
            elif ds_type.lower() == "radial_slices":
                dataset = RadialSlicesDataset(**ds_config, ds_key=ds_key, time_range=time_range)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            elif ds_type.lower() == "longitude_slices":
                dataset = LongitudeSlicesDataset(**ds_config, ds_key=ds_key, time_range=time_range)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            elif ds_type.lower() == "fixed_viewpoint_series":
                dataset = FixedViewpointSeriesDataset(**ds_config, ds_key=ds_key, time_range=time_range)
                dataset = RenderModeDataset(dataset, RenderMode.INSTRUMENT)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            assert ds_key not in valid_dict, f'Duplicate dataset key {ds_key}'
            valid_dict[ds_key] = dataset
        return valid_dict


class GenericThomsonDataset(TensorsDataset):
    def __init__(self, data_path_pB, data_path_tB, scaling, ds_key, instrument_key,
                 Rs_per_ds, seconds_per_dt, image_norm=512, hpc_norm=1e4, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, test=False, noise_level=False,
                 reference_frame='inertial', azimuthal_equidistant=True,
                 correction_config=None,
                 scaling_mask_config=None,
                 **kwargs):
        self.scaling = scaling
        # select files with min diff in dates
        tB_files = sorted(glob.glob(data_path_tB))
        pB_files = sorted(glob.glob(data_path_pB)) if data_path_pB is not None else None

        if debug:
            sampling = len(tB_files) // 20
            tB_files = tB_files[::sampling]
            pB_files = pB_files[::sampling] if pB_files is not None else None
        if test:
            # select file at center of the list
            idx = len(tB_files) // 2
            tB_files = tB_files[idx:idx + 1]
            pB_files = pB_files[idx:idx + 1] if pB_files is not None else None

        # load rays
        data_dict = {}
        with multiprocessing.Pool(os.cpu_count()) as p:
            loader = MapDataLoader(Rs_per_ds, reference_frame, azimuthal_equidistant=azimuthal_equidistant)
            data = [v for v in
                    tqdm(p.imap(loader.load, tB_files), total=len(tB_files), desc=f'Loading tB + rays')]
        observers = [d.pop('observer') for d in data]
        for k in data[0].keys():
            data_dict[k] = np.stack([d[k] for d in data], axis=0)
        tB_image_stack = data_dict['image']

        # load remaining images
        if pB_files is None:
            pB_image_stack = np.ones_like(tB_image_stack) * np.nan
        else:
            with multiprocessing.Pool(os.cpu_count()) as p:
                pB_image_stack = [v for v in
                                  tqdm(p.imap(fits.getdata, pB_files), total=len(pB_files), desc=f'Loading pB')]
                pB_image_stack = np.stack(pB_image_stack, axis=0)


        # apply correction if specified
        if correction_config is not None:
            alpha = float(correction_config.get('alpha', 1.0))
            tB_alpha = float(correction_config.get('tB_alpha', alpha))
            pB_alpha = float(correction_config.get('pB_alpha', alpha))
            if correction_config['type'] == 'percentile':
                pB_level = correction_config.get('pB_level', 20)
                tB_level = correction_config.get('tB_level', 20)
                pB_correction = np.percentile(pB_image_stack, pB_level, axis=0, keepdims=True)
                tB_correction = np.percentile(tB_image_stack, tB_level, axis=0, keepdims=True)
                pB_image_stack = pB_image_stack - pB_alpha * pB_correction
                tB_image_stack = tB_image_stack - tB_alpha * tB_correction
            elif correction_config['type'] == 'file':
                pB_path = correction_config.get('pB', None)
                tB_path = correction_config.get('tB', None)
                if pB_path is None and tB_path is None:
                    raise ValueError("correction_config.type='file' requires at least one of 'tB' or 'pB'.")
                if pB_path is not None:
                    pB_correction = np.load(pB_path)
                    pB_image_stack = pB_image_stack - pB_alpha * pB_correction[None]
                if tB_path is not None:
                    tB_correction = np.load(tB_path)
                    tB_image_stack = tB_image_stack - tB_alpha * tB_correction[None]
            elif correction_config['type'] == 'basic':
                pass
            else:
                raise ValueError(f'Unknown correction type {correction_config["type"]}')

            min_value = correction_config.get('min_value', 0)
            pB_below = int(np.count_nonzero(pB_image_stack <= min_value))
            tB_below = int(np.count_nonzero(tB_image_stack <= min_value))
            print(f'Filtering {pB_below} pB pixels and {tB_below} tB pixels below min value {min_value}')
            pB_image_stack[pB_image_stack <= min_value] = np.nan
            tB_image_stack[tB_image_stack <= min_value] = np.nan

            if correction_config.get('clean', False):
                clean_min_size = correction_config.get('clean_min_size', 128)
                opening_radius = correction_config.get('clean_opening_radius', 2)
                footprint = disk(opening_radius) if opening_radius > 0 else None
                # Clean each frame independently; stack-wide cleaning would connect components over time.
                for i in range(tB_image_stack.shape[0]):
                    # clean pB
                    pB_mask = np.isfinite(pB_image_stack[i])
                    if footprint is not None:
                        pB_mask = binary_opening(pB_mask, footprint=footprint)
                    pB_mask_clean = remove_small_objects(pB_mask, min_size=clean_min_size)
                    pB_image_stack[i][~pB_mask_clean] = np.nan
                    # clean tB
                    tB_mask = np.isfinite(tB_image_stack[i])
                    if footprint is not None:
                        tB_mask = binary_opening(tB_mask, footprint=footprint)
                    tB_mask_clean = remove_small_objects(tB_mask, min_size=clean_min_size)
                    tB_image_stack[i][~tB_mask_clean] = np.nan

        # save occultor mask before any correction/cleaning that may set more pixels to NaN
        # this mask is used to set rays/image coords/hpc coords to NaN for occulted pixels,
        # which should be ignored during training and evaluation
        occultor_mask = np.isnan(tB_image_stack) & np.isnan(pB_image_stack)

        image_stack = np.stack([tB_image_stack, pB_image_stack], axis=-1)
        image_stack = image_stack / scaling
        image_stack[image_stack <= 0] = np.nan  # set non-positive values to NaN = unphysical

        if noise_level:
            mean_B = np.nanmean(image_stack)
            noise = np.random.normal(0, 1, size=image_stack.shape).astype(np.float32)
            noise = noise * noise_level * mean_B
            image_stack += noise

        data_dict['image'] = image_stack

        if scaling_mask_config is not None:
            projected_radius = data_dict['projected_radius']
            scaling_mask = create_scaling_mask(projected_radius, scaling_mask_config)
            data_dict['scaling_mask'] = scaling_mask / scaling

        # expand and normalize times
        times = data_dict['time']
        ref_date = min(times) if ref_date is None else ref_date
        self.ref_date = ref_date
        self.times = times
        normalized_times = np.array([normalize_datetime(t, seconds_per_dt, ref_date) for t in times])
        self.normalized_times = normalized_times
        times_arr = np.tile(normalized_times[:, None, None, None], (1, *image_stack.shape[1:3], 1))
        data_dict['time'] = times_arr

        # add hpc coordinates
        hpc_coords = data_dict['hpc_coords']
        hpc_coords[..., :2] /= hpc_norm  # norm angle Tx and Tz
        hpc_coords[..., 2] /= Rs_per_ds  # norm distance by Rs_per_ds
        data_dict['hpc_coords'] = hpc_coords

        # add image coordinates
        ny, nx = image_stack.shape[1], image_stack.shape[2]
        image_coords = np.stack(np.mgrid[:ny, :nx], axis=-1).astype(np.float32)
        # center: y uses ny, x uses nx
        image_coords[..., 0] -= 0.5 * (ny - 1)
        image_coords[..., 1] -= 0.5 * (nx - 1)
        # normalize
        image_coords /= image_norm
        image_coords = image_coords[None, :, :, :].repeat(image_stack.shape[0], axis=0)  # repeat over time
        data_dict['image_coords'] = image_coords

        # apply occultor mask
        data_dict['rays'][occultor_mask] = np.nan
        data_dict['image_coords'][occultor_mask] = np.nan
        data_dict['hpc_coords'][occultor_mask] = np.nan
        if 'scaling_mask' in data_dict:
            data_dict['scaling_mask'][occultor_mask] = np.nan

        if not test:
            cmap = cm.soholasco2.copy()
            cmap.set_bad(color='green')
            log_overview(data_dict["image"] * scaling, data_dict['pose'], normalized_times, cmap, seconds_per_dt, Rs_per_ds,
                         ref_date, ds_key=ds_key)
            print('----- Data Overview -----')
            print(
                f'Image shape: {data_dict["image"].shape}; MIN: {np.nanmin(data_dict["image"])}; MAX: {np.nanmax(data_dict["image"])}')
            print(f'Time shape: {times_arr.shape}; MIN: {np.nanmin(times_arr)}; MAX: {np.nanmax(times_arr)}')

        tensors = {k: v.reshape((-1, *v.shape[3:])) for k, v in data_dict.items() if
                   k in ['image', 'rays', 'time', 'image_coords', 'hpc_coords', 'scaling_mask']}

        # set all values where image (tB) is NaN to NaN --> skip for training
        if not test:
            nan_mask = np.isnan(tensors['image']).all(-1)
            for k, v in tensors.items():
                tensors[k][nan_mask] = np.nan

        # info for plotting
        self.image_shape = image_stack.shape[1:3]

        # add observable information
        for observer in observers:
            observer['observables'] = ['tB', 'pB'] if pB_files is not None else ['tB']

        # data config for model checkpoint
        data_config = {}
        # load reference info
        ref_map = Map(tB_files[0])
        data_config['image_shape'] = ref_map.data.shape
        data_config['wcs'] = ref_map.wcs
        data_config['wavelength'] = ref_map.wavelength
        data_config['observers'] = observers
        self.data_config = data_config

        super().__init__(tensors=tensors, batch_size=batch_size, shuffle=not test, filter_nans=not test,
                         instrument=instrument_key, **kwargs)


class HAOThomsonDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=5e-5, reference_frame='heliographic', azimuthal_equidistant=True, **kwargs)


class COR2Dataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class LASCOC2Dataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class MetisDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class PunchWFIDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class CCORDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class PSICMEDataset(GenericThomsonDataset):

    def __init__(self, **kwargs):
        super().__init__(scaling=1.0e-9, reference_frame='inertial', azimuthal_equidistant=False, **kwargs)


class ReferenceCubeDataset(TensorsDataset):

    def __init__(self, data_path, ref_date, seconds_per_dt, Rs_per_ds, min_radius=30, max_radius=100, **kwargs):
        o = scipy.io.readsav(data_path)
        date0 = parse("2010-04-03T09:04:00.000")
        time = date0 + timedelta(hours=float(o['this_time']))
        time = normalize_datetime(time, seconds_per_dt, ref_date)

        density = o['dens'].astype(np.float32).T
        ph = o['ph1d'].astype(np.float32)
        r = o['r1d'].astype(np.float32)
        th = o['th1d'].astype(np.float32) - np.pi / 2

        # clip radius to 100 Rsun
        mask = (r < max_radius) & (r > min_radius)
        r = r[mask]
        density = density[mask]

        radius, theta, phi, t = np.meshgrid(r, th, ph, np.array([time]), indexing="ij")
        spherical_coords = np.stack([radius, theta, phi], axis=-1)

        cartesian_coords = spherical_to_cartesian(spherical_coords)
        cartesian_coords = cartesian_coords / Rs_per_ds
        x, y, z = cartesian_coords[..., 0], cartesian_coords[..., 1], cartesian_coords[..., 2]

        query_points = np.stack([x, y, z, t], axis=-1, dtype=np.float32)
        query_points = query_points[:, :, :, 0]  # squeeze time dimension

        print('Query points range: ', query_points.reshape(-1, 4).min(0), query_points.reshape(-1, 4).max(0))

        self.cube_shape = query_points.shape[:-1]

        query_points = query_points.reshape(-1, 4)
        density = density.reshape(-1)
        spherical_coords = spherical_coords.reshape(-1, 3)

        tensors = {'query_points': query_points,
                   'spherical_coords': spherical_coords,
                   'rho': density}
        super().__init__(tensors, **kwargs)


class RadialSlicesDataset(TensorsDataset):
    """
    Produces query_points on multiple constant-radius shells AND multiple time steps.

    Dimensions:
      - radii: (Nr,) in R_sun
      - theta: (Ntheta,) latitude in radians [-pi/2, pi/2)
      - phi: (Nphi,) longitude in radians [0, 2pi)
      - time: (Nt,) normalized time steps (default: 5, evenly sampled over time_range)

    Output:
      - query_points: (Nr*Ntheta*Nphi*Nt, 4)  [x,y,z,t] (normalized x/y/z by Rs_per_ds)
      - spherical_coords: (Nr*Ntheta*Nphi*Nt, 3) [r, theta, phi] (r in R_sun)
      - meta.*: radii/theta/phi/times (useful for callbacks)
    """

    def __init__(
            self,
            time_range,
            Rs_per_ds,
            seconds_per_dt, ref_date,
            radii=(5, 7.5, 10, 15, 20),
            Ntheta=180,
            Nphi=360,
            n_times=5,  # default: 5 time steps (rows)
            **kwargs,
    ):
        # --- angular grids ---
        radii = np.asarray(radii, dtype=np.float32)  # (Nr,) in R_sun
        theta = np.linspace(-np.pi / 2, np.pi / 2, int(Ntheta), endpoint=False, dtype=np.float32)  # lat
        phi = np.linspace(0, 2 * np.pi, int(Nphi), endpoint=False, dtype=np.float32)  # lon

        # --- time grid ---
        t0, t1 = float(time_range[0]), float(time_range[1])
        t = np.linspace(t0, t1, int(n_times), dtype=np.float32)

        # coords: (Nr, Ntheta, Nphi, Nt, 4) with [r, th, ph, t]
        # meshgrid order matches indexing="ij"
        rr, th, ph_carr, tt = np.meshgrid(radii, theta, phi, t, indexing="ij")
        spherical_coords = np.stack([rr, th, ph_carr], axis=-1).astype(np.float32)

        # Build a second longitude cube for the actual model query points.
        # Validation plots should remain in Carrington coordinates, while the
        # Cartesian samples must be rotated into the inertial frame at each time.
        ph_query = np.array(ph_carr, copy=True)
        datetimes = [unnormalize_datetime(it, seconds_per_dt, ref_date) for it in t]
        longitudes = convert_carrington_to_inertial(phi * u.rad, datetimes)
        for i, longitude_inertial in enumerate(longitudes):
            ph_query[:, :, :, i] = longitude_inertial

        coords = np.stack([rr, th, ph_query, tt], axis=-1)  # (Nr, Ntheta, Nphi, Nt, 4)

        cart = spherical_to_cartesian(coords[..., :3], np).astype(np.float32)  # (..,3) in R_sun
        cart /= Rs_per_ds  # normalize spatial coords

        query_points = np.concatenate([cart, coords[..., 3:4]], axis=-1).astype(np.float32)  # (..,4)

        # store shapes + meta for callbacks
        self.cube_shape = query_points.shape[:-1]  # (Nr, Ntheta, Nphi, Nt)
        self.radii = radii
        self.theta = theta
        self.phi = phi
        self.times = t

        query_points = query_points.reshape(-1, 4)
        spherical_coords = spherical_coords.reshape(-1, 3)

        tensors = {'query_points': query_points,
                   'spherical_coords': spherical_coords}
        super().__init__(tensors, shuffle=False, filter_nans=False, **kwargs)


class LongitudeSlicesDataset(TensorsDataset):
    """
    Produces query_points on multiple constant-longitude slices AND multiple time steps.

    Dimensions:
      - r:         (Nr,) radius in R_sun
      - latitude:  (Nlat,) in radians [-pi/2, pi/2)
      - longitude: (Nlon_slices,) in radians
                   (default: [0,30,60,90,120,150] deg)
      - time:      (Nt,) normalized time steps

    Output:
      - query_points: (Nr*Nlat*Nlon_slices*Nt, 4)  [x,y,z,t]
                      (x/y/z normalized by Rs_per_ds)
      - spherical_coords: (Nr*Nlat*Nlon_slices*Nt, 3) [r, latitude, longitude]
                          (r in R_sun)
      - meta.*: r/latitude/longitude/times
    """

    def __init__(
            self,
            time_range,
            Rs_per_ds,
            seconds_per_dt,
            ref_date,
            longitude_deg=(0, 30, 60, 90, 120, 150),
            radius_range=(1.5, 15),  # in R_sun
            Nlatitude=360,
            Nradius=180,
            n_times=5,
            **kwargs,
    ):
        # --- radial + angular grids ---
        r0, r1 = float(radius_range[0]), float(radius_range[1])
        r = np.linspace(r0, r1, int(Nradius), dtype=np.float32)  # radius in R_sun

        latitude = np.linspace(
            0,
            2 * np.pi,
            int(Nlatitude),
            endpoint=False,
            dtype=np.float32,
        )  # latitude in radians

        # --- time grid ---
        t0, t1 = float(time_range[0]), float(time_range[1])
        times = np.linspace(t0, t1, int(n_times), dtype=np.float32)

        datetimes = [unnormalize_datetime(t, seconds_per_dt, ref_date) for t in times]
        longitudes = convert_carrington_to_inertial(longitude_deg * u.deg, datetimes)

        # meshgrid: (Nr, Nlat, Nlon_slices, Nt)
        rr, lat, lon_carr, tt = np.meshgrid(
            r, latitude, np.zeros_like(longitude_deg), times, indexing="ij"
        )
        spherical_coords = np.stack([rr, lat, lon_carr], axis=-1).astype(np.float32)

        # Keep metadata longitudes in Carrington, but rotate query points into
        # the inertial frame used by the model at each validation time.
        lon_query = np.array(lon_carr, copy=True)
        for i, longitude_inertial in enumerate(longitudes):
            lon_query[:, :, :, i] = longitude_inertial

        coords = np.stack([rr, lat, lon_query, tt], axis=-1).astype(np.float32)

        cart = spherical_to_cartesian(coords[..., :3], np).astype(np.float32)
        cart /= Rs_per_ds

        query_points = np.concatenate(
            [cart, coords[..., 3:4]], axis=-1
        ).astype(np.float32)

        # meta for callbacks
        self.cube_shape = query_points.shape[:-1]  # (Nr, Nlat, Nlon_slices, Nt)
        self.r = r
        self.latitude = latitude
        self.longitude = longitudes
        self.longitude_deg = longitude_deg
        self.times = times

        query_points = query_points.reshape(-1, 4)
        spherical_coords = spherical_coords.reshape(-1, 3)

        tensors = {
            "query_points": query_points,
            "spherical_coords": spherical_coords,
        }

        super().__init__(tensors, shuffle=False, filter_nans=False, **kwargs)


class FixedViewpointSeriesDataset(TensorsDataset):
    """
    Produces rays for a fixed observer (lat/lon/distance) and multiple times.
    Returns instrument-style batch keys: rays, time, image, image_coords, hpc_coords, instrument
    """

    def __init__(self,
                 instrument_key,
                 lat_deg=0.0,
                 lon_deg=0.0,
                 distance_AU=1.0,
                 n_times=6,
                 time_range=None,  # normalized [tmin,tmax]
                 ref_date=None,
                 seconds_per_dt=1.0,
                 Rs_per_ds=1.0,
                 resolution=(256, 256),
                 scale_arcsec=(2400 / 256, 2400 / 256),
                 image_norm=512,
                 hpc_norm=1e4,
                 **kwargs):

        self.image_shape = tuple(int(x) for x in resolution)
        self.n_times = int(n_times)

        if time_range is None:
            times_norm = np.array([0.0], dtype=np.float32)
        else:
            t0, t1 = float(time_range[0]), float(time_range[1])
            times_norm = np.linspace(t0, t1, self.n_times, dtype=np.float32)

        # build a mock SunPy map just to get pixel coords (Tx,Ty)
        # Note: use some reasonable reference coord; this is only for WCS/pixel geometry.
        # Observer at each time is encoded via pose, not WCS.
        t_ref = ref_date if isinstance(ref_date, datetime) else datetime(2010, 1, 1)
        obs = SkyCoord(0 * u.deg, 0 * u.deg, (distance_AU * u.AU).to(u.solRad),
                       frame=frames.HeliographicStonyhurst, obstime=t_ref)
        reference_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=t_ref, observer=obs,
                                   frame=frames.Helioprojective)

        mock = np.zeros(self.image_shape, dtype=np.float32)
        header = make_fitswcs_header(mock, reference_coord,
                                     scale=[scale_arcsec[0], scale_arcsec[1]] * u.arcsec / u.pix)
        ref_map = Map(mock, header)
        img_coords = all_coordinates_from_map(ref_map).transform_to(frames.Helioprojective)

        # rays for each time
        lat = np.deg2rad(float(lat_deg))
        lon = np.deg2rad(float(lon_deg))
        dist_solRad = (distance_AU * u.AU).to_value(u.solRad)

        rays_all = []
        time_all = []
        img_all = []
        imgcoords_all = []
        hpccoords_all = []

        ny, nx = self.image_shape
        image_coords = np.stack(np.mgrid[:ny, :nx], axis=-1).astype(np.float32)
        image_coords[..., 0] -= 0.5 * (ny - 1)
        image_coords[..., 1] -= 0.5 * (nx - 1)
        image_coords /= float(image_norm)

        hpc_coords = np.zeros((ny, nx, 2), dtype=np.float32) / float(hpc_norm)

        for tnorm in times_norm:
            pose = pose_spherical(lon, lat, dist_solRad / float(Rs_per_ds)).numpy()
            rays_o, rays_d = get_rays(img_coords.Tx, img_coords.Ty, pose)
            rays = np.stack([rays_o, rays_d], axis=-2).astype(np.float32)  # (ny,nx,2,3)

            rays_all.append(rays)
            time_all.append(np.ones((ny, nx, 1), dtype=np.float32) * tnorm)

            # NaN target image (tB,pB); model will still render
            img_all.append(np.ones((ny, nx, 2), dtype=np.float32) * np.nan)

            imgcoords_all.append(image_coords)
            hpccoords_all.append(hpc_coords)

        rays_all = np.stack(rays_all, axis=0)  # (Nt,ny,nx,2,3)
        time_all = np.stack(time_all, axis=0)  # (Nt,ny,nx,1)
        img_all = np.stack(img_all, axis=0)  # (Nt,ny,nx,2)
        imgcoords_all = np.stack(imgcoords_all, 0)  # (Nt,ny,nx,2)
        hpccoords_all = np.stack(hpccoords_all, 0)  # (Nt,ny,nx,2)

        tensors = {
            "rays": torch.from_numpy(rays_all.reshape(-1, 2, 3)),
            "time": torch.from_numpy(time_all.reshape(-1, 1)),
            "image": torch.from_numpy(img_all.reshape(-1, 2)),
            "image_coords": torch.from_numpy(imgcoords_all.reshape(-1, 2)),
            "hpc_coords": torch.from_numpy(hpccoords_all.reshape(-1, 2)),
        }

        super().__init__(tensors=tensors, **kwargs)


def convert_carrington_to_inertial(longitude: list[float], datetimes: list[datetime]) -> list[float]:
    longitudes = []
    for t in datetimes:
        # convert longitude from Carrington to inertial frame
        sky_coords = SkyCoord(lon=longitude, lat=0 * u.deg,
                              radius=1 * u.AU,
                              frame=frames.HeliographicCarrington, observer='self',
                              obstime=t)
        longitude_inertial = sky_coords.transform_to(frames.HeliocentricInertial).lon.to_value(u.rad)
        longitudes.append(longitude_inertial)
    return longitudes
