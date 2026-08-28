import copy
from concurrent.futures import ThreadPoolExecutor
import glob
import multiprocessing
import os
import re
from datetime import timedelta, datetime, timezone

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
from sunerf.data.loader.insitu import PSPDataset, SolarOrbiterDataset
from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset
from sunerf.data.ray_sampling import get_rays
from sunerf.physics.thomson import electron_density_normalization_cm3
from sunerf.train.callback import log_overview
from sunerf.train.coordinate_transformation import spherical_to_cartesian, pose_spherical
from sunerf.train.render_mode import RenderModeDataset, RenderMode


def _available_cpu_count():
    """Return the CPU allocation visible to this process."""
    counts = [os.cpu_count() or 1]
    if hasattr(os, 'sched_getaffinity'):
        try:
            counts.append(len(os.sched_getaffinity(0)))
        except OSError:
            pass
    for variable in ('SLURM_CPUS_PER_TASK', 'PBS_NP'):
        try:
            value = int(os.environ.get(variable, ''))
        except ValueError:
            continue
        if value > 0:
            counts.append(value)
    return max(1, min(counts))


def _pool_size(requested_workers, task_count):
    if task_count < 1:
        return 0
    requested_workers = _available_cpu_count() if requested_workers is None else int(requested_workers)
    if requested_workers < 1:
        return 0
    return min(requested_workers, _available_cpu_count(), task_count)


def _iter_parallel(function, items, workers):
    """Map in input order while respecting the job's CPU allocation."""
    worker_count = _pool_size(workers, len(items))
    if worker_count <= 1:
        return map(function, items), None
    pool = multiprocessing.Pool(worker_count)
    return pool.imap(function, items), pool


def _load_map_stack(files, loader, workers, description):
    """Load maps directly into final stacked arrays instead of retaining a list."""
    if not files:
        raise ValueError(f'No files available for {description}.')

    iterator, pool = _iter_parallel(loader.load, files, workers)
    arrays = {}
    observers = []
    try:
        for index, result in enumerate(tqdm(iterator, total=len(files), desc=description)):
            result = dict(result)
            observers.append(result.pop('observer'))
            for key, value in result.items():
                value_array = np.asarray(value)
                if key not in arrays:
                    arrays[key] = np.empty(
                        (len(files), *value_array.shape),
                        dtype=value_array.dtype,
                    )
                arrays[key][index] = value
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return arrays, observers


def _load_fits_stack(files, workers, description):
    """Load a FITS sequence into one preallocated stack."""
    if not files:
        raise ValueError(f'No files available for {description}.')
    iterator, pool = _iter_parallel(fits.getdata, files, workers)
    stack = None
    try:
        for index, image in enumerate(tqdm(iterator, total=len(files), desc=description)):
            image = np.asarray(image)
            if stack is None:
                stack = np.empty((len(files), *image.shape), dtype=image.dtype)
            elif image.shape != stack.shape[1:]:
                raise ValueError(
                    f'Inconsistent image shape for {files[index]}: '
                    f'{image.shape} != {stack.shape[1:]}'
                )
            stack[index] = image
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return stack


def _fit_radial_mad_scale(image_stack, projected_radius, config):
    """Estimate one positive radial MAD scale from the temporal-mean image."""
    if image_stack.shape != projected_radius.shape or image_stack.ndim != 3:
        raise ValueError("Radial MAD fitting expects matching (frame, y, x) arrays.")

    image_count = np.sum(np.isfinite(image_stack), axis=0)
    mean_image = np.divide(
        np.nansum(image_stack, axis=0),
        image_count,
        out=np.full(image_stack.shape[1:], np.nan, dtype=np.float64),
        where=image_count > 0,
    )
    radius_count = np.sum(np.isfinite(projected_radius), axis=0)
    mean_radius = np.divide(
        np.nansum(projected_radius, axis=0),
        radius_count,
        out=np.full(projected_radius.shape[1:], np.nan, dtype=np.float64),
        where=radius_count > 0,
    )

    valid = np.isfinite(mean_image) & np.isfinite(mean_radius) & (mean_radius > 0)
    if not np.any(valid):
        return None

    radius_values = mean_radius[valid]
    radius_min = float(config.get('radius_min', np.nanmin(radius_values)))
    radius_max = float(config.get('radius_max', np.nanmax(radius_values)))
    if not np.isfinite(radius_min) or not np.isfinite(radius_max) or radius_max <= radius_min:
        return None

    n_bins = int(config.get('n_bins', 96))
    min_samples = int(config.get('min_samples', 128))
    if n_bins < 2:
        raise ValueError("scaling_mask_config.n_bins must be at least 2.")
    if min_samples < 1:
        raise ValueError("scaling_mask_config.min_samples must be positive.")

    use_log_radius = bool(config.get('log_radius', True)) and radius_min > 0
    if use_log_radius:
        edges = np.geomspace(radius_min, radius_max, n_bins + 1)
        centers = np.sqrt(edges[:-1] * edges[1:])
    else:
        edges = np.linspace(radius_min, radius_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])

    trim = config.get('trim_percentiles', [10.0, 90.0])
    if trim is not None:
        if len(trim) != 2 or not 0 <= float(trim[0]) < float(trim[1]) <= 100:
            raise ValueError("scaling_mask_config.trim_percentiles must be two increasing values in [0, 100].")
        trim = (float(trim[0]), float(trim[1]))

    radial_scale = np.full(n_bins, np.nan, dtype=np.float64)
    bin_indices = np.digitize(mean_radius, edges) - 1
    for bin_idx in range(n_bins):
        values = mean_image[valid & (bin_indices == bin_idx)]
        if values.size < min_samples:
            continue
        if trim is not None:
            lower, upper = np.nanpercentile(values, trim)
            values = values[(values >= lower) & (values <= upper)]
        if values.size < max(8, min_samples // 4):
            continue
        center = np.nanmedian(values)
        radial_scale[bin_idx] = 1.4826 * np.nanmedian(np.abs(values - center))

    good = np.isfinite(radial_scale) & (radial_scale > 0)
    if np.count_nonzero(good) < 2:
        return None

    coordinate = np.log(centers) if use_log_radius else centers
    log_scale = np.interp(coordinate, coordinate[good], np.log(radial_scale[good]))

    median_scale = float(np.exp(np.nanmedian(log_scale)))
    floor_fraction = float(config.get('floor_fraction', 1e-3))
    absolute_floor = float(config.get('min_scale', 0.0))
    if floor_fraction < 0 or absolute_floor < 0:
        raise ValueError("scaling-mask floors cannot be negative.")
    scale_floor = max(absolute_floor, floor_fraction * median_scale, np.finfo(np.float32).tiny)

    clipped_radius = np.clip(projected_radius, radius_min, radius_max)
    pixel_coordinate = np.log(clipped_radius) if use_log_radius else clipped_radius
    fitted_scale = np.exp(np.interp(pixel_coordinate, coordinate, log_scale))
    return np.maximum(fitted_scale, scale_floor).astype(np.float32)


def create_scaling_mask(projected_radius, scaling_mask_config, image_stack=None):
    """Create either a legacy polynomial brightness mask or an annular MAD scale mask."""
    mask_type = scaling_mask_config.get('type', 'log_polyfit').lower()
    if mask_type == 'log_polyfit':
        tB_coeffs = np.load(scaling_mask_config['tB_coeffs_file'])
        pB_coeffs = np.load(scaling_mask_config['pB_coeffs_file'])

        tB_fit = np.exp(np.polyval(tB_coeffs, projected_radius))
        pB_fit = np.exp(np.polyval(pB_coeffs, projected_radius))
        mask = np.stack([tB_fit, pB_fit], axis=-1)
        return np.clip(mask, 1e-12, None).astype(np.float32)

    if mask_type != 'radial_mad':
        raise ValueError(f"Unknown scaling mask type: {mask_type}")
    if image_stack is None:
        raise ValueError("radial_mad scaling masks require image_stack.")
    if image_stack.shape[:-1] != projected_radius.shape:
        raise ValueError("image_stack and projected_radius shapes are inconsistent.")

    mask = np.full_like(image_stack, np.nan, dtype=np.float32)
    fallback_scale = None
    for channel_idx in range(image_stack.shape[-1]):
        fitted_scale = _fit_radial_mad_scale(
            image_stack[..., channel_idx],
            projected_radius,
            scaling_mask_config,
        )
        if fitted_scale is not None:
            mask[..., channel_idx] = fitted_scale
            if fallback_scale is None:
                fallback_scale = fitted_scale
        elif fallback_scale is not None:
            mask[..., channel_idx] = fallback_scale
    if fallback_scale is None:
        raise ValueError("Could not estimate a radial MAD scale from any image channel.")
    for channel_idx in range(image_stack.shape[-1]):
        if np.isnan(mask[..., channel_idx]).all():
            mask[..., channel_idx] = fallback_scale
    return mask


def valid_training_rows(tensors):
    """Return rows with an observable target and completely finite geometry."""
    image_finite = np.isfinite(tensors['image'])
    valid = image_finite.any(axis=-1)
    for key in ('rays', 'time', 'image_coords', 'hpc_coords'):
        values = tensors[key]
        valid &= np.isfinite(values).all(axis=tuple(range(1, values.ndim)))
    if 'scaling_mask' in tensors:
        # A missing observable may have a missing scale; every observable channel
        # that contributes to the loss must have a finite normalization.
        valid &= np.all(~image_finite | np.isfinite(tensors['scaling_mask']), axis=-1)
    return valid


class ThomsonDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, Rs_per_ds, seconds_per_dt, ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False,
                 preprocess_workers=None, **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        if preprocess_workers is None:
            preprocess_workers = kwargs.get('num_workers')

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size,
                       'preprocess_workers': preprocess_workers}

        train_dict, ref_date = self._load_dataset(train_datasets, base_config)
        drho_cm3 = base_config.get('drho_cm3')

        module_config = {}
        for k, train_ds in train_dict.items():
            if not isinstance(train_ds, GenericThomsonDataset):
                continue
            dc = train_ds.data_config
            module_config[k] = {'type': 'thomson', 'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt,
                                'ref_date': ref_date, 'image_scaling': train_ds.scaling,
                                'wcs': dc['wcs'], 'image_shape': dc['image_shape'], 'times': train_ds.times,
                                'observers': dc['observers'], 'instrument_key': dc['instrument_key'],
                                'image_norm': dc['image_norm'], 'hpc_norm': dc['hpc_norm'],
                                'reference_frame': dc['reference_frame'],
                                'azimuthal_equidistant': dc['azimuthal_equidistant']}

        base_config['batch_size'] = validation_batch_size
        times = np.concatenate(
            [dataset.normalized_times for dataset in train_dict.values() if isinstance(dataset, GenericThomsonDataset)])
        time_range = [np.min(times), np.max(times)]
        valid_dict = self._load_valid_dataset(valid_datasets, base_config, time_range=time_range,
                                              seconds_per_dt=seconds_per_dt, ref_date=ref_date)

        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=module_config, **kwargs)
        self.drho_cm3 = drho_cm3

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
            elif ds_type.lower() == 'punch':
                dataset = PunchWFIDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'psi_cme':
                dataset = PSICMEDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'psp':
                self._inject_insitu_drho(ds_config, base_config, ds_key)
                dataset = PSPDataset(**ds_config, ds_key=ds_key)
            elif ds_type.lower() == 'solar_orbiter':
                self._inject_insitu_drho(ds_config, base_config, ds_key)
                dataset = SolarOrbiterDataset(**ds_config, ds_key=ds_key)
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
            if isinstance(dataset, GenericThomsonDataset):
                self._set_density_normalization_from_thomson(dataset, base_config)
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset
        return train_dict, ref_date

    @staticmethod
    def _set_density_normalization_from_thomson(dataset, base_config):
        if 'drho_cm3' in base_config:
            return
        base_config['drho_cm3'] = electron_density_normalization_cm3(dataset.scaling, base_config['Rs_per_ds'])

    @staticmethod
    def _inject_insitu_drho(ds_config, base_config, ds_key):
        if base_config.get('drho_cm3') is None:
            raise ValueError(
                f"In-situ dataset '{ds_key}' requires drho_cm3. "
                "Place at least one Thomson imaging dataset before in-situ datasets."
            )
        ds_config['drho_cm3'] = base_config['drho_cm3']

    def _load_valid_dataset(self, data_config, base_config, time_range, seconds_per_dt, ref_date):
        data_config = copy.deepcopy(data_config)

        valid_dict = {}
        for config in data_config:
            config = copy.deepcopy(config)
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            render_mode = config.pop('render_mode', None)
            validation_time = config.pop('validation_time', None)
            validation_time_range = config.pop('validation_time_range', None)
            ds_time_range = time_range
            if validation_time is not None:
                t = normalize_datetime(parse(validation_time), seconds_per_dt, ref_date)
                ds_time_range = [t, t]
            elif validation_time_range is not None:
                ds_time_range = [
                    normalize_datetime(parse(t), seconds_per_dt, ref_date)
                    if isinstance(t, str) else float(t)
                    for t in validation_time_range
                ]
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)
            if ds_type.lower() == 'hao':
                dataset = HAOThomsonDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'cor':
                dataset = COR2Dataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'lasco':
                dataset = LASCOC2Dataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'metis':
                dataset = MetisDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'ccor':
                dataset = CCORDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'punch':
                dataset = PunchWFIDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'psi_cme':
                dataset = PSICMEDataset(**ds_config, ds_key=ds_key, test=True)
                dataset = self._wrap_validation_dataset(dataset, render_mode)
            elif ds_type.lower() == 'reference_cube':
                dataset = ReferenceCubeDataset(**ds_config, ds_key=ds_key, shuffle=False, filter_nans=False)
                dataset = RenderModeDataset(dataset, render_mode=RenderMode.REFERENCE)
            elif ds_type.lower() == "radial_slices":
                dataset = RadialSlicesDataset(**ds_config, ds_key=ds_key, time_range=ds_time_range)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            elif ds_type.lower() == "longitude_slices":
                dataset = LongitudeSlicesDataset(**ds_config, ds_key=ds_key, time_range=ds_time_range)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            elif ds_type.lower() == "fixed_viewpoint_series":
                dataset = FixedViewpointSeriesDataset(**ds_config, ds_key=ds_key, time_range=ds_time_range)
                dataset = RenderModeDataset(dataset, RenderMode.INSTRUMENT)
            elif ds_type.lower() == "full_star_background":
                dataset = FullStarBackgroundDataset(**ds_config, ds_key=ds_key, time_range=ds_time_range)
                dataset = RenderModeDataset(dataset, RenderMode.BACKGROUND)
            elif ds_type.lower() == 'psp':
                self._inject_insitu_drho(ds_config, base_config, ds_key)
                dataset = PSPDataset(**ds_config, ds_key=ds_key, shuffle=False, filter_nans=False)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            elif ds_type.lower() == 'solar_orbiter':
                self._inject_insitu_drho(ds_config, base_config, ds_key)
                dataset = SolarOrbiterDataset(**ds_config, ds_key=ds_key, shuffle=False, filter_nans=False)
                dataset = RenderModeDataset(dataset, RenderMode.QUERY_POINTS)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            assert ds_key not in valid_dict, f'Duplicate dataset key {ds_key}'
            valid_dict[ds_key] = dataset
        return valid_dict

    @staticmethod
    def _wrap_validation_dataset(dataset, render_mode=None):
        if render_mode is None:
            mode = RenderMode.INSTRUMENT
        elif isinstance(render_mode, str):
            try:
                mode = RenderMode[render_mode.strip().upper()]
            except KeyError as exc:
                valid_modes = ", ".join(mode.name.lower() for mode in RenderMode)
                raise ValueError(f"Unknown render_mode '{render_mode}'. Expected one of: {valid_modes}") from exc
        else:
            mode = RenderMode(int(render_mode))

        return RenderModeDataset(dataset, render_mode=mode)


class GenericThomsonDataset(TensorsDataset):
    DATE_OBS_KEYS = ("DATE-OBS", "DATE_OBS", "DATE-BEG", "DATE_BEG", "DATE-AVG", "DATE_AVG")

    @staticmethod
    def _normalize_for_time_range(value):
        value = parse(value) if isinstance(value, str) else value
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    @classmethod
    def _parse_time_range(cls, time_range):
        if time_range is None:
            return None
        if isinstance(time_range, (str, datetime)):
            return "closest", cls._normalize_for_time_range(time_range)
        if len(time_range) == 1:
            return "closest", cls._normalize_for_time_range(time_range[0])
        if len(time_range) != 2:
            raise ValueError("time_range must contain one value for closest-date selection "
                             "or two values for start/end filtering.")
        start, end = [cls._normalize_for_time_range(t) for t in time_range]
        if start > end:
            raise ValueError("time_range start must be earlier than or equal to end.")
        return "range", start, end

    @classmethod
    def _read_obs_date(cls, file_path):
        headers = []
        for ext in (0, 1):
            try:
                headers.append(fits.getheader(file_path, ext))
            except Exception:
                continue
        for header in headers:
            for key in cls.DATE_OBS_KEYS:
                if key in header and header[key] not in (None, ""):
                    return cls._normalize_for_time_range(str(header[key]).strip())
        raise KeyError(f"Missing observation date in {file_path}. Expected one of: {', '.join(cls.DATE_OBS_KEYS)}")

    @classmethod
    def _pair_files_by_observation_time(cls, tB_files, pB_files, tolerance_seconds=1.0,
                                        workers=None):
        """Order tB/pB inputs by header time and reject silent mispairing."""
        if pB_files is not None and len(tB_files) != len(pB_files):
            raise ValueError(
                f"Found {len(tB_files)} tB files and {len(pB_files)} pB files; "
                "each brightness image must have one polarization partner."
            )
        if tolerance_seconds < 0:
            raise ValueError("pairing_tolerance_seconds must be non-negative.")

        all_files = [*tB_files, *(pB_files or [])]
        worker_count = _pool_size(workers, len(all_files))
        if worker_count <= 1:
            all_dates = list(map(cls._read_obs_date, all_files))
        else:
            # Header reads are small and I/O-bound. Threads avoid transferring
            # FITS metadata through another process pool during pair validation.
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                all_dates = list(executor.map(cls._read_obs_date, all_files))
        split = len(tB_files)
        tB_records = sorted(zip(all_dates[:split], tB_files))
        pB_records = (
            sorted(zip(all_dates[split:], pB_files))
            if pB_files is not None else None
        )
        for label, records in (('tB', tB_records), ('pB', pB_records)):
            if records is None:
                continue
            dates = [date for date, _ in records]
            if len(set(dates)) != len(dates):
                raise ValueError(
                    f"Duplicate {label} observation timestamps make pairing ambiguous."
                )
        if pB_records is None:
            return [path for _, path in tB_records], None

        paired_tB = []
        paired_pB = []
        for (tB_time, tB_path), (pB_time, pB_path) in zip(tB_records, pB_records):
            separation = abs((tB_time - pB_time).total_seconds())
            if separation > tolerance_seconds:
                raise ValueError(
                    "Could not pair tB and pB sequences by observation time: "
                    f"{os.path.basename(tB_path)} ({tB_time.isoformat()}) versus "
                    f"{os.path.basename(pB_path)} ({pB_time.isoformat()}), "
                    f"separated by {separation:.3f} s."
                )
            paired_tB.append(tB_path)
            paired_pB.append(pB_path)
        return paired_tB, paired_pB

    @classmethod
    def _filter_files_by_time_range(cls, tB_files, pB_files, time_range):
        parsed_range = cls._parse_time_range(time_range)
        if parsed_range is None:
            return tB_files, pB_files

        if len(tB_files) == 0:
            raise ValueError("No tB files found.")
        if pB_files is not None and len(pB_files) != len(tB_files):
            raise ValueError(f"Cannot apply time_range filter to {len(tB_files)} tB files and "
                             f"{len(pB_files)} pB files. File counts must match.")

        mode = parsed_range[0]
        if mode == "closest":
            target = parsed_range[1]
            obs_dates = [cls._read_obs_date(tB_file) for tB_file in tB_files]
            idx = int(np.argmin([abs((obs_date - target).total_seconds()) for obs_date in obs_dates]))
            filtered_tB_files = [tB_files[idx]]
            filtered_pB_files = [pB_files[idx]] if pB_files is not None else None
            print(f"Selected closest tB file to {target.isoformat()}: "
                  f"{os.path.basename(tB_files[idx])} at {obs_dates[idx].isoformat()}.")
            if pB_files is not None:
                pB_date = cls._read_obs_date(pB_files[idx])
                print(f"Selected paired pB file: {os.path.basename(pB_files[idx])} "
                      f"at {pB_date.isoformat()}.")
            return filtered_tB_files, filtered_pB_files

        start, end = parsed_range[1:]
        filtered_tB_files = []
        filtered_pB_files = [] if pB_files is not None else None
        paired_files = zip(tB_files, pB_files) if pB_files is not None else ((f, None) for f in tB_files)
        for tB_file, pB_file in paired_files:
            obs_date = cls._read_obs_date(tB_file)
            if start <= obs_date <= end:
                filtered_tB_files.append(tB_file)
                if filtered_pB_files is not None:
                    filtered_pB_files.append(pB_file)

        if len(filtered_tB_files) == 0:
            raise ValueError(f"No tB files found in time_range {start.isoformat()} to {end.isoformat()}.")
        print(f"Selected {len(filtered_tB_files)} of {len(tB_files)} tB files in time_range "
              f"{start.isoformat()} to {end.isoformat()}.")
        return filtered_tB_files, filtered_pB_files

    @staticmethod
    def _parse_cadence(cadence):
        if cadence is None:
            return None
        if isinstance(cadence, timedelta):
            cadence_delta = cadence
        elif isinstance(cadence, (int, float)):
            cadence_delta = timedelta(seconds=float(cadence))
        elif isinstance(cadence, str):
            match = re.fullmatch(r"(?i)\s*(\d+)\s*([smhd])\s*", cadence)
            if match is None:
                raise ValueError(f"Invalid cadence '{cadence}'. Use formats like 30s, 15m, 1h, or 1d.")
            quantity = int(match.group(1))
            unit = match.group(2).lower()
            seconds_per_unit = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
            cadence_delta = timedelta(seconds=quantity * seconds_per_unit)
        else:
            raise TypeError("cadence must be None, a duration string, seconds, or datetime.timedelta.")
        if cadence_delta.total_seconds() <= 0:
            raise ValueError("cadence must be positive.")
        return cadence_delta

    @classmethod
    def _sample_files_at_cadence(cls, tB_files, pB_files, cadence):
        cadence = cls._parse_cadence(cadence)
        if cadence is None:
            return tB_files, pB_files

        sampled_tB_files = []
        sampled_pB_files = [] if pB_files is not None else None
        next_time = None
        paired_files = zip(tB_files, pB_files) if pB_files is not None else ((f, None) for f in tB_files)
        for tB_file, pB_file in paired_files:
            obs_date = cls._read_obs_date(tB_file)
            if next_time is not None and obs_date < next_time:
                continue
            sampled_tB_files.append(tB_file)
            if sampled_pB_files is not None:
                sampled_pB_files.append(pB_file)
            next_time = obs_date + cadence

        if len(sampled_tB_files) == 0:
            raise ValueError(f"No tB files remain after cadence sampling with cadence {cadence}.")
        print(f"Cadence sampling kept {len(sampled_tB_files)} of {len(tB_files)} tB files "
              f"at {cadence} spacing.")
        return sampled_tB_files, sampled_pB_files

    def __init__(self, data_path_pB, data_path_tB, scaling, ds_key, instrument_key,
                 Rs_per_ds, seconds_per_dt, image_norm=512, hpc_norm=1e4, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, test=False, noise_level=False,
                 reference_frame='inertial', azimuthal_equidistant=True,
                 correction_config=None,
                 scaling_mask_config=None,
                 time_range=None,
                 cadence=None,
                 pairing_tolerance_seconds=1.0,
                 shuffle=None,
                 filter_nans=None,
                 preprocess_workers=None,
                 log_data_overview=True,
                 **kwargs):
        self.scaling = scaling
        self.instrument_key = instrument_key
        self.image_norm = image_norm
        self.hpc_norm = hpc_norm
        self.reference_frame = reference_frame
        self.azimuthal_equidistant = azimuthal_equidistant
        # select files with min diff in dates
        tB_files = sorted(glob.glob(data_path_tB))
        pB_files = sorted(glob.glob(data_path_pB)) if data_path_pB is not None else None
        tB_files, pB_files = self._pair_files_by_observation_time(
            tB_files, pB_files,
            tolerance_seconds=float(pairing_tolerance_seconds),
            workers=preprocess_workers,
        )
        tB_files, pB_files = self._filter_files_by_time_range(tB_files, pB_files, time_range)
        tB_files, pB_files = self._sample_files_at_cadence(tB_files, pB_files, cadence)

        if debug:
            sampling = max(len(tB_files) // 20, 1)
            tB_files = tB_files[::sampling]
            pB_files = pB_files[::sampling] if pB_files is not None else None
        if test:
            # select file at center of the list
            idx = len(tB_files) // 2
            tB_files = tB_files[idx:idx + 1]
            pB_files = pB_files[idx:idx + 1] if pB_files is not None else None

        # load rays
        loader = MapDataLoader(Rs_per_ds, reference_frame, azimuthal_equidistant=azimuthal_equidistant)
        data_dict, observers = _load_map_stack(
            tB_files, loader, preprocess_workers, 'Loading tB + rays'
        )
        tB_image_stack = data_dict['image']

        # load remaining images
        if pB_files is None:
            pB_image_stack = np.ones_like(tB_image_stack) * np.nan
        else:
            pB_image_stack = _load_fits_stack(
                pB_files, preprocess_workers, 'Loading pB'
            )


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

            min_value = correction_config.get('min_value', None)
            if min_value is not None:
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
        occultor_mask = ~np.isfinite(tB_image_stack) & ~np.isfinite(pB_image_stack)

        image_stack = np.stack([tB_image_stack, pB_image_stack], axis=-1)
        image_stack = image_stack / scaling
        # image_stack[image_stack <= 0] = np.nan  # set non-positive values to NaN = unphysical

        if noise_level:
            mean_B = np.nanmean(image_stack)
            noise = np.random.normal(0, 1, size=image_stack.shape).astype(np.float32)
            noise = noise * noise_level * mean_B
            image_stack += noise

        data_dict['image'] = image_stack
        # The combined float32 stack is now authoritative; release the separate
        # channel stacks before allocating time/image-coordinate tensors.
        del tB_image_stack, pB_image_stack

        if scaling_mask_config is not None:
            projected_radius = data_dict['projected_radius']
            scaling_mask = create_scaling_mask(
                projected_radius,
                scaling_mask_config,
                image_stack=image_stack,
            )
            if scaling_mask_config.get('type', 'log_polyfit').lower() == 'log_polyfit':
                scaling_mask = scaling_mask / scaling
            data_dict['scaling_mask'] = scaling_mask
            del projected_radius
        data_dict.pop('projected_radius', None)

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
        del occultor_mask

        if log_data_overview and not test:
            cmap = cm.soholasco2.copy()
            cmap.set_bad(color='green')
            if 'scaling_mask' in data_dict:
                overview_images = data_dict['image'] / data_dict['scaling_mask']
                overview_asinh_a = scaling_mask_config.get('overview_asinh_a')
                if overview_asinh_a is not None:
                    overview_asinh_a = float(overview_asinh_a)
                    if overview_asinh_a <= 0:
                        raise ValueError("scaling_mask_config.overview_asinh_a must be positive.")
                    overview_images = (
                        np.arcsinh(overview_images / overview_asinh_a)
                        / np.arcsinh(1.0 / overview_asinh_a)
                    )
                    overview_mode = 'radially adjusted asinh brightness'
                else:
                    overview_mode = 'radially adjusted brightness'
            else:
                overview_images = data_dict['image'] * scaling
                overview_mode = 'physical brightness'
            log_overview(
                overview_images,
                data_dict['pose'],
                normalized_times,
                cmap,
                seconds_per_dt,
                Rs_per_ds,
                ref_date,
                ds_key=ds_key,
                brightness_mode=overview_mode,
            )
            del overview_images
        if log_data_overview and not test:
            print('----- Data Overview -----')
            print(
                f'Image shape: {data_dict["image"].shape}; MIN: {np.nanmin(data_dict["image"])}; MAX: {np.nanmax(data_dict["image"])}')
            print(f'Time shape: {times_arr.shape}; MIN: {np.nanmin(times_arr)}; MAX: {np.nanmax(times_arr)}')

        data_dict.pop('pose', None)
        tensors = {k: v.reshape((-1, *v.shape[3:])) for k, v in data_dict.items() if
                   k in ['image', 'rays', 'time', 'image_coords', 'hpc_coords', 'scaling_mask']}

        # FITS and correction pipelines can contain +/-Inf as well as NaN. Store
        # one canonical invalid representation so neither cache filtering nor
        # torch loss masks can accidentally treat infinity as a measurement.
        for values in tensors.values():
            values[~np.isfinite(values)] = np.nan

        # An explicit mask avoids writing NaNs through every dense tensor merely
        # so the cache writer can discover the same invalid image rows again.
        valid_mask = None if test else valid_training_rows(tensors)

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
        data_config['instrument_key'] = instrument_key
        data_config['image_norm'] = image_norm
        data_config['hpc_norm'] = hpc_norm
        data_config['reference_frame'] = reference_frame
        data_config['azimuthal_equidistant'] = azimuthal_equidistant
        data_config['scaling_mask_config'] = copy.deepcopy(scaling_mask_config)
        self.data_config = data_config
        dataset_shuffle = (not test) if shuffle is None else shuffle
        dataset_filter_nans = (not test) if filter_nans is None else filter_nans
        dataset_kwargs = {'instrument': instrument_key, **kwargs}
        super().__init__(tensors=tensors, batch_size=batch_size, shuffle=dataset_shuffle, filter_nans=dataset_filter_nans,
                         valid_mask=valid_mask, **dataset_kwargs)


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
      - latitude:  (Nlat,) in radians [0, 2pi)
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
        )  # full angular range for validation slice visualization
        longitude_carrington = np.deg2rad(np.asarray(longitude_deg, dtype=np.float32))

        # --- time grid ---
        t0, t1 = float(time_range[0]), float(time_range[1])
        times = np.linspace(t0, t1, int(n_times), dtype=np.float32)

        datetimes = [unnormalize_datetime(t, seconds_per_dt, ref_date) for t in times]
        longitudes = convert_carrington_to_inertial(longitude_carrington * u.rad, datetimes)

        # meshgrid: (Nr, Nlat, Nlon_slices, Nt)
        rr, lat, lon_carr, tt = np.meshgrid(
            r, latitude, longitude_carrington, times, indexing="ij"
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

        hpc_coords = np.stack([
            img_coords.Tx.to_value(u.arcsec) / float(hpc_norm),
            img_coords.Ty.to_value(u.arcsec) / float(hpc_norm),
            np.full((ny, nx), dist_solRad / float(Rs_per_ds), dtype=np.float32),
        ], axis=-1).astype(np.float32)

        for tnorm in times_norm:
            pose = pose_spherical(lon, lat, dist_solRad / float(Rs_per_ds))
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
        hpccoords_all = np.stack(hpccoords_all, 0)  # (Nt,ny,nx,3)

        tensors = {
            "rays": torch.from_numpy(rays_all.reshape(-1, 2, 3)),
            "time": torch.from_numpy(time_all.reshape(-1, 1)),
            "image": torch.from_numpy(img_all.reshape(-1, 2)),
            "image_coords": torch.from_numpy(imgcoords_all.reshape(-1, 2)),
            "hpc_coords": torch.from_numpy(hpccoords_all.reshape(-1, 3)),
        }

        kwargs.pop('shuffle', None)
        kwargs.pop('filter_nans', None)
        super().__init__(tensors=tensors, shuffle=False, filter_nans=False, **kwargs)


class FullStarBackgroundDataset(TensorsDataset):
    """
    Produces a full 4pi latitude/longitude grid of inertial sky directions for
    evaluating the star background module directly.

    Output:
      - rays: (Nlat*Nlon, 2, 3) with zero origins and unit directions
      - time: (Nlat*Nlon, 1) constant normalized time (midpoint of time_range)
    """

    def __init__(self,
                 instrument_key,
                 time_range=None,
                 Nlat=181,
                 Nlon=360,
                 **kwargs):
        lat = np.linspace(-np.pi / 2, np.pi / 2, int(Nlat), endpoint=True, dtype=np.float32)
        lon = np.linspace(0, 2 * np.pi, int(Nlon), endpoint=False, dtype=np.float32)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

        cos_lat = np.cos(lat_grid)
        rays_d = np.stack([
            cos_lat * np.cos(lon_grid),
            cos_lat * np.sin(lon_grid),
            np.sin(lat_grid),
        ], axis=-1).astype(np.float32)
        rays_o = np.zeros_like(rays_d, dtype=np.float32)
        rays = np.stack([rays_o, rays_d], axis=-2)

        t_value = 0.0 if time_range is None else 0.5 * (float(time_range[0]) + float(time_range[1]))
        time = np.full((*lat_grid.shape, 1), t_value, dtype=np.float32)

        self.sky_shape = lat_grid.shape
        self.image_shape = self.sky_shape
        self.latitude = lat
        self.longitude = lon

        tensors = {
            "rays": rays.reshape(-1, 2, 3),
            "time": time.reshape(-1, 1),
        }
        super().__init__(tensors=tensors, shuffle=False, filter_nans=False, instrument=instrument_key, **kwargs)


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
