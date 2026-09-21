import copy
import glob
import multiprocessing
import os
from datetime import timedelta

import numpy as np
import torch
from dateutil.parser import parse
from torch.utils.data import Dataset
from tqdm import tqdm

from sunerf.data.date_util import normalize_datetime
from sunerf.data.euv.observation import (
    discover_prepared_files,
    get_prepared_euv_adapter,
    load_prepared_map,
    match_prepared_channels,
)
from sunerf.data.dataset import TensorsDataset
from sunerf.data.loader.base_loader import BaseDataModule, MapDataLoader
from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset
from sunerf.train.callback import log_overview


def _pool_size(n_items, requested_workers):
    if n_items <= 1:
        return 0
    available = os.cpu_count() or 1
    requested = available if requested_workers is None else max(int(requested_workers), 0)
    return min(n_items, available, requested)


def _observation_split_indices(
        n_observations, *, test, holdout=None, debug=False, debug_max_observations=20
):
    """Return deterministic selected and held-out matched-observation indices."""
    if n_observations <= 0:
        raise ValueError('Cannot split an empty observation sequence.')

    if holdout is None:
        held_out = np.empty(0, dtype=np.int64)
    else:
        if not isinstance(holdout, dict):
            raise ValueError('holdout must be a mapping with strategy and count.')
        unknown = set(holdout) - {'strategy', 'count'}
        if unknown:
            raise ValueError(f'Unknown holdout options: {sorted(unknown)!r}.')
        strategy = holdout.get('strategy', 'center')
        count = int(holdout.get('count', 1))
        if strategy != 'center':
            raise ValueError(f'Unsupported holdout strategy {strategy!r}.')
        if count <= 0 or count >= n_observations:
            raise ValueError(
                f'holdout count must be between 1 and {n_observations - 1}, got {count}.'
            )
        start = (n_observations - count) // 2
        held_out = np.arange(start, start + count, dtype=np.int64)

    if test:
        if holdout is None:
            selected = np.array([n_observations // 2], dtype=np.int64)
        else:
            selected = held_out.copy()
    else:
        selected = np.setdiff1d(
            np.arange(n_observations, dtype=np.int64), held_out, assume_unique=True
        )
        if debug:
            n_debug = min(max(int(debug_max_observations), 1), len(selected))
            positions = np.unique(
                np.linspace(0, len(selected) - 1, n_debug, dtype=np.int64)
            )
            selected = selected[positions]
    return selected, held_out


def _load_euv_observation(payload):
    (paths, channel_ids, instrument_type, instrument_id, Rs_per_ds, max_radius,
     alignment_tolerance_arcsec, strict_metadata) = payload
    maps = [load_prepared_map(path) for path in paths]
    adapter = get_prepared_euv_adapter(instrument_type)
    observation = adapter.prepare(
        maps,
        channel_ids,
        instrument_id=instrument_id,
        source_paths=paths,
        alignment_tolerance_arcsec=alignment_tolerance_arcsec,
        strict_metadata=strict_metadata,
    )
    geometry = MapDataLoader(
        Rs_per_ds=Rs_per_ds,
        max_radius=max_radius,
        reference_frame='carrington',
    ).load(maps[0])
    return {
        'image': np.moveaxis(observation.image, 0, -1),
        'valid_mask': np.moveaxis(observation.valid_mask, 0, -1),
        'rays': geometry['rays'],
        'pose': geometry['pose'],
        # Rays use the validated reference-channel WCS/pose, while temporal
        # supervision represents the complete multi-channel exposure group.
        # Per-channel offsets remain available in observation.metadata().
        'time': observation.obstime,
        'observer': geometry['observer'],
        'wcs': observation.wcs,
        'wavelength': maps[0].wavelength,
        'metadata': observation.metadata(),
    }


def _load_euv_observations(payloads, requested_workers, description):
    workers = _pool_size(len(payloads), requested_workers)
    if workers == 0:
        return [
            _load_euv_observation(payload)
            for payload in tqdm(payloads, desc=description)
        ]
    with multiprocessing.Pool(workers) as pool:
        return [
            value for value in tqdm(
                pool.imap(_load_euv_observation, payloads),
                total=len(payloads),
                desc=description,
            )
        ]


class MultiInstrumentDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, Rs_per_ds=1, seconds_per_dt=86400, ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False, random_config=None, use_absorption=False,
                 num_workers=None, preparation_workers=None, holdout=None,
                 **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size,
                       'load_workers': preparation_workers if preparation_workers is not None else num_workers,
                       'holdout': copy.deepcopy(holdout)}

        train_dict = self._load_dataset(train_datasets, base_config)
        ref_date = base_config['ref_date'] # update ref date if not specified

        def dataset_module_config(ref_ds):
            dc = ref_ds.data_config
            return {
                'type': 'plasma',
                'instrument_key': dc['instrument_key'],
                'Rs_per_ds': Rs_per_ds,
                'seconds_per_dt': seconds_per_dt,
                'ref_date': ref_date,
                'wcs': dc['wcs'],
                'image_shape': dc['image_shape'],
                'times': ref_ds.times,
                'cmaps': dc['cmaps'],
                'channel_ids': dc['channel_ids'],
                'measurement_units': dc['measurement_units'],
                'sensitivity_conventions': dc['sensitivity_conventions'],
                'measurement_semantics': dc['measurement_semantics'],
                'native_pixel_solid_angle_sr': dc['native_pixel_solid_angle_sr'],
                'native_pixel_solid_angle_sr_by_observation': dc[
                    'native_pixel_solid_angle_sr_by_observation'
                ],
                'prepared_schema': dc['prepared_schema'],
            }

        module_config = {}
        for k, ref_ds in train_dict.items():
            module_config[k] = dataset_module_config(ref_ds)

        euv_training_datasets = tuple(train_dict.values())

        # include random sampling if specified
        if random_config is not None:
            times = np.concatenate(
                [dataset.normalized_times for dataset in euv_training_datasets]
            )
            time_range = [np.min(times), np.max(times)]
            random_ds = RandomSphericalCoordinateDataset(
                time_range=time_range, Rs_per_ds=Rs_per_ds, **random_config
            )
            train_dict['random'] = random_ds

        base_config['validation_batch_size'] = validation_batch_size
        valid_dict = self._load_dataset(
            valid_datasets, base_config, test_ds=True
        )
        for k, ref_ds in valid_dict.items():
            if k not in module_config:
                module_config[k] = dataset_module_config(ref_ds)

        if use_absorption:
            valid_dict['absorption'] = AbsorptionTestDataset(batch_size=validation_batch_size)

        super().__init__(train_dict, valid_dict,
                         Rs_per_ds=Rs_per_ds, seconds_per_dt=seconds_per_dt, ref_date=ref_date,
                         module_config=module_config, num_workers=num_workers, **kwargs)

    def _load_dataset(self, data_config, base_config, test_ds=False):
        ref_date = None if 'ref_date' not in base_config else base_config['ref_date']
        data_config = copy.deepcopy(data_config)

        train_dict = {}
        for config in data_config:
            ds_type = config.pop('type')
            ds_key = config.pop('key') if 'key' in config else ds_type
            instrument_key = config.pop('instrument_key') # instrument key is required
            ds_config = copy.deepcopy(base_config)
            ds_config.update(config)
            if 'scaling' in ds_config:
                raise ValueError(
                    f"Dataset '{ds_key}' defines 'scaling'; image divisors belong to "
                    f"instruments[].scaling.divisor of instrument '{instrument_key}'."
                )
            # Configured batch sizes are per rank. DDP already creates one loader
            # per process; multiplying by the host GPU count produces N^2 global
            # scaling and can exhaust memory.
            ds_config['batch_size'] = ds_config['validation_batch_size'] if test_ds else ds_config['batch_size']
            if ds_type == 'AIA':
                dataset = AIADataset(**ds_config, ds_key=ds_key, test=test_ds, instrument_key=instrument_key)
            elif ds_type == 'EUI':
                dataset = EUIDataset(**ds_config, ds_key=ds_key, test=test_ds, instrument_key=instrument_key)
            elif ds_type == 'EUVI':
                dataset = EUVIDataset(**ds_config, ds_key=ds_key, test=test_ds, instrument_key=instrument_key)
            else:
                raise ValueError(f'Unknown dataset type {ds_type}')
            # update ref time
            if ref_date is None:
                ref_date = dataset.ref_date
                base_config['ref_date'] = ref_date
            assert ds_key not in train_dict, f'Duplicate dataset key {ds_key}'
            train_dict[ds_key] = dataset
        return train_dict


class GenericEUVDataset(TensorsDataset):
    def __init__(self, file_dict, date_dict, work_directory, ds_key, instrument_key, Rs_per_ds=1, seconds_per_dt=86400, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, test=False, cmaps=None, static=False, max_radius=None,
                 instrument_type=None, match_tolerance_minutes=2, alignment_tolerance_arcsec=0.25,
                 strict_metadata=False, load_workers=None, debug_max_observations=20,
                 holdout=None, prepared_data_path=None, **kwargs):
        channel_ids = tuple(file_dict)
        instrument_type = instrument_key if instrument_type is None else instrument_type
        matched_files, matched_dates, reference_channel = match_prepared_channels(
            file_dict,
            date_dict,
            tolerance=timedelta(minutes=float(match_tolerance_minutes)),
            channel_order=channel_ids,
        )
        n_matched = len(matched_files[reference_channel])
        n_reference = len(date_dict[reference_channel])

        selection, held_out_indices = _observation_split_indices(
            n_matched,
            test=test,
            holdout=holdout,
            debug=debug,
            debug_max_observations=debug_max_observations,
        )
        holdout_manifest = [
            {
                'matched_index': int(index),
                'source_paths': [
                    str(matched_files[channel][index]) for channel in channel_ids
                ],
                'channel_times': [
                    matched_dates[channel][index].isoformat() for channel in channel_ids
                ],
            }
            for index in held_out_indices
        ]
        matched_files = {
            channel: np.asarray(paths, dtype=object)[selection]
            for channel, paths in matched_files.items()
        }
        matched_dates = {
            channel: np.asarray(dates, dtype=object)[selection]
            for channel, dates in matched_dates.items()
        }
        print(
            f'Using {len(selection)} complete, unique observations out of '
            f'{n_reference} reference-channel observations'
        )
        payloads = [
            (
                tuple(str(matched_files[channel][observation_index]) for channel in channel_ids),
                channel_ids,
                instrument_type,
                instrument_key,
                Rs_per_ds,
                max_radius,
                alignment_tolerance_arcsec,
                strict_metadata,
            )
            for observation_index in range(len(selection))
        ]
        records = _load_euv_observations(
            payloads, load_workers, f'Loading {instrument_key} prepared observations'
        )
        if not records:
            raise ValueError(f'No usable observations found for {ds_key!r}.')

        image_stack = np.stack([record['image'] for record in records], axis=0)
        channel_valid_mask = np.stack(
            [record['valid_mask'] for record in records], axis=0
        )
        rays = np.stack([record['rays'] for record in records], axis=0)
        poses = np.stack([record['pose'] for record in records], axis=0)
        observation_times = [record['time'] for record in records]
        observers = [record['observer'] for record in records]

        # Preserve calibrated signed noise. Clipping valid negative detector
        # rates biases faint optically thin emission upward; only the explicit
        # validity mask decides whether a value is supervised.
        ray_valid_mask = np.all(np.isfinite(rays), axis=(-2, -1))
        channel_valid_mask &= ray_valid_mask[..., None]
        self.channel_ids = tuple(str(channel) for channel in channel_ids)
        # Images stay in the prepared physical units; the loss-space divisor is a
        # fixed property of the instrument (instruments[].scaling.divisor).
        images = np.where(channel_valid_mask, image_stack, 0).astype(np.float32)
        safe_rays = np.where(np.isfinite(rays), rays, 0).astype(np.float32)

        metadata = [record['metadata'] for record in records]
        measurement_units = tuple(metadata[0]['measurement_units'])
        sensitivity_conventions = tuple(metadata[0]['sensitivity_conventions'])
        measurement_semantics = tuple(metadata[0]['measurement_semantics'])
        native_pixel_solid_angle_sr = tuple(
            metadata[0]['native_pixel_solid_angle_sr']
        )
        for record_metadata in metadata[1:]:
            if tuple(record_metadata['measurement_units']) != measurement_units:
                raise ValueError('Prepared EUV measurement units change between observations.')
            if tuple(record_metadata['sensitivity_conventions']) != sensitivity_conventions:
                raise ValueError(
                    'Prepared EUV sensitivity conventions change between observations.'
                )
            if tuple(record_metadata['measurement_semantics']) != measurement_semantics:
                raise ValueError(
                    'Prepared EUV measurement semantics change between observations.'
                )

        cmap_values = ['gray'] * len(channel_ids) if cmaps is None else list(cmaps)
        if len(cmap_values) != len(channel_ids):
            raise ValueError('cmaps must contain one entry per ordered EUV channel.')
        data_config = {
            'instrument_key': instrument_key,
            'image_shape': image_stack.shape[1:3],
            'wcs': records[0]['wcs'][0],
            'wcs_by_channel': records[0]['wcs'],
            'wavelength': records[0]['wavelength'],
            'channel_ids': self.channel_ids,
            'cmaps': cmap_values,
            'measurement_units': measurement_units,
            'sensitivity_conventions': sensitivity_conventions,
            'measurement_semantics': measurement_semantics,
            'native_pixel_solid_angle_sr': native_pixel_solid_angle_sr,
            'native_pixel_solid_angle_sr_by_observation': [
                list(record_metadata['native_pixel_solid_angle_sr'])
                for record_metadata in metadata
            ],
            'prepared_schema': metadata[0]['schema'],
            'prepared_observations': metadata,
            'prepared_data_path': (
                None if prepared_data_path is None
                else str(prepared_data_path)
            ),
            'observers': observers,
            'reference_channel': str(reference_channel),
            'match_tolerance_seconds': float(match_tolerance_minutes) * 60,
            'batch_size_convention': 'per_rank',
            'split': {
                'role': 'validation' if test else 'training',
                'holdout': copy.deepcopy(holdout),
                'selected_matched_indices': selection.astype(int).tolist(),
                'held_out_matched_indices': held_out_indices.astype(int).tolist(),
                'held_out_observations': holdout_manifest,
            },
        }
        self.data_config = data_config

        # set to same time if static
        if static:
            ref_date = min(observation_times) if ref_date is None else ref_date
            model_times = [ref_date] * len(observation_times)
        else:
            model_times = observation_times

        # expand and normalize times
        ref_date = min(model_times) if ref_date is None else ref_date
        self.ref_date = ref_date
        self.times = model_times
        self.observation_times = observation_times
        normalized_times = np.array([
            normalize_datetime(value, seconds_per_dt, ref_date) for value in model_times
        ], dtype=np.float32)
        self.normalized_times = normalized_times
        times_arr = np.broadcast_to(
            normalized_times[:, None, None, None],
            (*images.shape[:-1], 1),
        ).copy()

        if not test:
            overview_images = np.where(channel_valid_mask, images, np.nan)
            log_overview(
                overview_images, poses, normalized_times, cmap_values, seconds_per_dt,
                Rs_per_ds, ref_date, ds_key=ds_key, channel_labels=self.channel_ids,
            )

        tensors = {
            'image': images.reshape(-1, len(channel_ids)),
            'rays': safe_rays.reshape(-1, 2, 3),
            'time': times_arr.reshape(-1, 1),
            'valid_mask': channel_valid_mask.reshape(-1, len(channel_ids)).astype(np.float32),
            'ray_valid': ray_valid_mask.reshape(-1, 1).astype(np.float32),
        }
        training_row_mask = (
            ray_valid_mask.reshape(-1)
            & np.any(channel_valid_mask, axis=-1).reshape(-1)
            & np.isfinite(times_arr.reshape(-1))
        )

        super().__init__(
            tensors=tensors,
            work_directory=work_directory,
            batch_size=batch_size,
            shuffle=not test,
            filter_nans=not test,
            valid_mask=None if test else training_row_mask,
            ds_name=ds_key,
            instrument=instrument_key,
            channel_ids=self.channel_ids,
        )


def _dataset_files(
        data_path, wavelengths, *, instrument_type, spacecraft=None,
):
    if not data_path:
        raise ValueError(f'{instrument_type} requires a prepared FITS data_path.')
    patterns = [data_path] if isinstance(data_path, str) else list(data_path)
    files = sorted({
        os.path.abspath(path)
        for pattern in patterns
        for path in glob.glob(os.path.expanduser(str(pattern)), recursive=True)
        if os.path.isfile(path)
    })
    if not files:
        raise ValueError(f'No prepared FITS files match {data_path!r}.')
    return files


_DISCOVERY_CACHE = {}


def _discover_prepared_files(files, wavelengths):
    # Training and validation datasets point at the same prepared files;
    # reading every FITS header is the dominant cost of the validation
    # datasets, which only load their held-out observations afterwards.
    cache_key = (tuple(files), tuple(wavelengths))
    if cache_key not in _DISCOVERY_CACHE:
        _DISCOVERY_CACHE[cache_key] = discover_prepared_files(files, wavelengths)
    return copy.deepcopy(_DISCOVERY_CACHE[cache_key])


class AIADataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, **kwargs):
        wavelengths = [94, 131, 171, 193, 211, 304, 335] if wavelengths is None else wavelengths
        cmaps_dict = {94: 'sdoaia94', 131: 'sdoaia131', 171: 'sdoaia171', 193: 'sdoaia193',
                      211: 'sdoaia211', 304: 'sdoaia304', 335: 'sdoaia335'}
        cmaps = [cmaps_dict[wl] for wl in wavelengths]

        files = _dataset_files(
            data_path, wavelengths, instrument_type='AIA',
        )

        file_dict, date_dict = _discover_prepared_files(files, wavelengths)
        super().__init__(
            file_dict, date_dict, cmaps=cmaps,
            instrument_type='AIA', prepared_data_path=data_path, **kwargs,
        )


class EUIDataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, **kwargs):
        wavelengths = [174, 304] if wavelengths is None else wavelengths
        cmaps_dict = {174: 'sdoaia171', 304: 'sdoaia304'}
        cmaps = [cmaps_dict[wl] for wl in wavelengths]
        files = _dataset_files(
            data_path, wavelengths, instrument_type='EUI',
        )
        file_dict, date_dict = _discover_prepared_files(files, wavelengths)
        super().__init__(
            file_dict, date_dict, cmaps=cmaps,
            instrument_type='EUI', prepared_data_path=data_path, **kwargs,
        )

class EUVIDataset(GenericEUVDataset):

    def __init__(self, data_path, wavelengths=None, **kwargs):
        wavelengths = [171, 195, 284, 304] if wavelengths is None else wavelengths
        cmaps = {171: 'sdoaia171', 195: 'sdoaia193', 284: 'sdoaia211', 304: 'sdoaia304'}
        cmaps = [cmaps[wl] for wl in wavelengths]

        spacecraft = kwargs.get('instrument_key', '').rsplit('-', 1)[-1].upper()
        spacecraft = spacecraft if spacecraft in {'A', 'B'} else None
        files = _dataset_files(
            data_path, wavelengths, instrument_type='EUVI', spacecraft=spacecraft,
        )

        file_dict, date_dict = _discover_prepared_files(files, wavelengths)
        super().__init__(file_dict, date_dict,
                         cmaps=cmaps,
                         instrument_type='EUVI',
                         prepared_data_path=data_path,
                         **kwargs)
class AbsorptionTestDataset(Dataset):

    def __init__(self, n_logT=100, n_logNe=100, logT_range=(3.7, 8), logNe_range=(-3, 4), batch_size=1024):
        self.data = np.stack(np.meshgrid(np.linspace(*logT_range, n_logT),
                                         np.linspace(*logNe_range, n_logNe), indexing='ij'), -1)
        self.image_shape = (n_logT, n_logNe)
        self.data_tensor = torch.from_numpy(self.data).float().reshape(-1, 2)
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(self.data_tensor.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, idx):
        data = self.data_tensor[idx * self.batch_size: (idx + 1) * self.batch_size]
        return {'log_T': data[:, 0:1], 'log_ne': data[:, 1:2]}
