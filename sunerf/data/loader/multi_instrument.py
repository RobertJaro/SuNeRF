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
from sunerf.data.euv.observation import _utc_naive
from sunerf.data.euv.observation import (
    discover_prepared_files,
    get_prepared_euv_adapter,
    load_prepared_observation,
    match_prepared_channels,
)
from sunerf.data.dataset import ObservationCacheWriter, TensorsDataset, shuffled_valid_rows
from sunerf.data.loader.base_loader import BaseDataModule, MapDataLoader
from sunerf.data.loader.volume_sampling import RandomSphericalCoordinateDataset
from sunerf.train.callback import log_overview


def _pool_size(n_items, requested_workers):
    if n_items <= 1:
        return 0
    available = os.cpu_count() or 1
    requested = available if requested_workers is None else max(int(requested_workers), 0)
    return min(n_items, available, requested)


def _iso(value):
    """UTC-naive ISO time of a log line, ``open`` for a missing window bound."""
    return 'open' if value is None else _utc_naive(value).isoformat(timespec='seconds')


def _time_window_indices(dates, start=None, end=None):
    """Return the indices of ``dates`` inside ``[start, end)``.

    Either bound may be ``None``; the bounds are ISO strings or datetimes and
    are compared as UTC-naive values like the prepared observation times.
    """
    dates = np.asarray(dates, dtype=object)
    keep = np.ones(dates.shape, dtype=bool)
    if start is not None:
        start = _utc_naive(start)
        keep &= np.asarray([value >= start for value in dates], dtype=bool)
    if end is not None:
        end = _utc_naive(end)
        keep &= np.asarray([value < end for value in dates], dtype=bool)
    if start is not None and end is not None and start >= end:
        raise ValueError(f'start {start.isoformat()} must be earlier than end {end.isoformat()}.')
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError(
            f'No observations between {start!r} and {end!r} '
            f'({len(dates)} matched observations).'
        )
    return indices


def _cadence_indices(dates, complete, cadence_minutes):
    """Return the indices of time-ordered ``dates`` thinned to a fixed cadence.

    Every observation is assigned to the nearest point of a grid with spacing
    ``cadence_minutes`` that starts at the first observation, so timing jitter
    of a few seconds does not drop observations of a sequence already at that
    cadence. Each grid point keeps one observation: one with all channels
    (``complete``) if there is one, and of those the one closest to the point.
    """
    cadence = float(cadence_minutes) * 60
    origin = _utc_naive(dates[0])
    best = {}
    for index, value in enumerate(dates):
        offset = (_utc_naive(value) - origin).total_seconds() / cadence
        slot = int(round(offset))
        rank = (not complete[index], abs(offset - slot))
        if slot not in best or rank < best[slot][0]:
            best[slot] = (rank, index)
    return np.asarray(sorted(index for _, index in best.values()), dtype=np.int64)


def _select_observations(dates, complete, *, test, time=None, debug=False, debug_max_observations=20):
    """Return the matched-observation indices of one dataset.

    Training uses every observation (``debug`` subsamples them evenly).
    Validation renders the one observation with all channels present that lies
    closest to ``time``, or closest to the middle of the sequence without one.
    """
    n_observations = len(dates)
    if n_observations == 0:
        raise ValueError('Cannot select from an empty observation sequence.')
    if not test:
        if not debug:
            return np.arange(n_observations, dtype=np.int64)
        n_debug = min(max(int(debug_max_observations), 1), n_observations)
        return np.unique(np.linspace(0, n_observations - 1, n_debug, dtype=np.int64))

    candidates = np.flatnonzero(np.asarray(complete, dtype=bool))
    if candidates.size == 0:
        raise ValueError('No observation with all channels to validate on.')
    if time is None:
        offsets = np.abs(candidates - n_observations // 2)
    else:
        target = _utc_naive(time)
        offsets = [abs((_utc_naive(dates[index]) - target).total_seconds()) for index in candidates]
    return candidates[[int(np.argmin(offsets))]].astype(np.int64)


def _load_euv_observation(payload):
    (paths, channel_ids, instrument_type, instrument_id, Rs_per_ds, max_radius_rsun,
     alignment_tolerance_arcsec, strict_metadata, saturation_dn) = payload
    # Image and VALID_MASK come from one open per file; the adapter must not
    # reopen the files for their masks.
    # A None path is a channel without a prepared file at this time step; the
    # adapter turns it into a NaN frame with an all-false valid mask.
    loaded = [
        (None, None) if path is None
        else load_prepared_observation(
            path, required_mask=strict_metadata, saturation_dn=saturation_dn
        )
        for path in paths
    ]
    maps = [s_map for s_map, _ in loaded]
    reference_map = next(s_map for s_map in maps if s_map is not None)
    adapter = get_prepared_euv_adapter(instrument_type)
    observation = adapter.prepare(
        maps,
        channel_ids,
        instrument_id=instrument_id,
        source_paths=paths,
        alignment_tolerance_arcsec=alignment_tolerance_arcsec,
        strict_metadata=strict_metadata,
        valid_masks=[valid_mask for _, valid_mask in loaded],
    )
    geometry = MapDataLoader(
        Rs_per_ds=Rs_per_ds,
        max_radius=max_radius_rsun,
        reference_frame='carrington',
    ).load(reference_map)
    return {
        'image': np.moveaxis(observation.image, 0, -1),
        'valid_mask': np.moveaxis(observation.valid_mask, 0, -1),
        'rays': geometry['rays'],
        'pose': geometry['pose'],
        # Rays use the WCS/pose of the first present channel (all present
        # channels are validated to be co-registered), while temporal
        # supervision represents the multi-channel exposure group. Per-channel
        # offsets remain available in observation.metadata().
        'time': observation.obstime,
        'observer': geometry['observer'],
        'wcs': observation.wcs,
        'wavelength': reference_map.wavelength,
        'metadata': observation.metadata(),
    }


def _load_euv_rows(payload):
    """Load one observation and reduce it to its cache rows in a worker.

    The worker keeps only the rows the cache needs: for training, the valid
    rows in random order (``seed``), so the main process never holds a full
    frame; the frame itself is returned only when it is wanted for the
    overview figure.
    """
    observation_payload, training, seed, keep_frame = payload
    record = _load_euv_observation(observation_payload)
    image = np.asarray(record['image'], dtype=np.float32)
    channel_valid_mask = np.asarray(record['valid_mask'], dtype=bool)
    rays = np.asarray(record['rays'], dtype=np.float32)
    # The adapter already marks non-finite and non-positive pixels invalid per
    # channel; only the validity mask decides whether a value is supervised. A
    # pixel row is cached when any channel is valid and dropped when none is.
    ray_valid_mask = np.all(np.isfinite(rays), axis=(-2, -1))
    channel_valid_mask &= ray_valid_mask[..., None]
    n_channels = image.shape[-1]
    # Images stay in the prepared physical units; the loss-space divisor is a
    # fixed property of the instrument (instruments[].scaling.divisor).
    rows = {
        'image': np.where(channel_valid_mask, image, 0).reshape(-1, n_channels),
        'rays': np.where(np.isfinite(rays), rays, 0).reshape(-1, 2, 3),
        'valid_mask': channel_valid_mask.reshape(-1, n_channels).astype(np.float32),
        'ray_valid': ray_valid_mask.reshape(-1, 1).astype(np.float32),
    }
    if training:
        training_rows = ray_valid_mask.reshape(-1) & np.any(channel_valid_mask, axis=-1).reshape(-1)
        indices = shuffled_valid_rows(rows, training_rows, rng=np.random.default_rng(seed))
        rows = {key: value[indices] for key, value in rows.items()}
    return {
        'rows': rows,
        'image_shape': image.shape,
        'frame': np.where(channel_valid_mask, image, np.nan) if keep_frame else None,
        'pose': record['pose'],
        'time': record['time'],
        'observer': record['observer'],
        'wcs': record['wcs'],
        'wavelength': record['wavelength'],
        'metadata': record['metadata'],
    }


def _iter_euv_rows(payloads, requested_workers, description):
    """Yield reduced observations in payload order without retaining them."""
    workers = _pool_size(len(payloads), requested_workers)
    if workers == 0:
        for payload in tqdm(payloads, desc=description):
            yield _load_euv_rows(payload)
        return
    with multiprocessing.Pool(workers) as pool:
        yield from tqdm(
            pool.imap(_load_euv_rows, payloads),
            total=len(payloads),
            desc=description,
        )


class MultiInstrumentDataModule(BaseDataModule):

    def __init__(self, train_datasets, valid_datasets, work_directory, Rs_per_ds=1, seconds_per_dt=86400, ref_date=None,
                 batch_size=int(2 ** 10), validation_batch_size=int(2 ** 11), debug=False, random_config=None, use_absorption=False,
                 num_workers=4, preparation_workers=None,
                 **kwargs):
        os.makedirs(work_directory, exist_ok=True)

        ref_date = parse(ref_date) if ref_date is not None else None  # parse ref time if specified
        base_config = {'Rs_per_ds': Rs_per_ds, 'seconds_per_dt': seconds_per_dt, 'ref_date': ref_date,
                       'debug': debug, 'work_directory': work_directory, 'batch_size': batch_size,
                       'load_workers': preparation_workers}

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
                # Carrington observer position and time of every observation (as the
                # Thomson save state); the per-observation prepared metadata with
                # source paths stays in the dataset and is not published.
                'observers': dc['observers'],
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


_PER_CHANNEL_METADATA = (
    'measurement_units', 'sensitivity_conventions', 'measurement_semantics',
    'native_pixel_solid_angle_sr',
)


def _merge_channel_metadata(channel_ids, observations):
    """Per-channel metadata of the instrument, merged over the observations.

    A channel missing from an observation carries ``None`` there; the merged
    value is the one shared by every observation that has the channel. Every
    channel must be present in at least one observation.
    """
    merged = {key: [None] * len(channel_ids) for key in _PER_CHANNEL_METADATA}
    for metadata in observations:
        for key in _PER_CHANNEL_METADATA:
            values = metadata[key]
            if len(values) != len(channel_ids):
                raise ValueError(f'Prepared EUV {key} must hold one value per channel.')
            for index, value in enumerate(values):
                if value is None:
                    continue
                if merged[key][index] is None:
                    merged[key][index] = value
                elif merged[key][index] != value:
                    raise ValueError(
                        f'Prepared EUV {key} of channel {channel_ids[index]!r} change '
                        f'between observations: {merged[key][index]!r} != {value!r}.'
                    )
    absent = [
        channel for channel, value in zip(channel_ids, merged['measurement_units'])
        if value is None
    ]
    if absent:
        raise ValueError(
            f'Channels {absent!r} have no prepared file in any selected observation.'
        )
    return {key: tuple(values) for key, values in merged.items()}


class GenericEUVDataset(TensorsDataset):
    def __init__(self, file_dict, date_dict, work_directory, ds_key, instrument_key, Rs_per_ds=1, seconds_per_dt=86400, ref_date=None,
                 batch_size=int(2 ** 10), debug=False, test=False, cmaps=None, static=False, max_radius_rsun=None,
                 instrument_type=None, match_tolerance_minutes=2, alignment_tolerance_arcsec=0.25,
                 strict_metadata=False, load_workers=None, debug_max_observations=20,
                 time=None, prepared_data_path=None, start=None, end=None, cadence_minutes=None,
                 shuffle_seed=None, saturation_dn=None, **kwargs):
        channel_ids = tuple(file_dict)
        instrument_type = instrument_key if instrument_type is None else instrument_type
        # Observations may miss channels (None path); the matcher keeps a time
        # step as soon as one channel is present.
        matched_files, _, matched_times = match_prepared_channels(
            file_dict,
            date_dict,
            tolerance=timedelta(minutes=float(match_tolerance_minutes)),
            channel_order=channel_ids,
        )
        complete = np.asarray([
            all(matched_files[channel][index] is not None for channel in channel_ids)
            for index in range(len(matched_times))
        ], dtype=bool)
        n_available = len(matched_times)
        n_files = sum(len(paths) for paths in file_dict.values())
        log = [
            f'{ds_key}: {n_files} prepared files in {n_available} observations '
            f'({_iso(matched_times[0])} .. {_iso(matched_times[-1])})'
        ]
        # Optional time window [start, end) on the observation times, then an
        # optional fixed cadence, e.g. a background sequence that must not
        # oversample an interval with extra observations.
        keep = np.arange(n_available)
        if start is not None or end is not None:
            n_before = len(keep)
            keep = keep[_time_window_indices(matched_times[keep], start, end)]
            log.append(
                f'  time window [{_iso(start)}, {_iso(end)}): {len(keep)} of {n_before}'
            )
        if cadence_minutes is not None:
            n_before = len(keep)
            keep = keep[_cadence_indices(matched_times[keep], complete[keep], cadence_minutes)]
            log.append(f'  cadence {float(cadence_minutes):g} min: {len(keep)} of {n_before}')

        # Training datasets use every observation of their window; a validation
        # dataset renders the observation with all channels closest to
        # data.valid_datasets[].time (or to the middle of its window).
        n_before = len(keep)
        selection = _select_observations(
            matched_times[keep], complete[keep], test=test, time=time,
            debug=debug, debug_max_observations=debug_max_observations,
        )
        keep = keep[selection]
        if test:
            target = 'middle of the sequence' if time is None else f'time {_iso(time)}'
            log.append(
                f'  validation at {target}: observation {_iso(matched_times[keep[0]])} '
                f'(1 of {n_before})'
            )
        elif debug:
            log.append(f'  debug: {len(keep)} of {n_before}')
        matched_files = {
            channel: np.asarray(paths, dtype=object)[keep]
            for channel, paths in matched_files.items()
        }
        matched_times = matched_times[keep]
        n_used_files = sum(
            path is not None for paths in matched_files.values() for path in paths
        )
        missing = ', '.join(
            f'{channel}: {count}'
            for channel in channel_ids
            if (count := sum(path is None for path in matched_files[channel]))
        )
        log.append(
            f'  using {len(keep)} of {n_available} observations, {n_used_files} of '
            f'{n_files} files ({int(complete[keep].sum())} with all channels'
            + (f'; missing {missing}' if missing else '') + ')'
        )
        print('\n'.join(log))
        payloads = [
            (
                tuple(
                    None if matched_files[channel][observation_index] is None
                    else str(matched_files[channel][observation_index])
                    for channel in channel_ids
                ),
                channel_ids,
                instrument_type,
                instrument_key,
                Rs_per_ds,
                # data.*_datasets[].max_radius_rsun: field of view applied at
                # load time (None: the full prepared image).
                max_radius_rsun,
                alignment_tolerance_arcsec,
                strict_metadata,
                # instruments[].saturation_dn: pixels whose peak detector DN
                # reach it are invalid (None: no saturation mask).
                saturation_dn,
            )
            for observation_index in range(len(keep))
        ]
        if not payloads:
            raise ValueError(f'No usable observations found for {ds_key!r}.')
        self.channel_ids = tuple(str(channel) for channel in channel_ids)

        # Each loader worker reduces its observation to the cache rows, already
        # shuffled for training, so the main process holds one compact copy of
        # the dataset and never a full image or ray stack. The observation time
        # is constant per observation and is tiled by the writer once the
        # reference date is known, after the last observation.
        n_observations = len(payloads)
        writer = ObservationCacheWriter(
            work_directory, n_observations, ds_name=ds_key, shuffle=not test,
            shuffle_seed=shuffle_seed,
        )
        overview_step = max(1, n_observations // 10)
        seeds = np.random.SeedSequence(shuffle_seed).spawn(n_observations)
        row_payloads = [
            (payload, not test, seeds[index], not test and index % overview_step == 0)
            for index, payload in enumerate(payloads)
        ]
        overview_images = []
        overview_poses = []
        overview_indices = []
        observation_times = []
        observers = []
        metadata = []
        first = None
        try:
            for index, record in enumerate(_iter_euv_rows(
                row_payloads, load_workers, f'Loading {instrument_key} prepared observations'
            )):
                if first is None:
                    first = {
                        'image_shape': record['image_shape'], 'wcs': record['wcs'],
                        'wavelength': record['wavelength'], 'metadata': record['metadata'],
                    }
                    if record['image_shape'][-1] != len(channel_ids):
                        raise ValueError('Loaded image channels do not match channel_ids.')
                else:
                    if record['image_shape'] != first['image_shape']:
                        raise ValueError(
                            f'Prepared EUV image shape changes between observations: '
                            f'{record["image_shape"]} != {first["image_shape"]}.'
                        )
                observation_times.append(record['time'])
                observers.append(record['observer'])
                metadata.append(record['metadata'])
                if record['frame'] is not None:
                    overview_images.append(record['frame'])
                    overview_poses.append(np.asarray(record['pose']))
                    overview_indices.append(index)
                writer.add(record['rows'])
                del record

            channel_metadata = _merge_channel_metadata(channel_ids, metadata)
            native_pixel_solid_angle_sr = channel_metadata['native_pixel_solid_angle_sr']

            cmap_values = ['gray'] * len(channel_ids) if cmaps is None else list(cmaps)
            if len(cmap_values) != len(channel_ids):
                raise ValueError('cmaps must contain one entry per ordered EUV channel.')
            data_config = {
                'instrument_key': instrument_key,
                'image_shape': tuple(first['image_shape'][:2]),
                'wcs': next(wcs for wcs in first['wcs'] if wcs is not None),
                'wcs_by_channel': first['wcs'],
                'wavelength': first['wavelength'],
                'channel_ids': self.channel_ids,
                'cmaps': cmap_values,
                'measurement_units': channel_metadata['measurement_units'],
                'sensitivity_conventions': channel_metadata['sensitivity_conventions'],
                'measurement_semantics': channel_metadata['measurement_semantics'],
                'native_pixel_solid_angle_sr': native_pixel_solid_angle_sr,
                # Missing channels take the instrument value so the contract
                # check stays one finite number per channel and observation.
                'native_pixel_solid_angle_sr_by_observation': [
                    [
                        native_pixel_solid_angle_sr[index] if value is None else value
                        for index, value in enumerate(record_metadata['native_pixel_solid_angle_sr'])
                    ]
                    for record_metadata in metadata
                ],
                'prepared_schema': metadata[0]['schema'],
                'prepared_observations': metadata,
                'prepared_data_path': (
                    None if prepared_data_path is None
                    else str(prepared_data_path)
                ),
                'observers': observers,
                'match_tolerance_seconds': float(match_tolerance_minutes) * 60,
                'channel_matching': 'partial_channels_within_tolerance',
                'time_window': {
                    'start': None if start is None else _utc_naive(start).isoformat(),
                    'end': None if end is None else _utc_naive(end).isoformat(),
                },
                'cadence_minutes': None if cadence_minutes is None else float(cadence_minutes),
                'batch_size_convention': 'per_rank',
                'split': {
                    'role': 'validation' if test else 'training',
                    'time': None if time is None else _utc_naive(time).isoformat(),
                    'selected_matched_indices': keep.astype(int).tolist(),
                },
            }
            self.data_config = data_config
            del first

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

            if not test:
                log_overview(
                    np.stack(overview_images, axis=0), np.stack(overview_poses, axis=0),
                    normalized_times[overview_indices], cmap_values, seconds_per_dt,
                    Rs_per_ds, ref_date, ds_key=ds_key, channel_labels=self.channel_ids,
                )
                del overview_images, overview_poses

            cache = writer.finalize(constant_tensors={'time': normalized_times[:, None]})
        except Exception:
            writer.abort()
            raise

        super().__init__(
            cache=cache,
            batch_size=batch_size,
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


def _discover_prepared_files(files, wavelengths, load_workers=None):
    # Training and validation datasets point at the same prepared files;
    # reading every FITS header is the dominant cost of the validation
    # datasets, which only load their one observation afterwards. The headers
    # are read in the same worker pool as the observations.
    cache_key = (tuple(files), tuple(wavelengths))
    if cache_key not in _DISCOVERY_CACHE:
        _DISCOVERY_CACHE[cache_key] = discover_prepared_files(
            files, wavelengths, workers=_pool_size(len(files), load_workers)
        )
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

        file_dict, date_dict = _discover_prepared_files(
            files, wavelengths, kwargs.get('load_workers')
        )
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
        file_dict, date_dict = _discover_prepared_files(
            files, wavelengths, kwargs.get('load_workers')
        )
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

        file_dict, date_dict = _discover_prepared_files(
            files, wavelengths, kwargs.get('load_workers')
        )
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
