"""Explicit per-instrument image-scaling tables for the plasma pipeline.

The loss-space divisor is a fixed constant of an *instrument*
(``instruments[].scaling.divisor``), never a by-product of loading a dataset.
Every dataset mapped to the instrument (background sequence, high-cadence event
sequence, validation-only view) therefore shares it, and the prepared images
stay in physical units.

This module estimates the divisors once from prepared FITS files, stores them in
a small YAML table, and resolves table references in a plasma configuration:

    python -m sunerf.data.euv.estimate_scaling --config config/plasma/<run>.yaml

Instruments whose ``scaling.divisor`` is a path are estimated from their training
datasets (validation-only instruments from their validation datasets) and written
to that path. An existing table is reused unless
``--overwrite`` is given, so repeated or resumed runs keep the same constants.

The constants only set the loss scale, so a representative sample suffices: by
default 200 evenly spaced prepared files per dataset are opened, nothing else is
scanned, and each file contributes to its own channel. The result is printed as a
table entry that can be pasted into the packaged defaults
(``sunerf/resources/calibration/image_scaling.yaml``, referenced as
``divisor: "builtin:image_scaling"``).
"""

import argparse
import copy
import multiprocessing
import os
from collections.abc import Mapping
from datetime import datetime, timezone

import numpy as np
import yaml
from tqdm import tqdm

from sunerf.configuration import canonical_channel_id
from sunerf.resources import is_builtin, resolve_artifact_path
from astropy.io import fits

from sunerf.data.euv.observation import (
    _channel_number,
    _fits_valid_mask,
    _header_channel,
    normalize_image_scaling,
    observation_scaling_statistics,
    reduce_scaling_statistics,
)

SCALING_TABLE_SCHEMA = 'sunerf.image_scaling_table.v1'


DEFAULT_MAX_FILES = 200


def _table_path(path):
    if is_builtin(path):
        return str(resolve_artifact_path(path))
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(path))))


def load_scaling_table(path):
    path = _table_path(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'Image-scaling table {path!r} does not exist; create it with '
            '`python -m sunerf.data.euv.estimate_scaling --config <config>`.'
        )
    with open(path) as f:
        table = yaml.safe_load(f)
    if (
        not isinstance(table, Mapping)
        or table.get('schema') != SCALING_TABLE_SCHEMA
        or not isinstance(table.get('instruments'), Mapping)
    ):
        raise ValueError(f'{path!r} is not a {SCALING_TABLE_SCHEMA} image-scaling table.')
    return table


def write_scaling_table(path, instruments):
    if is_builtin(path):
        raise ValueError(f'{path!r} is a packaged table; edit it in sunerf/resources instead.')
    path = _table_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = {
        'schema': SCALING_TABLE_SCHEMA,
        'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'instruments': instruments,
    }
    temporary_path = f'{path}.tmp{os.getpid()}'
    with open(temporary_path, 'w') as f:
        yaml.safe_dump(table, f, sort_keys=False)
    os.replace(temporary_path, path)
    return path


def resolve_instrument_scaling(instruments_config):
    """Replace ``scaling.divisor`` table paths by their explicit channel vectors.

    The returned copy carries numbers only, so logged hyperparameters and saved
    artifacts record the constants that were actually used.
    """
    instruments_config = copy.deepcopy(instruments_config)
    tables = {}
    for instrument in instruments_config:
        scaling = instrument.get('scaling')
        if not isinstance(scaling, Mapping) or not isinstance(scaling.get('divisor'), str):
            continue
        path = _table_path(scaling['divisor'])
        if path not in tables:
            tables[path] = load_scaling_table(path)
        instrument_key = instrument['key']
        entry = tables[path]['instruments'].get(instrument_key)
        if entry is None:
            raise ValueError(
                f'Image-scaling table {path!r} has no entry for instrument '
                f'{instrument_key!r}; rerun estimate_scaling with --overwrite.'
            )
        channels = instrument['temperature_response']['channels']
        try:
            divisors, _ = normalize_image_scaling(entry, channels)
        except ValueError as error:
            raise ValueError(
                f'Image-scaling table {path!r} entry {instrument_key!r} does not '
                f'match response channels {list(channels)}: {error}'
            ) from error
        scaling['divisor'] = [float(value) for value in divisors]
        scaling['divisor_source'] = path
    return instruments_config


def _file_statistics(payload):
    """Percentile of one prepared file, placed at its channel index; NaN elsewhere."""
    path, channel_numbers, percentile = payload
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        image = np.asarray(hdul[0].data, dtype=np.float64)
    row = np.full(len(channel_numbers), np.nan, dtype=np.float64)
    channel = _header_channel(header)
    if channel not in channel_numbers:
        return row
    valid_mask = _fits_valid_mask(path, image.shape, required=False)
    row[channel_numbers.index(channel)] = observation_scaling_statistics(
        image[..., None], valid_mask[..., None], percentile
    )[0]
    return row


def _dataset_payloads(dataset, channel_numbers, percentile, max_files):
    # Imported lazily: resolving a table must not pull in the training stack.
    from sunerf.data.loader.multi_instrument import _dataset_files

    wavelengths = dataset.get('wavelengths', dataset.get('channels'))
    if sorted(_channel_number(value) for value in wavelengths) != sorted(channel_numbers):
        raise ValueError(
            f"Dataset {dataset.get('key')!r} channels {list(wavelengths)} do not match "
            f'response channels {list(channel_numbers)}.'
        )
    files = _dataset_files(dataset['data_path'], wavelengths, instrument_type=dataset['type'])
    # Only the sampled files are ever opened: no header scan, no channel
    # matching. Prepared files sort by name, which interleaves the channels.
    if max_files is not None and len(files) > max_files:
        positions = np.unique(np.linspace(0, len(files) - 1, int(max_files), dtype=np.int64))
        files = [files[index] for index in positions]
    return len(files), [(path, tuple(channel_numbers), percentile) for path in files]


def estimate_instrument_scaling(
        datasets, channels, *, percentile=99.5, max_files=DEFAULT_MAX_FILES, workers=None,
):
    """Pool the per-file statistics of a sample of all given datasets of one instrument."""
    channels = [str(channel) for channel in channels]
    channel_numbers = [_channel_number(channel) for channel in channels]
    payloads = []
    for dataset in datasets:
        _, dataset_payloads = _dataset_payloads(dataset, channel_numbers, percentile, max_files)
        payloads.extend(dataset_payloads)
    if not payloads:
        raise ValueError('No prepared files available to estimate image scaling.')

    available = os.cpu_count() or 1
    workers = min(len(payloads), available if workers is None else max(int(workers), 1))
    description = f"Image scaling {datasets[0]['instrument_key']}"
    if workers <= 1:
        statistics = [_file_statistics(p) for p in tqdm(payloads, desc=description)]
    else:
        with multiprocessing.Pool(workers) as pool:
            statistics = list(tqdm(
                pool.imap(_file_statistics, payloads, chunksize=4),
                total=len(payloads), desc=description,
            ))
    _, metadata = reduce_scaling_statistics(np.stack(statistics), channels, percentile=percentile)
    metadata['source'] = {
        'datasets': [str(dataset.get('key', dataset['type'])) for dataset in datasets],
        'data_paths': [dataset['data_path'] for dataset in datasets],
        'file_count': len(payloads),
    }
    return metadata


def estimate_config_scaling(
        config, *, datasets=None, percentile=99.5, max_files=DEFAULT_MAX_FILES, workers=None,
        overwrite=False,
):
    """Estimate and write every table referenced by ``instruments[].scaling.divisor``."""
    data = config['data']
    requested = {}
    for instrument in config['instruments']:
        divisor = (instrument.get('scaling') or {}).get('divisor')
        if isinstance(divisor, str):
            requested.setdefault(_table_path(divisor), []).append(instrument)
    if not requested:
        print('No instrument references an image-scaling table; nothing to estimate.')
        return []

    written = []
    for path, instruments in requested.items():
        entries = {}
        if any(is_builtin(i['scaling']['divisor']) for i in instruments):
            print(f'Packaged image-scaling table {path} is used as is.')
            continue
        if os.path.isfile(path) and not overwrite:
            entries = dict(load_scaling_table(path)['instruments'])
        missing = [i for i in instruments if i['key'] not in entries]
        if not missing:
            print(f'Reusing image-scaling table {path} (pass --overwrite to re-estimate).')
            continue
        for instrument in missing:
            instrument_datasets = [
                dataset for dataset in data['train_datasets']
                if dataset['instrument_key'] == instrument['key']
                and (datasets is None or dataset.get('key') in datasets)
            ]
            if not any(
                dataset['instrument_key'] == instrument['key']
                for dataset in data['train_datasets']
            ):
                # Validation-only instrument: nothing is trained on it, so its
                # own observations may set the (purely diagnostic) loss scale.
                instrument_datasets = [
                    dataset for dataset in data['valid_datasets']
                    if dataset['instrument_key'] == instrument['key']
                ]
            if not instrument_datasets:
                raise ValueError(
                    f"No selected training dataset provides instrument {instrument['key']!r}; "
                    'set its scaling.divisor explicitly instead.'
                )
            entries[instrument['key']] = estimate_instrument_scaling(
                instrument_datasets,
                instrument['temperature_response']['channels'],
                percentile=percentile,
                max_files=max_files,
                workers=workers,
            )
            entry = entries[instrument['key']]
            print(f"{instrument['key']} ({entry['source']['file_count']} files):")
            print(yaml.safe_dump(
                {instrument['key']: {
                    'channel_ids': [str(c) for c in entry['channel_ids']],
                    'divisor': [round(float(v), 4) for v in entry['divisor']],
                }},
                sort_keys=False, default_flow_style=None, indent=2,
            ).rstrip())
        written.append(write_scaling_table(path, entries))
        print(f'Wrote image-scaling table {written[-1]}')
    return written


def main(argv=None):
    from sunerf.configuration import validate_plasma_config
    from sunerf.train.util import load_yaml_config

    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument(
        '--datasets', nargs='+', default=None,
        help='training dataset keys to estimate from (default: all). For mixed-cadence '
             'runs name the background sequence so a short event does not set the scale.',
    )
    parser.add_argument('--percentile', type=float, default=99.5)
    parser.add_argument(
        '--max-files', type=int, default=DEFAULT_MAX_FILES,
        help='evenly subsample each dataset to at most this many prepared files '
             f'(default {DEFAULT_MAX_FILES}; 0 opens every file)',
    )
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args, overwrite_args = parser.parse_known_args(argv)

    config = validate_plasma_config(load_yaml_config(args.config, overwrite_args))
    estimate_config_scaling(
        config,
        datasets=None if args.datasets is None else set(args.datasets),
        percentile=args.percentile,
        max_files=args.max_files or None,
        workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
