"""Series reconstruction: one time window moved forward in fixed steps.

A regular ``run_plasma`` configuration (``base_config``) defines the first
window: the ``start``/``end`` of every training dataset and the ``time`` (or
``start``/``end``) of every validation dataset. Step ``k`` shifts all of them
by ``k * step_days``. ``data.ref_date`` is not shifted, so model time means the
same date in every step. A focus is a training dataset with a shorter window
inside the background window; it moves with the rest.

Step 0 is the base run itself, trained from scratch with ``run_plasma``; its
save state is ``initial_state``. Every later step starts from the weights of
the step before (``training.init_state``) with the ``overrides`` deep-merged
into the base configuration, and runs ``python -m sunerf.run_plasma`` on its
own config in ``<base_path>/steps/<index>_<validation time>/``. A finished step
keeps ``config.yaml`` and ``save_state.snf``; its Lightning checkpoints and its
data cache (``<work_directory>/steps/...``) are removed. An interrupted step
resumes from its own last checkpoint, so the driver can be restarted at any
time, e.g. by a chain of batch jobs; ``<base_path>/series.done`` marks the end.

Series configuration::

    base_config: "config/plasma/<run>.yaml"
    initial_state: "/path/to/<run>/save_state.snf"
    base_path: "/path/to/<run>_series"
    work_directory: "/scratch/<run>_series"
    end: "2014-10-01T00:00:00"
    step_days: 5
    overrides:
      training: {max_steps: 50000}

Usage::

    python -m sunerf.run_plasma_series --config config/plasma/<run>_series.yaml
    python -m sunerf.run_plasma_series --config ... --dry-run
"""

import argparse
import copy
import glob
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta

import yaml

from sunerf.configuration import validate_plasma_config
from sunerf.train.util import load_yaml_config

DONE_MARKER = 'series_step.done'
# Written to the series base_path once every step is finished.
SERIES_DONE_MARKER = 'series.done'
STEP_CONFIG = 'config.yaml'
SAVE_STATE = 'save_state.snf'


def _parse_time(value, name):
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be an ISO 8601 timestamp string')
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def _format_time(value):
    return value.isoformat(timespec='seconds')


def deep_merge(base, overrides):
    """``base`` with ``overrides`` merged into it; nested mappings merge, other values replace."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_series_config(path):
    with open(path) as f:
        series = yaml.safe_load(f)
    if not isinstance(series, Mapping):
        raise ValueError(f'Series configuration {path} must be a mapping')
    for key in ('base_config', 'initial_state', 'base_path', 'end', 'step_days'):
        if series.get(key) in (None, ''):
            raise ValueError(f'Series configuration {path} must define {key}')
    if not float(series['step_days']) > 0:
        raise ValueError('step_days must be positive')
    overrides = series.get('overrides', {}) or {}
    if not isinstance(overrides, Mapping):
        raise ValueError('overrides must be a mapping')
    if 'data' in overrides:
        raise ValueError('overrides must not change data; the windows come from base_config')
    return series


def _window_fields(base):
    """Every (group, index, field) of the data windows that moves with the series."""
    data = base['data']
    if data.get('ref_date') is None:
        raise ValueError('The base configuration must fix data.ref_date for a series')
    fields = []
    for index, dataset in enumerate(data['train_datasets']):
        if dataset.get('start') is None or dataset.get('end') is None:
            raise ValueError(
                f'data.train_datasets[{index}] needs start and end: every training '
                'dataset of a series is a window'
            )
        fields += [('train_datasets', index, 'start'), ('train_datasets', index, 'end')]
    for index, dataset in enumerate(data['valid_datasets']):
        present = [field for field in ('time', 'start', 'end') if dataset.get(field) is not None]
        if not present:
            raise ValueError(
                f'data.valid_datasets[{index}] needs time (or start and end) to move with the series'
            )
        fields += [('valid_datasets', index, field) for field in present]
    return fields


def shifted_config(base, shift):
    """Copy of ``base`` with every data window moved forward by ``shift``."""
    config = copy.deepcopy(base)
    for group, index, field in _window_fields(base):
        dataset = config['data'][group][index]
        dataset[field] = _format_time(_parse_time(dataset[field], f'data.{group}[{index}].{field}') + shift)
    return config


def _window_end(config):
    return max(_parse_time(dataset['end'], 'end') for dataset in config['data']['train_datasets'])


def _validation_time(config):
    dataset = config['data']['valid_datasets'][0]
    if dataset.get('time') is not None:
        return _parse_time(dataset['time'], 'time')
    start, end = _parse_time(dataset['start'], 'start'), _parse_time(dataset['end'], 'end')
    return start + (end - start) / 2


def series_schedule(base, series):
    """``(index, name, config)`` of every window: 0 is the base run, then one per step."""
    step = timedelta(days=float(series['step_days']))
    end = _parse_time(series['end'], 'end')
    schedule = []
    index = 0
    while True:
        config = shifted_config(base, step * index)
        if _window_end(config) > end:
            break
        name = f'{index:03d}_{_validation_time(config):%Y-%m-%dT%H}'
        schedule.append((index, name, config))
        index += 1
    if not schedule:
        raise ValueError('The first window of base_config already ends after the series end')
    return schedule


def step_config(config, series, name, init_state):
    """Run configuration of one series step (index >= 1)."""
    config = deep_merge(config, series.get('overrides', {}) or {})
    config['base_path'] = os.path.join(series['base_path'], 'steps', name)
    work_root = series.get('work_directory', series['base_path'])
    config['work_directory'] = os.path.join(work_root, 'steps', name)
    training = config.setdefault('training', {})
    training['init_state'] = init_state
    training['meta_path'] = 'last'
    logging_config = config.setdefault('logging', {})
    logging_config['name'] = f"{logging_config.get('name', 'series')} | {name}"
    logging_config.setdefault('group', os.path.basename(os.path.normpath(series['base_path'])))
    return validate_plasma_config(config)


def _clean_finished_step(step_dir, work_directory):
    for path in glob.glob(os.path.join(step_dir, '*.ckpt')):
        os.remove(path)
    if os.path.isdir(work_directory):
        shutil.rmtree(work_directory)


def run_series(series_path, *, n_steps=None, dry_run=False):
    series = load_series_config(series_path)
    base = load_yaml_config(series['base_config'])
    validate_plasma_config(base)
    schedule = series_schedule(base, series)
    print(f'Series of {len(schedule)} windows (step 0 = {series["base_config"]}), '
          f'step {float(series["step_days"]):g} days')

    state = series['initial_state']
    if not dry_run and not os.path.isfile(state):
        raise FileNotFoundError(f'initial_state {state} does not exist; train the base run first')
    trained = 0
    for index, name, window in schedule[1:]:
        config = step_config(window, series, name, state)
        step_dir = config['base_path']
        step_state = os.path.join(step_dir, SAVE_STATE)
        if os.path.isfile(os.path.join(step_dir, DONE_MARKER)):
            state = step_state
            continue
        if dry_run:
            train = config['data']['train_datasets']
            print(f'{name}: ' + ', '.join(
                f"{dataset.get('key', dataset['type'])} {dataset['start'][:10]}..{dataset['end'][:10]}"
                for dataset in train))
            state = step_state
            continue
        if n_steps is not None and trained >= n_steps:
            print(f'Stopping after {trained} steps; the next step is {name}')
            return
        os.makedirs(step_dir, exist_ok=True)
        config_path = os.path.join(step_dir, STEP_CONFIG)
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f'Series step {name}: {config_path}', flush=True)
        subprocess.run([sys.executable, '-m', 'sunerf.run_plasma', '--config', config_path], check=True)
        if not os.path.isfile(step_state):
            raise RuntimeError(f'Step {name} finished without {step_state}')
        with open(os.path.join(step_dir, DONE_MARKER), 'w') as f:
            f.write(datetime.now().isoformat(timespec='seconds') + '\n')
        _clean_finished_step(step_dir, config['work_directory'])
        state = step_state
        trained += 1
    if dry_run:
        print('Dry run complete')
        return
    with open(os.path.join(series['base_path'], SERIES_DONE_MARKER), 'w') as f:
        f.write(datetime.now().isoformat(timespec='seconds') + '\n')
    print('Series complete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True, help='series configuration YAML')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='train at most this many steps, then stop (e.g. per batch job)')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate every step configuration and print the windows without training')
    args = parser.parse_args(argv)
    run_series(args.config, n_steps=args.n_steps, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
