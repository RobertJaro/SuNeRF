import copy
from datetime import datetime, timedelta

import pytest
import torch
import yaml

from sunerf.configuration import PlasmaConfigError, validate_plasma_config
from sunerf.run_plasma import load_initial_state
from sunerf.run_plasma_series import (
    deep_merge,
    load_series_config,
    series_schedule,
    shifted_config,
    step_config,
)
from sunerf.train.util import load_yaml_config

BASE_CONFIG = 'config/plasma/polar_reversal.yaml'
SERIES_CONFIG = 'config/plasma/polar_reversal_series.yaml'


@pytest.fixture
def base():
    return load_yaml_config(BASE_CONFIG)


@pytest.fixture
def series():
    return load_series_config(SERIES_CONFIG)


def _times(config, group, field):
    return [datetime.fromisoformat(d[field]) for d in config['data'][group] if field in d]


def test_base_config_is_valid(base):
    validate_plasma_config(base)


def test_schedule_covers_the_interval_in_five_day_steps(base, series):
    schedule = series_schedule(base, series)
    assert len(schedule) == 262
    assert schedule[0][2] == base
    assert schedule[1][1] == '001_2011-02-24T12'
    last = schedule[-1][2]
    assert max(_times(last, 'train_datasets', 'end')) <= datetime(2014, 10, 1)
    shifted_once_more = shifted_config(base, timedelta(days=5 * len(schedule)))
    assert max(_times(shifted_once_more, 'train_datasets', 'end')) > datetime(2014, 10, 1)


def test_shift_moves_every_window_and_keeps_the_time_origin(base):
    shifted = shifted_config(base, timedelta(days=10))
    assert shifted['data']['ref_date'] == base['data']['ref_date']
    for group, field in (('train_datasets', 'start'), ('train_datasets', 'end'),
                         ('valid_datasets', 'time')):
        before, after = _times(base, group, field), _times(shifted, group, field)
        assert after and all(b - a == timedelta(days=-10) for a, b in zip(after, before))
    focus = [d for d in shifted['data']['train_datasets'] if d['key'].endswith('-focus')]
    assert {(d['start'], d['end']) for d in focus} == {('2011-02-27T00:00:00', '2011-03-04T00:00:00')}


def test_step_config_warm_starts_with_the_overrides(base, series):
    _, name, window = series_schedule(base, series)[3]
    config = step_config(window, series, name, '/previous/save_state.snf')
    assert config['base_path'].endswith(f'polar_reversal_series_v01/steps/{name}')
    assert config['work_directory'].endswith(f'steps/{name}')
    assert config['training']['init_state'] == '/previous/save_state.snf'
    assert config['training']['max_steps'] == 20000
    assert config['training']['epochs'] == base['training']['epochs']
    assert config['module']['lr_config']['start'] == 2.0e-4
    assert config['module']['regularization_log_T_range'] == base['module']['regularization_log_T_range']
    assert config['shuffle']['cold_iterations'] == 0
    assert config['logging']['group'] == 'polar_reversal_series_v01'
    # The generated file is a plain run_plasma config.
    assert validate_plasma_config(yaml.safe_load(yaml.safe_dump(config))) == config


def test_series_rejects_open_training_windows(base, series):
    broken = copy.deepcopy(base)
    del broken['data']['train_datasets'][0]['start']
    with pytest.raises(ValueError, match='needs start and end'):
        series_schedule(broken, series)
    unanchored = copy.deepcopy(base)
    del unanchored['data']['ref_date']
    with pytest.raises(ValueError, match='ref_date'):
        series_schedule(unanchored, series)


def test_series_overrides_must_not_change_the_data(tmp_path):
    path = tmp_path / 'series.yaml'
    path.write_text(yaml.safe_dump({
        'base_config': BASE_CONFIG, 'initial_state': 'x', 'base_path': 'y',
        'end': '2014-10-01T00:00:00', 'step_days': 5, 'overrides': {'data': {'batch_size': 2}},
    }))
    with pytest.raises(ValueError, match='must not change data'):
        load_series_config(path)


def test_deep_merge_merges_mappings_and_replaces_values():
    merged = deep_merge({'a': {'b': 1, 'c': 2}, 'd': [1]}, {'a': {'b': 3}, 'd': [2]})
    assert merged == {'a': {'b': 3, 'c': 2}, 'd': [2]}


def test_training_budget_and_initial_state_are_validated(base):
    config = copy.deepcopy(base)
    config['training'].update(max_steps=10, init_state='/path/save_state.snf')
    validate_plasma_config(config)
    config['training']['max_steps'] = 0
    with pytest.raises(PlasmaConfigError, match='max_steps'):
        validate_plasma_config(config)
    config['training'].update(max_steps=10, init_state='')
    with pytest.raises(PlasmaConfigError, match='init_state'):
        validate_plasma_config(config)


def test_load_initial_state_copies_weights_strictly(tmp_path):
    source, target = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    path = tmp_path / 'save_state.snf'
    torch.save({'state_dict': source.state_dict()}, path)
    load_initial_state(target, str(path))
    assert torch.equal(target.weight, source.weight)
    with pytest.raises(RuntimeError):
        load_initial_state(torch.nn.Linear(4, 2), str(path))
    torch.save({'weights': 1}, path)
    with pytest.raises(ValueError, match='no state_dict'):
        load_initial_state(target, str(path))


def test_run_series_trains_resumes_and_cleans_up(tmp_path, monkeypatch):
    from sunerf import run_plasma_series

    initial = tmp_path / 'initial_save_state.snf'
    initial.write_bytes(b'')
    series_path = tmp_path / 'series.yaml'
    series_path.write_text(yaml.safe_dump({
        'base_config': BASE_CONFIG, 'initial_state': str(initial),
        'base_path': str(tmp_path / 'series'), 'work_directory': str(tmp_path / 'scratch'),
        'end': '2011-03-20T00:00:00', 'step_days': 5,
        'overrides': {'training': {'max_steps': 10}},
    }))
    calls = []

    def fake_run(command, check):
        config = yaml.safe_load(open(command[-1]))
        calls.append(config['training']['init_state'])
        for path in (config['base_path'], config['work_directory']):
            __import__('os').makedirs(path, exist_ok=True)
        (tmp_path / config['base_path'] / 'save_state.snf').write_bytes(b'')
        (tmp_path / config['base_path'] / 'last.ckpt').write_bytes(b'')

    monkeypatch.setattr(run_plasma_series.subprocess, 'run', fake_run)
    run_plasma_series.run_series(series_path, n_steps=1)
    steps = tmp_path / 'series' / 'steps'
    assert calls == [str(initial)]
    assert not (tmp_path / 'series' / 'series.done').exists()

    run_plasma_series.run_series(series_path)
    names = sorted(p.name for p in steps.iterdir())
    assert names == ['001_2011-02-24T12', '002_2011-03-01T12', '003_2011-03-06T12']
    assert calls[1:] == [str(steps / names[0] / 'save_state.snf'), str(steps / names[1] / 'save_state.snf')]
    for name in names:
        assert sorted(p.name for p in (steps / name).iterdir()) == ['config.yaml', 'save_state.snf', 'series_step.done']
        assert not (tmp_path / 'scratch' / 'steps' / name).exists()
    assert (tmp_path / 'series' / 'series.done').exists()
