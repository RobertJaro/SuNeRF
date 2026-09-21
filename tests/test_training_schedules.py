import math

import pytest
import torch
from torch import nn

from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.thomson import LossWeightSchedule, ThomsonSuNeRFModule
from sunerf.run_thomson import _load_stage_initial_weights


class _Rendering(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.batch_end_calls = 0

    def forward(self, batch):
        return batch

    def on_train_batch_end(self, *args, **kwargs):
        self.batch_end_calls += 1


def _base_module(lr_config):
    return BaseSuNeRFModule(
        Rs_per_ds=1.0,
        seconds_per_dt=1.0,
        rendering=_Rendering(),
        validation_dataset_mapping={},
        lr_config=lr_config,
    )


def _optimizer_and_scheduler(module):
    configuration = module.configure_optimizers()
    return (
        configuration['optimizer'],
        configuration['lr_scheduler']['scheduler'],
        configuration,
    )


def test_lr_scheduler_is_step_interval_and_clamps_at_configured_end():
    module = _base_module({'start': 1.0, 'end': 0.01, 'iterations': 4})
    optimizer, scheduler, configuration = _optimizer_and_scheduler(module)

    assert configuration['lr_scheduler']['interval'] == 'step'
    assert configuration['lr_scheduler']['frequency'] == 1
    assert optimizer.param_groups[0]['lr'] == 1.0

    learning_rates = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]['lr'])

    expected = [10 ** -0.5, 0.1, 10 ** -1.5, 0.01, 0.01, 0.01]
    assert learning_rates == pytest.approx(expected)


def test_lr_scheduler_resume_continues_from_saved_step_without_manual_advance():
    config = {'start': 1.0, 'end': 0.01, 'iterations': 4}
    module = _base_module(config)
    optimizer, scheduler, _ = _optimizer_and_scheduler(module)
    for _ in range(2):
        optimizer.step()
        scheduler.step()

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    resumed = _base_module(config)
    resumed_optimizer, resumed_scheduler, _ = _optimizer_and_scheduler(resumed)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    resumed_optimizer.step()
    resumed_scheduler.step()
    assert resumed_scheduler.last_epoch == 3
    assert math.isclose(resumed_optimizer.param_groups[0]['lr'], 10 ** -1.5)

    # The module hook only logs the already-advanced LR; Lightning is the single
    # owner of scheduler stepping.
    resumed.log = lambda *args, **kwargs: None
    epoch_before = resumed_scheduler.last_epoch
    resumed.on_train_batch_end()
    assert resumed_scheduler.last_epoch == epoch_before
    assert resumed.rendering.batch_end_calls == 1


def test_loss_weight_schedule_buffers_and_absolute_step_clamping():
    schedule = LossWeightSchedule.from_config({
        'start': 1.0,
        'end': 0.01,
        'iterations': 4,
    })

    assert schedule['type'] == 'exponential_decay'
    assert set(schedule.state_dict()) == {'value', 'start', 'end', 'gamma', 'iterations'}
    assert dict(schedule.named_parameters()) == {}

    schedule.set_step(2)
    torch.testing.assert_close(schedule['value'], torch.tensor(0.1))
    schedule.set_step(4)
    torch.testing.assert_close(schedule['value'], schedule['end'])
    schedule.set_step(100)
    torch.testing.assert_close(schedule['value'], schedule['end'])

    step_schedule = LossWeightSchedule.from_config({
        'type': 'step',
        'steps': 4,
        'start': 0.0,
        'end': 1.0,
    })
    step_schedule.set_step(3)
    assert step_schedule['value'].item() == 0.0
    step_schedule.set_step(4)
    assert step_schedule['value'].item() == 1.0


def _schedule_only_thomson_module():
    module = ThomsonSuNeRFModule.__new__(ThomsonSuNeRFModule)
    nn.Module.__init__(module)
    module.lambdas = nn.ModuleDict({
        'image': LossWeightSchedule.from_config({
            'start': 1.0,
            'end': 0.01,
            'iterations': 4,
        }),
        'ratio': LossWeightSchedule.from_config(2.0),
    })
    module.validation_outputs = {}
    return module


def test_legacy_checkpoint_without_lambda_buffers_is_migrated_at_saved_step():
    module = _schedule_only_thomson_module()
    checkpoint = {'state_dict': {}, 'global_step': 2}

    module.on_load_checkpoint(checkpoint)

    torch.testing.assert_close(module.lambdas['image']['value'], torch.tensor(0.1))
    assert 'lambdas.image.value' in checkpoint['state_dict']
    assert 'lambdas.ratio.value' in checkpoint['state_dict']
    module.load_state_dict(checkpoint['state_dict'], strict=True)


def test_legacy_raw_state_dict_strict_load_uses_configured_schedule_defaults():
    module = _schedule_only_thomson_module()

    result = module.load_state_dict({}, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert module.lambdas['image']['value'].item() == 1.0


def test_stage_initial_weights_keep_new_yaml_lambda_schedule_but_resume_restores_it():
    prior_stage = _schedule_only_thomson_module()
    prior_stage._set_lambda_schedule_step(2)
    prior_state = prior_stage.state_dict()

    new_stage = ThomsonSuNeRFModule.__new__(ThomsonSuNeRFModule)
    nn.Module.__init__(new_stage)
    new_stage.lambdas = nn.ModuleDict({
        'image': LossWeightSchedule.from_config({
            'type': 'step',
            'steps': 4,
            'start': 0.0,
            'end': 3.0,
        }),
        'ratio': LossWeightSchedule.from_config(5.0),
    })
    new_stage.validation_outputs = {}

    _load_stage_initial_weights(new_stage, prior_state)

    assert new_stage.lambdas['image']['type'] == 'step'
    assert new_stage.lambdas['image']['value'].item() == 0.0
    assert new_stage.lambdas['image']['end'].item() == 3.0
    assert new_stage.lambdas['ratio']['value'].item() == 5.0
    assert 'lambdas.image.gamma' in prior_state  # caller state was not mutated

    # A normal state-dict load is the resume path and must restore saved progress.
    new_stage._set_lambda_schedule_step(4)
    resumed_state = new_stage.state_dict()
    resumed = ThomsonSuNeRFModule.__new__(ThomsonSuNeRFModule)
    nn.Module.__init__(resumed)
    resumed.lambdas = nn.ModuleDict({
        'image': LossWeightSchedule.from_config({
            'type': 'step',
            'steps': 4,
            'start': 0.0,
            'end': 3.0,
        }),
        'ratio': LossWeightSchedule.from_config(5.0),
    })
    resumed.validation_outputs = {}
    resumed.load_state_dict(resumed_state, strict=True)
    assert resumed.lambdas['image']['value'].item() == 3.0
