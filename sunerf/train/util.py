import torch
import wandb
import yaml
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torch import nn


class TimeShuffler(nn.Module):
    def __init__(self, probability=0.5, data_sets=None, iterations=1e5):
        super().__init__()
        self.data_sets = data_sets

        self.prob = nn.Parameter(torch.tensor(probability, dtype=torch.float32), requires_grad=False)
        self.gamma = probability / iterations if iterations > 0 else 0

    def forward(self, batch):
        if self.prob <= 0:
            return batch
        for ds_key in batch.keys():
            if self.data_sets is None or ds_key in self.data_sets:
                self._shuffle_times(batch[ds_key]['time'], self.prob)
        return batch

    def _shuffle_times(self, time, prob):
        # shuffle random fraction of times
        n_shuffle = int(time.shape[0] * prob)
        idx = torch.randperm(n_shuffle)
        rand_idx = torch.randint(0, time.shape[0] - n_shuffle, (1,))
        time[rand_idx:rand_idx + n_shuffle] = time[rand_idx + idx]

    def on_train_batch_end(self, *args, **kwargs):
        if self.prob > 0:
            new_prob = self.prob - self.gamma
            self.prob.copy_(new_prob)
        else:
            new_prob = torch.zeros_like(self.prob)
            self.prob.copy_(new_prob)
        wandb.log({'time_random': self.prob.detach().cpu().numpy()})



class NormalTimeShuffler(nn.Module):
    """Gaussian time jitter whose width decays linearly from start to end.

    The width stays at ``start`` for the first ``cold_iterations`` steps, then
    decays over ``iterations`` steps to ``end`` and stays there.
    """

    def __init__(self, start=50, end=0.0, iterations=1e5, cold_iterations=0,
                 instruments=None, seconds_per_dt=None):
        super().__init__()
        self.start = float(start)
        self.end = end
        self.cold_iterations = int(cold_iterations)
        if self.cold_iterations < 0:
            raise ValueError('cold_iterations must be non-negative.')
        self.scaling = nn.Parameter(torch.tensor(start, dtype=torch.float32), requires_grad=False)
        # Width removed per training step after the cold phase.
        self.gamma = torch.tensor((start - end) / iterations, dtype=torch.float32)
        self.instruments = instruments
        # Conversion from normalized model time to seconds; enables logging the
        # jitter width in physical units (hours) alongside the normalized value.
        self.seconds_per_dt = None if seconds_per_dt is None else float(seconds_per_dt)

    def forward(self, batch):
        if self.scaling == 0:
            return batch
        for ds_key in batch.keys():
            if self.instruments is None or ds_key in self.instruments:
                self._shuffle_times(batch[ds_key]['time'])
        return batch

    def _shuffle_times(self, time):
        time += torch.randn_like(time) * self.scaling

    def set_step(self, global_step):
        """Set the width from the absolute optimizer step (resume-safe)."""
        decay_steps = max(0, int(global_step) - self.cold_iterations)
        width = max(self.start - float(self.gamma) * decay_steps, float(self.end))
        self.scaling.fill_(width)

    def on_train_batch_end(self, *args, global_step=None, **kwargs):
        if global_step is not None:
            self.set_step(global_step)
        elif self.cold_iterations:
            raise ValueError('cold_iterations requires the global step.')
        else:
            self.scaling.copy_((self.scaling - self.gamma).clamp_min(self.end))
        @rank_zero_only
        def _log():
            scaling = float(self.scaling.detach().cpu())
            if self.seconds_per_dt is not None:
                wandb.log({'time_random_hours': scaling * self.seconds_per_dt / 3600.0}, commit=False)
            else:
                wandb.log({'time_random': scaling}, commit=False)
        _log()

class RotatedTimeShuffler(nn.Module):
    def __init__(self, start=50, end=1e-2, iterations=1e5, instruments=None):
        super().__init__()
        self.end = end
        self.scaling = nn.Parameter(torch.tensor(start, dtype=torch.float32), requires_grad=False)
        self.gamma = torch.tensor((end / start) ** (1 / iterations), dtype=torch.float32)
        self.instruments = instruments

    def forward(self, batch):
        if self.scaling == 0:
            return batch
        for ds_key in batch.keys():
            if self.instruments is None or ds_key in self.instruments:
                self._shuffle_times(batch[ds_key])
        return batch

    def _shuffle_times(self, batch):
        time = batch['time']
        time += torch.randn_like(time) * self.scaling

    def on_train_batch_end(self, *args, **kwargs):
        if self.scaling > self.end:
            new_gamma = self.scaling * self.gamma
            self.scaling.copy_(new_gamma)
        else:
            new_gamma = torch.zeros_like(self.scaling)
            self.scaling.copy_(new_gamma)
        wandb.log({'time_random': self.scaling.detach().cpu().numpy()}, commit=False)



def load_yaml_config(yaml_config_file, overwrite_args=None):
    overwrite_args = [] if overwrite_args is None else overwrite_args
    if len(overwrite_args) % 2:
        raise ValueError('Configuration overrides must be provided as --name value pairs')
    if not all(k.startswith('--') and len(k) > 2 for k in overwrite_args[::2]):
        raise ValueError('Configuration override names must start with --')
    overwrite_args = {
        key[2:]: value
        for key, value in zip(overwrite_args[::2], overwrite_args[1::2])
    }
    with open(yaml_config_file) as f:
        config_str = f.read()
    for overwrite_key, overwrite_value in overwrite_args.items():
        placeholder = '{%s}' % overwrite_key
        if placeholder not in config_str:
            raise ValueError(
                f"Unknown configuration override --{overwrite_key}: "
                f"placeholder {placeholder!r} is absent from {yaml_config_file}"
            )
        config_str = config_str.replace(placeholder, overwrite_value)
    config = yaml.safe_load(config_str)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration root in {yaml_config_file} must be a mapping')
    return config


def atan2_safe(numerator, denominator):
    epsilon = 1e-7
    nudge = (denominator == 0) * epsilon
    denominator = denominator + nudge
    out = torch.atan2(numerator, denominator)
    return out


def acos_safe(x):
    epsilon = 1e-7
    nudge_pos = (x == 1) * epsilon
    nudge_neg = (x == -1) * epsilon
    x = x - nudge_pos + nudge_neg
    out = torch.acos(x)
    return out


def asin_safe(x):
    epsilon = 1e-7
    nudge_pos = (x == 1) * epsilon
    nudge_neg = (x == -1) * epsilon
    x = x - nudge_pos + nudge_neg
    out = torch.asin(x)
    return out
