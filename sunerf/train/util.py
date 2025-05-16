import torch
import wandb
import yaml
from torch import nn


class TimeShuffler(nn.Module):
    def __init__(self, probability=0.5, data_sets=None, iterations=1e5):
        super().__init__()
        self.prob = nn.Parameter(torch.tensor(probability, dtype=torch.float32), requires_grad=False)
        self.data_sets = data_sets
        self.gamma = probability / iterations

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
    def __init__(self, start=10, end=1e-3, iterations=1e5, instruments=None):
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
                self._shuffle_times(batch[ds_key]['time'])
        return batch

    def _shuffle_times(self, time):
        time += torch.randn_like(time) * self.scaling

    def on_train_batch_end(self, *args, **kwargs):
        if self.scaling > self.end:
            new_gamma = self.scaling * self.gamma
            self.scaling.copy_(new_gamma)
        else:
            new_gamma = torch.zeros_like(self.scaling)
            self.scaling.copy_(new_gamma)
        wandb.log({'time_random': self.scaling.detach().cpu().numpy()})


def load_yaml_config(yaml_config_file, overwrite_args=None):
    overwrite_args = [] if overwrite_args is None else overwrite_args
    assert all([k.startswith('--') for k in overwrite_args[::2]]), \
        'Only accept --config and overwrite arguments (must start with --)'
    overwrite_args = {k.replace('--', ''): v for k, v in zip(overwrite_args[::2], overwrite_args[1::2])}
    with open(yaml_config_file) as f:
        config_str = f.read()
    for overwrite_key, overwrite_value in overwrite_args.items():
        config_str = config_str.replace('{%s}' % overwrite_key, overwrite_value)
    config = yaml.safe_load(config_str)
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
