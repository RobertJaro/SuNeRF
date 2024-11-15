import torch
import yaml
from torch import nn


class TimeShuffler(nn.Module):
    def __init__(self, probability=0.5, instruments=None):
        super().__init__()
        self.prob = probability
        self.instruments = instruments

    def forward(self, batch):
        if self.prob == 0:
            return batch
        for instr_key in batch.keys():
            if instr_key in self.instruments or self.instruments is None:
                self._shuffle_times(batch[instr_key]['time'], self.prob)
        return batch

    def _shuffle_times(self, time, prob):
        # shuffle random fraction of times
        n_shuffle = int(time.shape[0] * prob)
        idx = torch.randperm(n_shuffle)
        rand_idx = torch.randint(0, time.shape[0] - n_shuffle, (1,))
        time[rand_idx:rand_idx + n_shuffle] = time[rand_idx + idx]


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
