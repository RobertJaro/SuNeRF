import os

import numpy as np
from torch.utils.data import Dataset


class MmapDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_file = list(self.batches_file_paths.values())[0]
        n_batches = np.ceil(np.load(ref_file, mmap_mode='r').shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        # lazy load data
        data = {k: np.copy(np.load(bf, mmap_mode='r')[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, bf in self.batches_file_paths.items()}
        return data

    def clear(self):
        [os.remove(f) for f in self.batches_file_paths.values()]


class ArrayDataset(Dataset):

    def __init__(self, array_dict, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.array_dict = array_dict
        self.batch_size = int(batch_size)

    def __len__(self):
        ref_array = list(self.array_dict.values())[0]
        n_batches = np.ceil(ref_array.shape[0] / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        data = {k: np.copy(v[idx * self.batch_size: (idx + 1) * self.batch_size])
                for k, v in self.array_dict.items()}
        return data
