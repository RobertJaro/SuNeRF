# sunerf/train/render_mode.py
from enum import IntEnum

import torch
from torch.utils.data import Dataset


class RenderMode(IntEnum):
    INSTRUMENT = 0   # normal instrument rendering (current: 'instrument' in batch)
    BACKGROUND = 1   # star background only (new explicit mode)
    QUERY_POINTS = 2         # query_points pred-only (current: else branch)
    REFERENCE = 3    # query_points + rho GT (current: 'rho' in batch)


class RenderModeDataset(Dataset):
    """
    Dataset wrapper that injects a render_mode field into each sample.

    Similar to IndexedDataset, but for inference/rendering behavior.
    """

    def __init__(self, dataset, render_mode: RenderMode | int,
                 key: str = "render_mode",
                 instrument: str | None = None):
        """
        :param dataset: base dataset
        :param render_mode: enum or int (RenderMode)
        :param key: key used in batch dict
        :param instrument: optional constant instrument string to inject
        """
        self.dataset = dataset
        self.key = key
        self.render_mode = int(render_mode)
        self.instrument = instrument

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # inject render mode (scalar tensor)
        data[self.key] = torch.tensor(self.render_mode, dtype=torch.long)

        return data