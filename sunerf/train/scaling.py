import numpy as np
import torch
from torch import nn


class ImageLogScaling(nn.Module):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = nn.Parameter(torch.tensor(vmin, dtype=torch.float32), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=False)

    def forward(self, image):
        image = (torch.log(image) - self.vmin) / (self.vmax - self.vmin)
        return image


class ImageAsinhScaling(nn.Module):

    def __init__(self, vmax=1, a=0.005):
        super().__init__()
        self.normalization = nn.Parameter(torch.tensor(np.arcsinh(1 / a), dtype=torch.float32), requires_grad=False)
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=False)

    def forward(self, image):
        image = image / self.vmax
        image = torch.asinh(image / self.a) / self.normalization
        return image
