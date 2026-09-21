import numpy as np
import torch
from torch import nn


def _divisor_tensor(divisor):
    # Scalar or one fixed value per channel (last image axis).
    divisor = torch.as_tensor(divisor, dtype=torch.float32)
    if divisor.ndim > 1:
        raise ValueError('Image scaling divisor must be a scalar or a per-channel vector.')
    if not torch.isfinite(divisor).all() or torch.any(divisor <= 0):
        raise ValueError('Image scaling divisors must be finite and positive.')
    return divisor


class ImageLogScaling(nn.Module):
    def __init__(self, vmin=0, vmax=1, divisor=1.0):
        super().__init__()
        self.vmin = nn.Parameter(torch.tensor(vmin, dtype=torch.float32), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=False)
        # Rebuilt from the configuration; kept out of the state dict so existing
        # checkpoints stay loadable.
        self.register_buffer('divisor', _divisor_tensor(divisor), persistent=False)

    def forward(self, image):
        image = torch.clamp(image / self.divisor, min=1e-8)
        image = (torch.log(image) - self.vmin) / (self.vmax - self.vmin)
        return image


class ImageAsinhScaling(nn.Module):

    def __init__(self, vmax=1, a=0.005, divisor=1.0):
        super().__init__()
        self.normalization = nn.Parameter(torch.tensor(np.arcsinh(1 / a), dtype=torch.float32), requires_grad=False)
        self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=False)
        # Rebuilt from the configuration; kept out of the state dict so existing
        # checkpoints stay loadable.
        self.register_buffer('divisor', _divisor_tensor(divisor), persistent=False)

    def forward(self, image):
        image = image / self.divisor / self.vmax
        image = torch.asinh(image / self.a) / self.normalization
        return image

class ImageLinearScaling(nn.Module):

    def __init__(self, vmin=0, vmax=1, divisor=1.0):
        super().__init__()
        self.vmin = nn.Parameter(torch.tensor(vmin, dtype=torch.float32), requires_grad=False)
        self.vmax = nn.Parameter(torch.tensor(vmax, dtype=torch.float32), requires_grad=False)
        # Rebuilt from the configuration; kept out of the state dict so existing
        # checkpoints stay loadable.
        self.register_buffer('divisor', _divisor_tensor(divisor), persistent=False)

    def forward(self, image):
        image = (image / self.divisor - self.vmin) / (self.vmax - self.vmin)
        return image
