from typing import Tuple, Optional

import torch
from torch import nn


class NeRF(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4,),
            d_viewdirs: Optional[int] = None
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = Sine()
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, d_output)

    def forward(
            self,
            x: torch.Tensor,
            viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, scale_factor: float = 2., log_space: bool = False):
        """

        Parameters
        ----------
        d_input: number of input dimensions
        n_freqs: number of frequencies used for encoding
        scale_factor: factor to adjust box size limit of 2pi (default 2; 4pi)
        log_space: use frequencies in powers of 2
        """
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2. ** torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.n_freqs - 1), self.n_freqs)
            print('freq bands', freq_bands)

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq / scale_factor))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq / scale_factor))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class NeRF_absortpion(nn.Module):
    r"""
    Neural radiance fields module.
    """

    def __init__(
            self,
            d_input: int = 4,
            d_output: int = 2,
            absortpion_output: int = 1,
            n_layers: int = 8,
            d_filter: int = 256,
            skip: Tuple[int] = (4,)
    ):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)] +
            [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
                 else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
        )

        # If no viewdirs, use simpler output
        self.output = nn.Linear(d_filter, d_output)
        self.abs_coeff = nn.Parameter(torch.rand(1, 1))

    def forward(
            self,
            x: torch.Tensor,
            viewdirs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Forward pass with optional view direction.
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Simple output
        x = torch.cat(self.output(x), self.abs_coeff, dim=-1)

        return x


def init_models(
        d_input: int,
        d_output: int,
        n_freqs: int,
        n_layers: int,
        d_filter: int,
        log_space: bool,
        use_fine_model: bool,
        skip: Tuple[int],
):
    r"""_summary_

    Initialize models, and encoders for NeRF training.

    Args:
        d_input (int): Number of input dimensions (x,y,z,t)
        d_output (int): wavelength absorption + emission
        n_freqs (int): Number of encoding functions for samples
        n_layers (int): Number of layers in network bottleneck
        d_filter (int): Dimensions of linear layer filters
        log_space (bool): If set, frequencies scale in log space
        use_fine_model (bool): If set, creates a fine model
        skip (Tuple[int]): Layers at which to apply input residual
    Returns:
        coarse_model:
        fine_model:
        encode:
    """

    # Encoders
    encoder = PositionalEncoder(
        d_input,
        n_freqs,
        log_space=log_space
    )
    encode = lambda x: encoder(x)

    # Models
    coarse_model = NeRF(
        encoder.d_output,
        d_output,
        n_layers=n_layers,
        d_filter=d_filter,
        skip=skip)
    model_params = list(coarse_model.parameters())
    if use_fine_model:
        fine_model = NeRF(
            encoder.d_output,
            d_output,
            n_layers=n_layers,
            d_filter=d_filter,
            skip=skip
        )
        model_params = model_params + list(fine_model.parameters())
    else:
        fine_model = None

    return coarse_model, fine_model, encode, model_params
