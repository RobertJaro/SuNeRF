import numpy as np
import torch
from astropy import units as u
from torch import nn
from torch._C._nn import linear

from sunerf.model.generic_model import GenericModel


class SirenNet(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            dim=512,
            n_layers=8,
            w0=1.,
            w0_initial=30.,
            use_bias=True,
            final_activation=None,
            dropout=0.
    ):
        super().__init__()
        self.num_layers = n_layers
        self.dim_hidden = dim

        self.layers = nn.ModuleList([])
        for ind in range(n_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = in_dim if is_first else dim

            layer = SirenLayer(
                dim_in=layer_dim_in,
                dim_out=dim,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not final_activation is not None else final_activation
        self.last_layer = SirenLayer(dim_in=dim, dim_out=out_dim, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)


class EmissionModel(SirenNet):

    def __init__(self, n_channels=1, **kwargs):
        super().__init__(in_dim=4, out_dim=n_channels * 2, **kwargs)
        self.n_channels = n_channels

    def forward(self, x):
        out = super().forward(x)
        emission = torch.exp(out[..., :self.n_channels])
        alpha = nn.functional.relu(out[..., self.n_channels:])
        return {'emission': emission, 'alpha': alpha}


class PlasmaModel(GenericModel):

    def __init__(self, log_T, decay_distance=2.0, **kwargs):
        super().__init__(in_dim=4, out_dim=3, encoding='gaussian', **kwargs)
        self.log_T = nn.Parameter(torch.tensor(log_T, dtype=torch.float32), requires_grad=False)
        self.decay_distance = decay_distance

        self.T_range = nn.Parameter(torch.tensor([3.8, 8.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        raw = super().forward(x)

        center_log_T, scaling, sigma = raw[..., 0:1], raw[..., 1:2], raw[..., 2:3]
        # velocity = raw[..., 3:]

        # assure that mean_log_T is in the range of the temperature bins
        # TODO: should we use fixed temperature range? filaments can be very cold 5e3 - 10e3 K?
        # maybe allow for very dense plasma in the cold temperature regime?
        center_log_T = torch.sigmoid(center_log_T) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        sigma = torch.sigmoid(sigma) + 1e-2

        log_T_range = self.log_T.reshape([1] * (len(center_log_T.shape) - 1) + [-1])
        # log10 --> 10 ** (scaling) * exp(N) * (2 * pi * sigma ** 2) ** -0.5
        log10_e = 0.4342944819032518  # log10(e)
        log_ne = (scaling - (log_T_range - center_log_T) ** 2 / (2 * sigma ** 2) * log10_e -
                  0.5 * torch.log10(2 * torch.pi * sigma ** 2))

        distance = torch.norm(x[..., :3], dim=-1)
        distance_threshold = torch.clip(distance - self.decay_distance, min=0, max=1) * 2
        log_ne = log_ne - distance_threshold[..., None]
        # log_ne = torch.clamp(log_ne, min=-30)  # prevent negative infinity

        # compute total number density
        ne = 10 ** log_ne
        total_ne = torch.sum(ne, dim=-1, keepdim=True)
        total_log_ne = torch.log10(total_ne)

        # assert not torch.isnan(log_ne).any(), 'NaN in log_ne'
        # assert not torch.isnan(total_ne).any(), 'NaN in total_ne'
        # assert not torch.isnan(total_log_ne).any(), 'NaN in total_log_ne'

        # compute mean Temperature
        temperatures = 10 ** log_T_range
        mean_temperature = (temperatures * ne).sum(-1, keepdim=True) / total_ne
        mean_log_T = torch.log10(mean_temperature)
        # replace invalid values
        mean_log_T[torch.isnan(mean_log_T)] = center_log_T[torch.isnan(mean_log_T)]

        return {'log_ne': log_ne,
                'mean_log_T': mean_log_T,
                'total_ne': total_ne,
                'total_log_ne': total_log_ne,
                'ne': ne, 'sigma': sigma
                }


class RhoModel(SirenNet):

    def __init__(self, Rs_per_ds, seconds_per_dt, **kwargs):
        super().__init__(in_dim=4, out_dim=4, **kwargs)
        v = 300 * (u.km / u.s)
        v = v.to_value(u.solRad / u.s) / Rs_per_ds * seconds_per_dt  # normalize to model units
        self.v_radial = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
        v_scale = 10 * (u.km / u.s)
        v_scale = v_scale.to_value(u.solRad / u.s) / Rs_per_ds * seconds_per_dt  # normalize to model units
        self.v_scale = nn.Parameter(torch.tensor(v_scale, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        coords = x
        radial_distance = torch.norm(coords[..., :3], dim=-1, keepdim=True)
        radial = coords[..., :3] / (radial_distance + 1e-8)

        x = super().forward(x)
        log_rho = x[..., 0:1] - 2 * torch.log(radial_distance)
        rho = torch.exp(log_rho)

        v = self.v_radial * radial
        v = v + x[..., 1:] * self.v_scale

        result = {'log_rho': log_rho, 'rho': rho, 'v': v}
        return result


class VelocityModel(SirenNet):

    def __init__(self, **kwargs):
        super().__init__(in_dim=4, out_dim=3, **kwargs)

    def forward(self, x):
        v = super().forward(x)
        return {'v': v}


class AbsorptionModel(SirenNet):

    def __init__(self, **kwargs):
        super().__init__(in_dim=2, out_dim=1, w0_initial=1, n_layers=2, dim=16, **kwargs)

    def forward(self, x):
        log_kappa = super().forward(x) - 2
        return {'log_kappa': log_kappa, 'kappa': 10 ** log_kappa}


class ConstantAbsorptionModel(nn.Module):

    def __init__(self, coefficient=-5, **kwargs):
        super().__init__()
        self.coefficient = coefficient

    def forward(self, x):
        total_log_ne = x[..., 0:1]
        mean_log_T = x[..., 1:2]
        # log_kappa = total_log_ne - mean_log_T + self.offset
        log_kappa = self.coefficient
        return {'log_kappa': log_kappa, 'kappa': 10 ** log_kappa}


class SirenLayer(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            w0=1.,
            c=6.,
            is_first=False,
            use_bias=True,
            activation=None,
            dropout=0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)
