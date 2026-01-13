from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from astropy import units as u
from torch import nn
from torch._C._nn import linear
from torch.distributions import Normal
from torch.nn import Identity


class SirenModel(nn.Module):
    def __init__(self, in_dim, out_dim, dim=512, n_layers=8, w0=1., encoding_config=None, skip_layers=(2, 5)):
        super().__init__()

        encoding_config = {'type': 'default', 'w0': 30.} if encoding_config is None else encoding_config
        encoding_type = encoding_config.pop('type', 'default')

        if encoding_type == "default":
            self.posenc = SirenLayer(in_dim=in_dim, out_dim=dim, is_first=True, **encoding_config)
            posenc_dim = dim
        elif encoding_type == "positional":
            self.posenc = PositionalEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.d_output
        elif encoding_type == "identity":
            self.posenc = Identity()
            posenc_dim = in_dim
        elif encoding_type == "multi_spectral":
            self.posenc = MultispectralEncoding(in_dim=in_dim, **encoding_config)
            posenc_dim = self.posenc.d_output
        else:
            raise ValueError(f"Unknown encoding: {encoding_type}")

        self.num_layers = n_layers
        self.dim_hidden = dim
        self.skip_layers = skip_layers

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=posenc_dim, out_dim=dim, w0=w0)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            if i in self.skip_layers:
                # this layer will receive [h, skip_ref], width increases by posenc_dim
                in_d = dim + posenc_dim
            else:
                in_d = dim
            layer = SirenLayer(in_dim=in_d, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # initialize the output layer
        self.out_layer = SirenLayer(in_dim=dim, out_dim=out_dim, w0=w0, activation=nn.Identity())

    def forward(self, inp):
        inp_encoded = self.posenc(inp)  # apply positional encoding
        x = self.in_layer(inp_encoded)

        for i, layer in enumerate(self.layers):
            if i in self.skip_layers:
                x = torch.cat([x, inp_encoded], dim=-1)
                x = layer(x)  # layer expects dim + ref_dim
            else:
                x = layer(x)  # standard SIREN layer

        x = self.out_layer(x)
        return x

    def step(self, global_step):
        if hasattr(self.posenc, 'step'):
            self.posenc.step(global_step)


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
                in_dim=layer_dim_in,
                out_dim=dim,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not final_activation is not None else final_activation
        self.last_layer = SirenLayer(in_dim=dim, out_dim=out_dim, w0=w0, use_bias=use_bias, activation=final_activation)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)

        return self.last_layer(x)


class GenericModel(nn.Module):

    def __init__(self, in_dim, out_dim, dim=512, n_layers=8, encoding=None, activation='sine'):
        super().__init__()
        if encoding is None or encoding == 'none':
            self.d_in = nn.Linear(in_dim, dim)
        elif encoding == 'positional':
            posenc = PositionalEncoding(in_dim, 10)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        elif encoding == 'gaussian':
            posenc = GaussianPositionalEncoding(in_dim)
            d_in = nn.Linear(posenc.d_output, dim)
            self.d_in = nn.Sequential(posenc, d_in)
        else:
            raise NotImplementedError(f'Unknown encoding {encoding}')
        lin = [nn.Linear(dim, dim) for _ in range(n_layers)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, out_dim)
        activation_mapping = {'relu': nn.ReLU, 'swish': Swish, 'tanh': nn.Tanh, 'sine': Sine}
        activation_f = activation_mapping[activation]
        self.in_activation = activation_f()
        self.activations = nn.ModuleList([activation_f() for _ in range(n_layers)])

    def forward(self, x):
        x = self.in_activation(self.d_in(x))
        for l, a in zip(self.linear_layers, self.activations):
            x = a(l(x))
        x = self.d_out(x)
        return x


class EmissionModel(GenericModel):

    def __init__(self, n_channels=1, **kwargs):
        super().__init__(in_dim=4, out_dim=n_channels * 2, **kwargs)
        self.n_channels = n_channels

    def forward(self, x):
        out = super().forward(x)
        emission = torch.exp(out[..., :self.n_channels])
        alpha = nn.functional.relu(out[..., self.n_channels:])
        return {'emission': emission, 'alpha': alpha}


class ConditionedNeRF(nn.Module):

    def __init__(self, n_channels=1, z_dim=128, **kwargs):
        super().__init__()
        self.posenc = GaussianPositionalEncoding(3, scales=64, num_frequencies=32)
        dim_encoding = self.posenc.d_output
        encoding_config = {'type': 'identity'}
        self.nerf = SirenModel(in_dim=dim_encoding + z_dim, out_dim=n_channels * 2, dim=512, n_layers=8, encoding_config=encoding_config, **kwargs)
        self.n_channels = n_channels

    def forward(self, x, z):
        # x = (batch, 3)
        # z = (batch, dim_z)
        encoded = self.posenc(x)
        nerf_input = torch.cat([encoded, z], dim=-1)
        out = self.nerf(nerf_input)
        emission = torch.exp(out[..., :self.n_channels])
        alpha = nn.functional.relu(out[..., self.n_channels:])
        return {'emission': emission, 'alpha': alpha}


class ResBlock(nn.Module):
    def __init__(self, ch, norm=nn.GroupNorm, gn_groups=8):
        super().__init__()
        g = min(gn_groups, ch)
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            norm(g, ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
            norm(g, ch),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class ImageToLatentCNN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        z_dim=2048,
        base_channels=32,
        max_channels=512,
        gn_groups=8,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 7, stride=2, padding=3),
            nn.GroupNorm(min(gn_groups, base_channels), base_channels),
            nn.ReLU(inplace=True),
        )

        stages = []
        ch = base_channels

        for _ in range(4):  # 4 stages is usually enough
            stages += [
                ResBlock(ch, gn_groups=gn_groups),
                nn.Conv2d(ch, min(ch * 2, max_channels), 3, stride=2, padding=1),
            ]
            ch = min(ch * 2, max_channels)

        self.encoder = nn.Sequential(*stages)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch, z_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.scale * self.fc(x)

class CoordinateToLatentModel(GenericModel):
    def __init__(
        self,
        z_dim=32, **kwargs
    ):
        super().__init__(in_dim=4, out_dim=z_dim, dim=32, n_layers=4, **kwargs)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x = super().forward(x)
        return self.scale * x


class PlasmaModel(GenericModel):

    def __init__(self, log_T, decay_distance=2.0, encoding='positional', **kwargs):
        super().__init__(in_dim=4, out_dim=3, encoding=encoding, **kwargs)
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
        sigma = torch.sigmoid(sigma) + 0.01

        log_T_range = self.log_T.reshape([1] * (len(center_log_T.shape) - 1) + [-1])
        # log10 --> 10 ** (scaling) * exp(N) * (2 * pi * sigma ** 2) ** -0.5
        log10_e = 0.4342944819032518  # log10(e)
        log_ne = scaling - (log_T_range - center_log_T) ** 2 / (2 * sigma ** 2) * log10_e

        distance = torch.norm(x[..., :3], dim=-1)
        distance_threshold = torch.clip(distance - self.decay_distance, min=0, max=1) * 2
        log_ne = log_ne - distance_threshold[..., None]

        # compute total number density
        ne = 10 ** log_ne
        total_ne = torch.sum(ne, dim=-1, keepdim=True)
        total_log_ne = torch.log10(total_ne)

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


class SirenPlasmaModel(SirenModel):

    def __init__(self, log_T, decay_distance=2.0, **kwargs):
        super().__init__(in_dim=4, out_dim=3, **kwargs)
        self.log_T = nn.Parameter(torch.tensor(log_T, dtype=torch.float32), requires_grad=False)
        self.decay_distance = decay_distance

        self.T_range = nn.Parameter(torch.tensor([3.8, 8.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        radius = torch.norm(x[..., :3], dim=-1, keepdim=True)
        raw = super().forward(x)

        center_log_T, scaling, sigma = raw[..., 0:1], raw[..., 1:2], raw[..., 2:3]
        # velocity = raw[..., 3:]

        # assure that mean_log_T is in the range of the temperature bins
        # TODO: should we use fixed temperature range? filaments can be very cold 5e3 - 10e3 K?
        # maybe allow for very dense plasma in the cold temperature regime?
        center_log_T = torch.sigmoid(center_log_T) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        sigma = torch.sigmoid(sigma) + 0.01

        # scale density with radius ** -2
        scaling = scaling - 2 * torch.log10(radius)

        log_T_range = self.log_T.reshape([1] * (len(center_log_T.shape) - 1) + [-1])
        # log10 --> 10 ** (scaling) * exp(N) * (2 * pi * sigma ** 2) ** -0.5
        log10_e = 0.4342944819032518  # log10(e)
        log_ne = scaling - (log_T_range - center_log_T) ** 2 / (2 * sigma ** 2) * log10_e

        # distance = torch.norm(x[..., :3], dim=-1)
        # distance_threshold = torch.clip(distance - self.decay_distance, min=0, max=1) * 2
        # log_ne = log_ne - distance_threshold[..., None]
        # log_ne = torch.clamp(log_ne, min=-30)  # prevent negative infinity

        # compute total number density
        ne = 10 ** log_ne
        total_ne = torch.sum(ne, dim=-1, keepdim=True)
        total_log_ne = torch.log10(total_ne)

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


class AbsorptionModel(GenericModel):

    def __init__(self, freeze=False, **kwargs):
        super().__init__(in_dim=2, out_dim=1, n_layers=2, dim=16, **kwargs)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

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
            in_dim,
            out_dim,
            w0=1.,
            c=6.,
            is_first=False,
            use_bias=True,
            activation=None,
            dropout=0.
    ):
        super().__init__()
        self.dim_in = in_dim
        self.is_first = is_first

        weight = torch.zeros(out_dim, in_dim)
        bias = torch.zeros(out_dim) if use_bias else None
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


class PositionalEncoding(nn.Module):

    def __init__(self, in_dim, num_frequencies=32, min_frequencies=0, max_frequencies=6):
        super().__init__()
        num_frequencies = [num_frequencies] * in_dim if not isinstance(num_frequencies, Iterable) else num_frequencies
        min_frequencies = [min_frequencies] * in_dim if not isinstance(min_frequencies, Iterable) else min_frequencies
        max_frequencies = [max_frequencies] * in_dim if not isinstance(max_frequencies, Iterable) else max_frequencies

        assert len(num_frequencies) == in_dim, 'num_frequencies length must match input dimension (in_dim)'
        assert len(min_frequencies) == in_dim, 'd_min_frequencies length must match input dimension (in_dim)'
        assert len(max_frequencies) == in_dim, 'max_frequencies length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, min_freq, max_freq in zip(num_frequencies, min_frequencies, max_frequencies):
            f = 2 ** torch.linspace(min_freq, max_freq, num_freq) * torch.pi
            param = nn.Parameter(f, requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.d_output = sum([n * 2 for n in num_frequencies])

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = torch.einsum('i,...j->...ij', frequencies, x[..., i:i + 1])
            encoded = encoded.reshape(*x.shape[:-1], -1)
            encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_frequencies=32, scales=64):
        super().__init__()
        num_frequencies = [num_frequencies] * d_input if not isinstance(num_frequencies, Iterable) else num_frequencies
        scales = [scales] * d_input if not isinstance(scales, Iterable) else scales

        assert len(num_frequencies) == d_input, 'num_frequencies length must match input dimension (in_dim)'
        assert len(scales) == d_input, 'scales length must match input dimension (in_dim)'

        frequencies = []
        for num_freq, scale in zip(num_frequencies, scales):
            dist = Normal(loc=0, scale=scale)
            f = dist.sample([num_freq])
            param = nn.Parameter(f, requires_grad=False)
            frequencies.append(param)
        self.frequencies = nn.ParameterList(frequencies)

        self.d_output = int(sum([n * 2 for n in num_frequencies])) + d_input

        self.num_frequencies = num_frequencies

    def forward(self, x):
        encoded_coordinates = []
        for i, frequencies in enumerate(self.frequencies):
            encoded = torch.einsum('i,...j->...ij', torch.pi * frequencies, x[..., i:i + 1])
            encoded = encoded.reshape(*x.shape[:-1], -1)
            encoded = torch.cat([torch.sin(encoded), torch.cos(encoded)], -1)
            encoded_coordinates.append(encoded)
        encoded = torch.cat(encoded_coordinates + [x], -1)  # append original coordinates
        return encoded


class TimeSplitEncoding(nn.Module):
    def __init__(self, dim, w0_spatial=30.0, w0_time=1.0):
        super().__init__()
        self.spatial_layer = SirenLayer(3, dim, w0=w0_spatial, is_first=True)
        self.time_layer = SirenLayer(1, dim, w0=w0_time, is_first=True)
        self.d_output = 2 * dim

    def forward(self, x):
        time = x[..., 3:]
        spatial = x[..., :3]
        spatial_encoded = self.spatial_layer(spatial)
        time_encoded = self.time_layer(time)
        encoded = torch.cat([spatial_encoded, time_encoded], dim=-1)
        return encoded


class MultispectralEncoding(nn.Module):

    def __init__(self, in_dim, num_dims=64, weights=30.0):
        super().__init__()
        num_dims = [num_dims] * in_dim if not isinstance(num_dims, Iterable) else num_dims
        weights = [weights] * in_dim if not isinstance(weights, Iterable) else weights

        assert len(num_dims) == in_dim, 'num_dims length must match input dimension (in_dim)'
        assert len(weights) == in_dim, 'weights length must match input dimension (in_dim)'

        layers = []
        for nd, w in zip(num_dims, weights):
            l = SirenLayer(in_dim=1, out_dim=nd, w0=w, is_first=True)
            layers.append(l)
        self.layers = nn.ModuleList(layers)

        self.d_output = sum(num_dims)

    def forward(self, x):
        encoded_coordinates = []
        for i, layer in enumerate(self.layers):
            coord = x[:, i:i + 1]
            encoded = layer(coord)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded
