from typing import Iterable

import numpy as np
import torch
from astropy import units as u
from torch import nn
from torch.distributions import Normal
from torch.nn import Identity
from torch.nn.functional import linear

from sunerf.train.coordinate_transformation import to_carrington_rotation_frame


class SirenModel(nn.Module):
    def __init__(self, in_dim, out_dim, dim=512, n_layers=8, w0=1., w0_init=1, input_weights=None, **kwargs):
        super().__init__()

        self.num_layers = n_layers
        self.dim_hidden = dim

        self.input_weights = nn.Parameter(torch.tensor(input_weights, dtype=torch.float32), requires_grad=False) if input_weights is not None else None

        # initialize the input layer
        self.in_layer = SirenLayer(in_dim=in_dim, out_dim=dim, w0=w0_init, is_first=True)

        # initialize the hidden layers
        layers = []
        for i in range(n_layers - 1):
            layer = SirenLayer(in_dim=dim, out_dim=dim, w0=w0)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # initialize the output layer
        self.out_layer = nn.Linear(dim, out_dim)

    def forward(self, inp):
        if self.input_weights is not None:
            inp = inp * self.input_weights
        x = self.in_layer(inp)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.out_layer(x)
        return x

    def step(self, global_step):
        pass


class SplitTemporalSirenModel(nn.Module):
    def __init__(
            self,
            spatial_dim=128,
            spatial_layers=8,
            time_dim=32,
            time_layers=2,
            fusion_dim=128,
            fusion_layers=2,
            w0_spatial=30.,
            w0_time=1.,
            alpha=0.1,
            cold_steps=5e4,
            warm_steps=1e4,
            output_dim=4,
            **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.alpha_max = nn.Parameter(torch.tensor(alpha, dtype=torch.float32), requires_grad=False)
        self.current_alpha = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=False)
        self.cold_steps = nn.Parameter(torch.tensor(int(cold_steps), dtype=torch.int64), requires_grad=False)
        self.warm_steps = nn.Parameter(torch.tensor(int(warm_steps), dtype=torch.int64), requires_grad=False)

        spatial_blocks = [SirenLayer(in_dim=3, out_dim=spatial_dim, w0=w0_spatial, is_first=True)]
        spatial_blocks.extend(
            SirenLayer(in_dim=spatial_dim, out_dim=spatial_dim, w0=1)
            for _ in range(max(spatial_layers - 1, 0))
        )
        self.spatial_net = nn.Sequential(*spatial_blocks)

        time_blocks = [SirenLayer(in_dim=1, out_dim=time_dim, w0=w0_time, is_first=True)]
        time_blocks.extend(
            SirenLayer(in_dim=time_dim, out_dim=time_dim, w0=1)
            for _ in range(max(time_layers - 1, 0))
        )
        self.time_net = nn.Sequential(*time_blocks)

        fusion_in_dim = spatial_dim + time_dim
        fusion_blocks = [SirenLayer(in_dim=fusion_in_dim, out_dim=fusion_dim, w0=1)]
        fusion_blocks.extend(
            SirenLayer(in_dim=fusion_dim, out_dim=fusion_dim, w0=1)
            for _ in range(max(fusion_layers - 1, 0))
        )
        self.fusion_net = nn.Sequential(*fusion_blocks)

        self.head_static = nn.Linear(spatial_dim, output_dim)
        self.head_dynamic = nn.Linear(fusion_dim, output_dim)

    def forward(self, inp):
        spatial = inp[..., :3]
        time = inp[..., 3:4]

        spatial_feat = self.spatial_net(spatial)
        time_feat = self.time_net(time)

        dynamic_feat = torch.cat([spatial_feat, time_feat], dim=-1)
        dynamic_feat = self.fusion_net(dynamic_feat)

        out_static = self.head_static(spatial_feat)
        out_dynamic = self.head_dynamic(dynamic_feat)
        return out_static + self.current_alpha * out_dynamic

    def step(self, global_step):
        step = int(global_step)
        cold_steps = int(self.cold_steps.item())
        warm_steps = int(self.warm_steps.item())
        alpha_max = float(self.alpha_max.item())

        if step < cold_steps:
            alpha = 0.0
        elif warm_steps <= 0:
            alpha = alpha_max
        elif step < cold_steps + warm_steps:
            progress = (step - cold_steps) / warm_steps
            alpha = alpha_max * progress
        else:
            alpha = alpha_max

        self.current_alpha.copy_(
            torch.tensor(alpha, dtype=self.current_alpha.dtype, device=self.current_alpha.device)
        )


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
            final_activation=None
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
                is_first=is_first
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


class RhoModel(nn.Module):

    def __init__(self, Rs_per_ds, seconds_per_dt, static=False, use_carrington_projection=True,
                 model_type='split_temporal', **kwargs):
        super().__init__()
        v = 300 * (u.km / u.s)
        v = v.to_value(u.solRad / u.s) / Rs_per_ds * seconds_per_dt  # normalize to model units
        self.v_radial = nn.Parameter(torch.tensor(v, dtype=torch.float32), requires_grad=False)
        v_scale = 100 * (u.km / u.s)
        v_scale = v_scale.to_value(u.solRad / u.s) / Rs_per_ds * seconds_per_dt  # normalize to model units
        self.v_scale = nn.Parameter(torch.tensor(v_scale, dtype=torch.float32), requires_grad=False)
        self.seconds_per_dt = seconds_per_dt

        self.static = static
        self.use_carrington_projection = use_carrington_projection
        if static:
            self.model = SirenModel(in_dim=3, out_dim=4, **kwargs)
        elif model_type == 'split_temporal':
            self.model = SplitTemporalSirenModel(output_dim=4, **kwargs)
        elif model_type in {'default', 'shared'}:
            self.model = SirenModel(in_dim=4, out_dim=4, **kwargs)
        else:
            raise ValueError(f"Unknown RhoModel model_type: {model_type}")

    def forward(self, coords):
        radial_distance = torch.norm(coords[..., :3], dim=-1, keepdim=True)
        radial = coords[..., :3] / (radial_distance + 1e-8)

        if self.use_carrington_projection:
            coords = to_carrington_rotation_frame(coords, self.seconds_per_dt)

        if self.static:
            model_out = self.model(coords[..., :3])
        else:
            model_out = self.model(coords)
        log_rho = model_out[..., 0:1] - 2 * torch.log(radial_distance)
        rho = torch.exp(log_rho)

        v = self.v_radial * radial + self.v_scale * model_out[..., 1:4]

        result = {'log_rho': log_rho, 'rho': rho, 'v': v}
        return result

    def step(self, global_step):
        if hasattr(self.model, 'step'):
            self.model.step(global_step)


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
            activation=None
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

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (np.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


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

    def __init__(self, d_input, num_freqs=32, scale=4):
        super().__init__()
        dist = Normal(loc=0, scale=scale)
        frequencies = dist.sample([num_freqs, d_input])
        self.frequencies = nn.Parameter(2 * torch.pi * frequencies, requires_grad=False)
        self.d_output = d_input * (num_freqs * 2 + 1)

    def forward(self, x):
        encoded = torch.einsum('...j,ij->...ij', x, self.frequencies)
        encoded = encoded.reshape(*x.shape[:-1], -1)
        encoded = torch.cat([x, torch.sin(encoded), torch.cos(encoded)], -1)
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
        assert x.shape[-1] == len(self.layers), f'Input dimension {x.shape[-1]} must match number of layers {len(self.layers)}'
        encoded_coordinates = []
        for i, layer in enumerate(self.layers):
            coord = x[..., i:i + 1]
            encoded = layer(coord)
            encoded_coordinates.append(encoded)

        encoded = torch.cat(encoded_coordinates, -1)
        return encoded
