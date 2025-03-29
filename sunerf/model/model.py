import torch
from torch import nn
from torch.distributions import Normal


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

    def __init__(self, n_channels=1, encoding='positional', **kwargs):
        super().__init__(in_dim=4, out_dim=n_channels * 2, encoding=encoding, **kwargs)
        self.n_channels = n_channels

    def forward(self, x):
        out = super().forward(x)
        emission = torch.exp(out[..., :self.n_channels])
        alpha = nn.functional.relu(out[..., self.n_channels:])
        return {'emission': emission, 'alpha': alpha}


class PlasmaModel(GenericModel):

    def __init__(self, log_T, decay_distance=2.0, encoding='gaussian', **kwargs):
        super().__init__(in_dim=4, out_dim=3, encoding=encoding, **kwargs)
        self.log_T = nn.Parameter(log_T, requires_grad=False)
        self.decay_distance = decay_distance

        self.T_range = nn.Parameter(torch.tensor([3.8, 8.0], dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        raw = super().forward(x)

        mean_log_T, scaling, sigma = raw[..., 0:1], raw[..., 1:2], raw[..., 2:3]
        # velocity = raw[..., 3:]

        # assure that mean_log_T is in the range of the temperature bins
        # TODO: should we use fixed temperature range? filaments can be very cold 5e3 - 10e3 K?
        # maybe allow for very dense plasma in the cold temperature regime?
        mean_log_T = torch.sigmoid(mean_log_T) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        sigma = torch.sigmoid(sigma) + 0.01

        log_T_range = self.log_T.reshape([1] * (len(mean_log_T.shape) - 1) + [-1])
        # log10 --> 10 ** (scaling) * exp(N) * (2 * pi * sigma ** 2) ** -0.5
        log10_e = 0.4342944819032518  # log10(e)
        log_ne = (scaling -
                  ((log_T_range - mean_log_T) ** 2 / (2 * sigma ** 2)) * log10_e -
                  0.5 * torch.log10(2 * torch.pi * sigma ** 2))

        distance = torch.norm(x[..., :3], dim=-1)
        distance_threshold = torch.clip(distance - self.decay_distance, min=0, max=1) * 2
        log_ne = log_ne - distance_threshold[..., None]

        ne = 10 ** log_ne
        total_ne = ne.sum(-1)[..., None]
        total_log_ne = torch.log10(total_ne)

        return {'log_ne': log_ne, 'log_T': self.log_T,
                'mean_log_T': mean_log_T,
                'total_ne': total_ne,
                'total_log_ne': total_log_ne,
                'ne': ne
                # 'velocity': velocity
                }


class RhoModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=4, out_dim=4, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        log_rho = x[..., 0:1]
        v = x[..., 1:]
        rho = 10 ** log_rho
        result = {'log_rho': log_rho, 'rho': rho, 'v': v}
        return result


class VelocityModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=4, out_dim=3, **kwargs)

    def forward(self, x):
        v = super().forward(x)
        return {'v': v}


class AbsorptionModel(GenericModel):

    def __init__(self, **kwargs):
        super().__init__(in_dim=2, out_dim=1, **kwargs)

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


class Sine(nn.Module):
    def __init__(self, w0: float = 1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class TrainablePositionalEncoding(nn.Module):

    def __init__(self, d_input, n_freqs=20):
        super().__init__()
        frequencies = torch.stack([torch.linspace(-3, 9, n_freqs, dtype=torch.float32) for _ in range(d_input)], -1)
        self.frequencies = nn.Parameter(frequencies[None, :, :], requires_grad=True)
        self.d_output = n_freqs * 2 * d_input

    def forward(self, x):
        # x = (batch, rays, coords)
        encoded = x[:, None, :] * torch.pi * 2 ** self.frequencies
        normalization = (torch.pi * 2 ** self.frequencies)
        encoded = torch.cat([torch.sin(encoded) / normalization, torch.cos(encoded) / normalization], -1)
        encoded = encoded.reshape(x.shape[0], -1)
        return encoded


class PositionalEncoding(nn.Module):

    def __init__(self, in_features, num_freqs=10, max_freq=9):
        super().__init__()
        frequencies = 2 ** torch.linspace(-max_freq, max_freq, num_freqs)
        self.frequencies = nn.Parameter(frequencies, requires_grad=False)
        self.d_output = in_features * (1 + num_freqs * 2)

    def forward(self, x):
        encoded = torch.einsum('...i,j->...ij', x, self.frequencies)
        encoded = torch.cat([
            torch.einsum('...j,j->...j', torch.sin(encoded), self.frequencies.pow(-1)).reshape(*x.shape[:-1], -1),
            torch.einsum('...j,j->...j', torch.cos(encoded), self.frequencies.pow(-1)).reshape(*x.shape[:-1], -1),
            x], -1)
        return encoded


class GaussianPositionalEncoding(nn.Module):

    def __init__(self, d_input, num_freqs=128, scale=512):
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
