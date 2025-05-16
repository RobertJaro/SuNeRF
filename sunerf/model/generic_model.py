import torch
from torch import nn
from torch.distributions import Normal

from sunerf.model.model import Swish, Sine


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
