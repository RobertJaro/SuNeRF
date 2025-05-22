import copy

import numpy as np
import torch
from torch import nn

from sunerf.model.model import AbsorptionModel, ConstantAbsorptionModel
from sunerf.rendering.base_tracing import cumprod_exclusive
from sunerf.train.convert_temperature_response_function import convert_response_function


class PlasmaRadiativeTransfer(nn.Module):

    def __init__(self, temperature_response_config, log_T_range, absorption_model=None):
        super().__init__()

        # load temperature response function
        normalization = temperature_response_config.get('normalization', None)
        channels = temperature_response_config.get('channels', None)
        temperature, response, normalization = convert_response_function(temperature_response_config['file'],
                                                                         log_T_range=log_T_range,
                                                                         normalization=normalization, channels=channels)

        log_T = np.log10(temperature)
        log_response = np.log10(response)

        self.absorption = absorption_model is not None
        print('Using absorption:', self.absorption)

        log_response = nn.Parameter(torch.tensor(log_response.T, dtype=torch.float32), requires_grad=False)

        instrument_scaling = nn.Parameter(torch.tensor(temperature_response_config['scaling'], dtype=torch.float32),
                                          requires_grad=temperature_response_config['learnable'])

        self.log_T = torch.from_numpy(log_T).float()
        self.temperature_response = log_response
        self.instrument_scaling = instrument_scaling
        self.absorption_model = absorption_model
        self.normalization = normalization

    def forward(self, log_ne, total_ne, mean_log_T, total_log_ne, z_vals: torch.Tensor,
                rays_d: torch.Tensor, query_points: torch.Tensor, **kwargs):
        r"""
        Convert the raw NeRF output into emission and absorption.

        raw: output of NeRF, 2 values per sampled point
        z_vals: distance along the ray as measure from the origin
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        # compute line element (dz) for integration
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists[..., :1], dists], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
        dists = dists[:, :, None]

        # emission_measure = raw['emission_measure']
        # ne = raw['ne']
        # T = raw['T']
        # log_ne = raw['log_ne']
        # log_T = raw['log_T']

        # intensity = raw['emission']
        # alpha = raw['alpha']
        # ne = intensity
        # log_T = intensity

        # find channel response for temperature and weight by electron density squared
        # assume dirac delta function for T distribution

        # intensity = torch.einsum('...ij,...i->...j', temperature_response, emission_measure)
        intensity = self.temperature_response[None, None, :, :] + 2 * log_ne[..., None]
        intensity = 10 ** (intensity + self.instrument_scaling)
        intensity = intensity.sum(-2)  # integrate over temperature bins
        #
        # # learn absorption based on electron density
        if self.absorption:
            absorption_input = torch.cat([total_log_ne, mean_log_T], dim=-1)
            log_kappa = self.absorption_model(absorption_input)['log_kappa']
            alpha = 10 ** (log_kappa + total_log_ne)
        else:
            alpha = torch.zeros_like(dists)  # ignore absorption for now

        # transmission per sampled point [n_rays, n_samples]
        absorption = torch.exp(-alpha * dists)
        # [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

        # compute total absorption for each light ray (intensity)
        # how much light is transmitted from each sampled point
        # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)
        integrated_absorption = cumprod_exclusive(absorption + 1e-10, dim=1)
        # [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
        # apply absorption to intensities
        emerging_intensity = intensity * integrated_absorption  # integrate total intensity [n_rays, n_samples - 1]
        # sum all intensity contributions
        integrated_intensity = (emerging_intensity * dists).sum(1)

        # set the weigths to the intensity contributions (sample primary contributing regions)
        weights = emerging_intensity.mean(-1)
        weights = weights / (weights.sum(1, keepdim=True) + 1e-10)

        mean_log_T = (mean_log_T * total_ne).sum(1) / total_ne.sum(1)
        total_density = total_ne.sum(1)
        mean_absorption = (1 - absorption).sum(1).mean(-1)

        # visualization outputs
        distance = query_points[..., :3].pow(2).sum(-1).pow(0.5)
        height_map = (weights * distance).sum(-1)

        # DEM
        em = (10 ** log_ne).sum(-1)
        dem = (10 ** log_ne).sum(1)

        return {'image': integrated_intensity, 'weights': weights, 'mean_absorption': mean_absorption,
                'log_ne': log_ne, 'mean_T': mean_log_T, 'total_ne': total_density, 'height_map': height_map,
                'distance': distance, 'em': em, 'dem': dem,
                'z_vals': z_vals, 'query_points': query_points}


def init_absorption_model(absorption_config):
    absorption_config = copy.deepcopy(absorption_config)
    absorption_type = absorption_config.pop('type', None)
    if absorption_type == 'constant':
        return ConstantAbsorptionModel(**absorption_config)
    elif absorption_type == 'learned':
        return AbsorptionModel(**absorption_config)
    elif absorption_type is None:
        return None
    else:
        raise NotImplementedError(f"Absorption type {absorption_type} not implemented.")