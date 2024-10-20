import torch
from torch import nn

from sunerf.model.model import PlasmaModel
from sunerf.rendering.base_tracing import SuNeRFRendering, cumprod_exclusive


class PlasmaRadiativeTransfer(SuNeRFRendering):

    def __init__(self, log_T, model_config=None, absorption=True, **kwargs):
        model_config = {} if model_config is None else model_config
        coarse_model = PlasmaModel(log_T=log_T, **model_config)
        fine_model = PlasmaModel(log_T=log_T, **model_config)
        super().__init__(coarse_model=coarse_model, fine_model=fine_model, **kwargs)
        self.absorption = absorption
        print('Using absorption:', absorption)

    def raw2outputs(self, raw: dict, z_vals: torch.Tensor, rays_d: torch.Tensor,
                    temperature_response: nn.Module, instrument_scaling: nn.Parameter, absorption_model: nn.Module,
                    **kwargs):
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

        log_ne = raw['log_ne']  # n_batches, n_samples, n_T_bins --> EM per bin
        total_ne = raw['total_ne']
        mean_log_T = raw['mean_log_T']
        total_log_ne = raw['total_log_ne']

        # intensity = raw['emission']
        # alpha = raw['alpha']
        # ne = intensity
        # log_T = intensity

        # find channel response for temperature and weight by electron density squared
        # assume dirac delta function for T distribution

        # intensity = torch.einsum('...ij,...i->...j', temperature_response, emission_measure)
        intensity = temperature_response[None, None, :, :] + 2 * log_ne[..., None]
        intensity = 10 ** (intensity + instrument_scaling)
        intensity = intensity.sum(-2)  # integrate over temperature bins
        #
        # # learn absorption based on electron density
        if self.absorption:
            absorption_input = torch.cat([total_log_ne, mean_log_T], dim=-1)
            log_kappa = absorption_model(absorption_input)['log_kappa']
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
        weights = weights / (weights.sum(1)[:, None] + 1e-10)

        mean_log_T = (mean_log_T * total_ne).sum(1) / total_ne.sum(1)
        total_density = total_ne.sum(1)
        mean_absorption = (1 - absorption).sum(1).mean(-1)

        # print('MIN/MAX log T', log_T.min(), log_T.max(), log_T.shape)
        # print('MIN/MAX log ne', log_ne.min(), log_ne.max(), log_ne.shape)
        # print('MIN/MAX TR', response.min(), response.max(), response.shape)
        # print('MIN/MAX INTENSITY', intensity.min(), intensity.max(), intensity.shape)
        # print('MIN/MAX INTEGRATED ABSORPTION', integrated_absorption.min(), integrated_absorption.max(),
        #       integrated_absorption.shape)
        # print('MIN/MAX INTEGRATED INTENSITY', integrated_intensity.min(), integrated_intensity.max(),
        #       integrated_intensity.shape)

        return {'image': integrated_intensity, 'weights': weights, 'mean_absorption': mean_absorption,
                'mean_T': mean_log_T, 'total_ne': total_density}
