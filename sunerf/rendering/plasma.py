import copy

import numpy as np
import torch
from torch import nn

from sunerf.model.model import PlasmaModel, AbsorptionModel, ConstantAbsorptionModel
from sunerf.rendering.base_tracing import MultiResolutionRenderingModule, cumprod_exclusive


class PlasmaRadiativeTransfer(MultiResolutionRenderingModule):

    def __init__(self, temperature_response_config, model_config=None, absorption_config=None, **kwargs):

        temperature = np.load(temperature_response_config[0]['file'])['temperature']
        self.log_T = torch.from_numpy(temperature).float()

        model_config = {} if model_config is None else model_config
        coarse_model = PlasmaModel(log_T=self.log_T, **model_config)
        fine_model = PlasmaModel(log_T=self.log_T, **model_config)
        super().__init__(coarse_model=coarse_model, fine_model=fine_model, **kwargs)
        self.absorption = absorption_config is not None and absorption_config['type'] is not None
        print('Using absorption:', self.absorption)

        temperature_response = [np.load(c['file'])['response'] for c in temperature_response_config]
        temperature_response = [nn.Parameter(torch.tensor(v.T, dtype=torch.float32), requires_grad=False)
                                for v in temperature_response]
        instrument_scaling = [
            nn.Parameter(torch.tensor(c['scaling'], dtype=torch.float32), requires_grad=c['learnable'])
            for c in temperature_response_config]
        temperature_response_mapping = {inst_key: i for i, c in enumerate(temperature_response_config)
                                        for inst_key in c['instruments']}

        # assert same number of temperature bins for all instruments
        assert len(set([v.shape[0] for v in temperature_response])) == 1, \
            "Number of temperature bins must be the same for all instruments."

        self.temperature_response = nn.ParameterList(temperature_response)
        self.instrument_scaling = nn.ParameterList(instrument_scaling)
        self.temperature_response_mapping = temperature_response_mapping

        absorption_config = copy.deepcopy(absorption_config) if self.absorption else {'type': None}
        absorption_type = absorption_config.pop('type', None)
        if absorption_type == 'constant':
            self.absorption_model = ConstantAbsorptionModel(**absorption_config)
        elif absorption_type == 'learned':
            self.absorption_model = AbsorptionModel(dim=16, n_layers=2)
        elif absorption_type is None:
            self.absorption_model = None
        else:
            raise NotImplementedError(f"Absorption type {absorption_type} not implemented.")

    def render_instruments(self, instrument_n_rays, state):
        ray_idx = 0
        render_out = {}
        for k in instrument_n_rays.keys():
            n = instrument_n_rays[k]  # number of rays for instrument k
            # split state for each instrument
            instrument_state = {k: v[ray_idx:ray_idx + n] for k, v in state.items()}
            # get temperature response and instrument scaling for each instrument
            tr_idx = self.temperature_response_mapping[k]
            temperature_response = self.temperature_response[tr_idx]
            instrument_scaling = self.instrument_scaling[tr_idx]
            absorption_model = self.absorption_model
            # render instrument output
            render_out[k] = self.raw2outputs(**instrument_state, temperature_response=temperature_response,
                                             instrument_scaling=instrument_scaling, absorption_model=absorption_model)
            ray_idx += n
        return render_out

    def raw2outputs(self, log_ne, total_ne, mean_log_T, total_log_ne,
                    z_vals: torch.Tensor, rays_d: torch.Tensor, query_points: torch.Tensor,
                    temperature_response: nn.Module,
                    instrument_scaling: nn.Parameter,
                    absorption_model: nn.Module, **kwargs):
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
