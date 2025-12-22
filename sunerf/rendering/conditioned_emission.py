import torch
from torch import nn

from sunerf.model.model import ConditionedNeRF
from sunerf.rendering.base_tracing import cumprod_exclusive
from sunerf.train.sampling import SphericalSampler, StratifiedSampler


class ConditionedRadiativeTransfer(nn.Module):

    def __init__(self, Rs_per_ds, z_dim, sampling_config=None, model_config=None, use_absorption=True, **kwargs):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds
        self.use_absorption = use_absorption

        # set default configurations
        sampling_config = {} if sampling_config is None else sampling_config

        # setup sampling strategy
        sampling_type = sampling_config.pop('type', 'spherical')
        if sampling_type == 'spherical':
            self.sampler = SphericalSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'stratified':
            self.sampler = StratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {sampling_type}')

        model_config = {} if model_config is None else model_config
        # setup models
        self.model = ConditionedNeRF(1, z_dim=z_dim, **model_config)

    def forward(self, rays_o, rays_d, z_latent, **kwargs):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""
        # Sample query points along each ray.
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']
        # query_points: [n_rays, n_samples, 3]; z_vals: [n_rays, n_samples]
        z_latent = z_latent[:, None, :].repeat(1, query_points.shape[1], 1)  # [n_rays, n_samples, z_dim]

        raw = self.model(query_points, z_latent)
        state = {**raw, 'z_vals': z_vals,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points}
        model_out = self.raw2outputs(**state)

        return {'model_out': model_out, 'z_vals': z_vals, 'z_vals_stratified': z_vals}

    def raw2outputs(self, emission, alpha, z_vals: torch.Tensor, rays_d: torch.Tensor, query_points, **kwargs):
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

        # emission ([..., 0]; epsilon(z)) and absorption ([..., 1]; kappa(z)) coefficient per unit volume
        # dtau = - kappa dz
        # I' / I = - kappa dz --> I' emerging intensity; I incident intensity;
        emission = emission[..., 0]  # torch.exp(raw[..., 0])
        intensity = emission * dists  # emission per sampled point [n_rays, n_samples]

        # transmission per sampled point [n_rays, n_samples]
        alpha = alpha[..., 0]  # nn.functional.relu(raw[..., 1])
        absorption = torch.exp(-alpha * dists)
        # [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

        # compute total absorption for each light ray (intensity)
        # how much light is transmitted from each sampled point
        # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)
        total_absorption = cumprod_exclusive(absorption + 1e-10)
        # [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
        # apply absorption to intensities
        if self.use_absorption:
            emerging_intensity = intensity * total_absorption  # integrate total intensity [n_rays, n_samples - 1]
        else:
            emerging_intensity = intensity  # integrate total intensity [n_rays, n_samples - 1]
        # sum all intensity contributions
        pixel_intensity = emerging_intensity.sum(1)[:, None]

        # set the weigths to the intensity contributions (sample primary contributing regions)
        weights = emerging_intensity
        weights = weights / (weights.sum(1)[:, None] + 1e-10)

        mean_absorption = (1 - absorption).mean(1)

        distance = query_points[..., :3].pow(2).sum(-1).pow(0.5)
        height_map = (weights * distance).sum(-1)

        return {'image': pixel_intensity, 'weights': weights, 'absorption_map': mean_absorption, 'height_map': height_map}
