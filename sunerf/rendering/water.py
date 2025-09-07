import numpy as np
import torch
from torch import nn

from sunerf.rendering.base_tracing import cumprod_exclusive


class WaterRadiativeTransfer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, log10_rho, z_vals: torch.Tensor, rays_d: torch.Tensor, query_points: torch.Tensor, **kwargs):
        """
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

        rho = 10 ** log10_rho
        intensity = (rho.pow(1/4) * dists)

        alpha = rho * 0
        absorption = torch.exp(-alpha * dists)
        integrated_absorption = cumprod_exclusive(absorption + 1e-10, dim=1)

        emerging_intensity = intensity * integrated_absorption
        integrated_intensity = emerging_intensity.sum(1)

        # set the weigths to the intensity contributions (sample primary contributing regions)
        weights = torch.ones_like(rho[..., 0]) #rho[..., 0]
        weights = weights / (weights.sum(1, keepdim=True) + 1e-10)

        # visualization outputs
        distance = query_points[..., :3].pow(2).sum(-1).pow(0.5)
        height_map = (weights * distance).sum(-1)

        return {'image': integrated_intensity, 'weights': weights, 'height_map': height_map,
                'distance': distance, 'z_vals': z_vals, 'query_points': query_points}
