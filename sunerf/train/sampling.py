from typing import Optional, Tuple

import torch

# Stratified with opaque Sun
def sample_non_uniform_box(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        grid_exponent: Optional[int] = 7,
        perturb: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
      Sample along ray with higher density bins closer to the solar surface.
      This is the traditional NeRF implementation,
      but modified to have a completely opaque solar sphere
    """

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)

    # solve quadratic equation --> find points between 1 and 1.1 solar radii
    a = rays_d.pow(2).sum(-1)
    b = (2 * rays_o * rays_d).sum(-1)
    c = rays_o.pow(2).sum(-1) - 1 ** 2

    # make the farthest point the surface of the Sun for points touching it
    new_far = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
    new_far[torch.isnan(new_far)] = far
    new_near = torch.ones(new_far.shape, device=rays_o.device) * near

    # Sample linearly between `near` and `far`
    z_vals = new_near[:, None] * (1. - t_vals[None, :]) + new_far[:, None] * (t_vals[None, :])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # Calculate distance from the sun
    radius = torch.sqrt(pts.pow(2).sum(-1))

    # Redefine the grid to become denser close to the sun
    new_t_vals = radius.pow(grid_exponent)
    new_t_vals = torch.cumsum(new_t_vals, dim=-1)
    new_t_vals = new_t_vals - torch.amin(new_t_vals, dim=-1)[:, None]
    new_t_vals = new_t_vals / torch.amax(new_t_vals, dim=-1)[:, None]

    # Redefine z_vals and sample points
    # Sample linearly between `near` and `far`
    z_vals = new_near[:, None] * (1. - new_t_vals) + new_far[:, None] * (new_t_vals)

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.concat([mids, z_vals[:, -1:]], dim=1)
        lower = torch.concat([z_vals[:, :1], mids], dim=1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand[None, :]

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals


# Stratified Sampling
def sample_to_solar_surface(rays_o: torch.Tensor, rays_d: torch.Tensor, near: float, far: float, n_samples: int,
                            perturb: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Sample from near to solar surface. If no points are on the solar surface this
    """

    # convert near and far from center to actual distance
    distance = rays_o.pow(2).sum(-1).pow(0.5)
    projected_near, projected_far = distance + near, distance + far

    # Grab samples for space integration along ray
    t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)

    # solve quadratic equation --> find points at 1 solar radii
    a = rays_d.pow(2).sum(-1)
    b = (2 * rays_o * rays_d).sum(-1)
    c = rays_o.pow(2).sum(-1) - 1 ** 2
    dist_far = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

    dist_far[torch.isnan(dist_far)] = projected_far[torch.isnan(dist_far)]

    z_vals = projected_near[:, None] * (1. - t_vals[None]) + dist_far[:, None] * (t_vals[None])

    # Draw uniform samples from bins along ray
    if perturb:
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.concat([mids, z_vals[:, -1:]], dim=1)
        lower = torch.concat([z_vals[:, :1], mids], dim=1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand[None, :]

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals


# Hierarchical Sampling
def sample_pdf(
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    """

    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device)  # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous()  # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                         index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                          index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples  # [n_rays, n_samples]


def sample_hierarchical(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                               perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples
