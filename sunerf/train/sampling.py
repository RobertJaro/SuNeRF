import torch
from torch import nn


class SphericalSampler(torch.nn.Module):

    def __init__(self, Rs_per_ds, min_distance=1.0, max_distance=2.0, n_samples=64, perturb=True,
                 radial_weighting=False, radial_weight_power=2.0, radial_weight_grid_size=256):
        super().__init__()
        self.perturb = perturb
        self.radial_weighting = radial_weighting
        self.radial_weight_power = radial_weight_power
        self.radial_weight_grid_size = radial_weight_grid_size

        self.max_distance = nn.Parameter(torch.tensor(max_distance / Rs_per_ds, dtype=torch.float32), requires_grad=False)
        self.min_distance = nn.Parameter(torch.tensor(min_distance / Rs_per_ds, dtype=torch.float32), requires_grad=False)

        t_vals = torch.linspace(0., 1., n_samples)[None]
        self.t_vals = nn.Parameter(torch.tensor(t_vals, dtype=torch.float32), requires_grad=False)

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        r"""
        Sample from near to solar surface. If no points are on the solar surface this
        """

        # solve quadratic equation --> find points at distance
        a = rays_d.pow(2).sum(-1)
        b = (2 * rays_o * rays_d).sum(-1)
        c = rays_o.pow(2).sum(-1) - self.max_distance ** 2
        dist_near = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a + 1e-8)
        dist_far = (-b + torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a + 1e-8)

        # solve quadratic equation --> find points at 1 solar radii
        # stop sampling at solar surface
        c = rays_o.pow(2).sum(-1) - self.min_distance ** 2
        dist_inner = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

        intersect_solar_surface = ~torch.isnan(dist_inner)
        dist_far[intersect_solar_surface] = dist_inner[intersect_solar_surface]

        # dist_far[torch.isnan(dist_far)] = projected_far[torch.isnan(dist_far)]
        # dist_far = projected_far

        if self.radial_weighting:
            t_vals = self.t_vals.to(device=rays_o.device, dtype=rays_o.dtype).expand(rays_o.shape[0], -1)
            z_vals = self._sample_radial_weighted(rays_o, rays_d, dist_near, dist_far, t_vals)
        else:
            z_vals = dist_near[:, None] * (1. - self.t_vals) + dist_far[:, None] * self.t_vals

            # Draw uniform samples from bins along ray
            if self.perturb:
                mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
                upper = torch.concat([mids, z_vals[:, -1:]], dim=1)
                lower = torch.concat([z_vals[:, :1], mids], dim=1)
                t_rand = torch.rand(z_vals.shape, device=z_vals.device)
                z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return {'points': pts, 'z_vals': z_vals}

    def _sample_radial_weighted(self, rays_o, rays_d, dist_near, dist_far, t_vals):
        grid_t = torch.linspace(
            0.0, 1.0, self.radial_weight_grid_size, device=rays_o.device, dtype=rays_o.dtype
        )[None]
        z_grid = dist_near[:, None] * (1.0 - grid_t) + dist_far[:, None] * grid_t

        pts_grid = rays_o[:, None, :] + rays_d[:, None, :] * z_grid[..., None]
        radius = torch.linalg.norm(pts_grid, dim=-1).clamp_min(1e-6)
        weights = radius.pow(-self.radial_weight_power)

        dz = z_grid[:, 1:] - z_grid[:, :-1]
        pdf = 0.5 * (weights[:, 1:] + weights[:, :-1]) * dz
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.concat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)
        cdf = cdf / cdf[:, -1:].clamp_min(1e-8)

        inds = torch.searchsorted(cdf.contiguous(), t_vals.contiguous(), right=True)
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)

        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
        z_g = torch.gather(z_grid.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0]).clamp_min(1e-8)
        local_t = (t_vals - cdf_g[..., 0]) / denom
        return z_g[..., 0] + local_t * (z_g[..., 1] - z_g[..., 0])


class StratifiedSampler(torch.nn.Module):

    def __init__(self, Rs_per_ds, max_distance=1.3, n_samples=64, perturb=True):
        super().__init__()
        self.perturb = perturb

        self.register_buffer('distance', torch.tensor(max_distance / Rs_per_ds, dtype=torch.float32))
        self.register_buffer('solar_R', torch.tensor(1 / Rs_per_ds, dtype=torch.float32))

        t_vals = torch.linspace(0., 1., n_samples)[None]
        self.register_buffer('t_vals', torch.tensor(t_vals, dtype=torch.float32))

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        r"""
        Sample from near to solar surface. If no points are on the solar surface this
        """

        # convert near and far from center to actual distance
        distance = rays_o.pow(2).sum(-1).pow(0.5)

        # solve quadratic equation --> find points at 1 solar radii
        a = rays_d.pow(2).sum(-1)
        b = (2 * rays_o * rays_d).sum(-1)
        # stop sampling at solar surface
        c = rays_o.pow(2).sum(-1) - self.solar_R ** 2
        dist_inner = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

        dist_near = distance - self.distance
        dist_far = distance + self.distance

        # replace endpoint with solar surface
        intersect_solar_surface = ~torch.isnan(dist_inner)
        dist_far[intersect_solar_surface] = dist_inner[intersect_solar_surface]

        z_vals = dist_near[:, None] * (1. - self.t_vals) + dist_far[:, None] * (self.t_vals)

        # Draw uniform samples from bins along ray
        if self.perturb:
            mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
            upper = torch.concat([mids, z_vals[:, -1:]], dim=1)
            lower = torch.concat([z_vals[:, :1], mids], dim=1)
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return {'points': pts, 'z_vals': z_vals}


class HierarchicalSampler(torch.nn.Module):

    def __init__(self, n_samples=128, perturb=False):
        super().__init__()
        self.n_samples = n_samples
        self.perturb = perturb

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor,
                z_vals: torch.Tensor, weights: torch.Tensor, ):
        r"""
        Apply hierarchical sampling to the rays.
        """

        # Draw samples from PDF using z_vals as bins and weights as probabilities.
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self.sample_pdf(z_vals_mid, weights[..., 1:-1])
        new_z_samples = new_z_samples.detach()

        # Resample points from ray based on PDF.
        z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        # [N_rays, N_samples + n_samples, 3]
        return {'points': pts, 'z_vals': z_vals_combined, 'new_z_samples': new_z_samples}

    def sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        r"""
        Apply inverse transform sampling to a weighted set of points.
        """

        # Normalize weights to get PDF.
        pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
        cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not self.perturb:
            u = torch.linspace(0., 1., self.n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [self.n_samples])  # [n_rays, n_samples]
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [self.n_samples], device=cdf.device)  # [n_rays, n_samples]

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

        return samples
