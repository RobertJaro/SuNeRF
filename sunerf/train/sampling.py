import torch


class SphericalSampler(torch.nn.Module):

    def __init__(self, Rs_per_ds, distance=2.0, n_samples=64, perturb=True):
        super().__init__()
        self.perturb = perturb

        self.register_buffer('distance', torch.tensor(distance / Rs_per_ds, dtype=torch.float32))
        self.register_buffer('solar_R', torch.tensor(1 / Rs_per_ds, dtype=torch.float32))

        t_vals = torch.linspace(0., 1., n_samples)[None]
        self.register_buffer('t_vals', torch.tensor(t_vals, dtype=torch.float32))

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        r"""
        Sample from near to solar surface. If no points are on the solar surface this
        """

        # convert near and far from center to actual distance
        distance = rays_o.pow(2).sum(-1).pow(0.5)

        # solve quadratic equation --> find points at distance
        a = rays_d.pow(2).sum(-1)
        b = (2 * rays_o * rays_d).sum(-1)
        c = rays_o.pow(2).sum(-1) - self.distance ** 2
        dist_near = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)
        dist_far = (-b + torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

        # solve quadratic equation --> find points at 1 solar radii
        # stop sampling at solar surface
        c = rays_o.pow(2).sum(-1) - self.solar_R ** 2
        dist_inner = (-b - torch.sqrt(b.pow(2) - 4 * a * c)) / (2 * a)

        intersect_solar_surface = ~torch.isnan(dist_inner)
        dist_far[intersect_solar_surface] = dist_inner[intersect_solar_surface]

        # dist_far[torch.isnan(dist_far)] = projected_far[torch.isnan(dist_far)]
        # dist_far = projected_far

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

class StratifiedSampler(torch.nn.Module):

    def __init__(self, Rs_per_ds, distance=1.3, n_samples=64, perturb=True):
        super().__init__()
        self.perturb = perturb

        self.register_buffer('distance', torch.tensor(distance / Rs_per_ds, dtype=torch.float32))
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
