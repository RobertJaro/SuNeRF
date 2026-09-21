import torch


def _safe_ray_inputs(rays_o: torch.Tensor, rays_d: torch.Tensor):
    """Replace invalid ray components without losing their validity flag."""
    if rays_o.ndim != 2 or rays_d.shape != rays_o.shape or rays_o.shape[-1] != 3:
        raise ValueError(
            f'rays_o and rays_d must both have shape (n_rays, 3), got '
            f'{tuple(rays_o.shape)} and {tuple(rays_d.shape)}.'
        )
    finite = torch.isfinite(rays_o).all(dim=-1) & torch.isfinite(rays_d).all(dim=-1)
    safe_o = torch.where(finite[:, None], rays_o, torch.zeros_like(rays_o))
    fallback_d = torch.zeros_like(rays_d)
    fallback_d[:, 0] = 1
    safe_d = torch.where(finite[:, None], rays_d, fallback_d)
    norm_squared = safe_d.square().sum(dim=-1)
    valid = finite & (norm_squared > torch.finfo(rays_d.dtype).eps)
    safe_d = torch.where(valid[:, None], safe_d, fallback_d)
    return safe_o, safe_d, valid


def _front_shell_intersection(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        inner_radius: torch.Tensor,
        outer_radius: torch.Tensor,
):
    """Return the visible forward segment of a concentric spherical shell."""
    safe_o, safe_d, valid = _safe_ray_inputs(rays_o, rays_d)
    a = safe_d.square().sum(dim=-1)
    b = 2 * (safe_o * safe_d).sum(dim=-1)

    outer_c = safe_o.square().sum(dim=-1) - outer_radius.square()
    outer_discriminant = b.square() - 4 * a * outer_c
    valid &= outer_discriminant >= 0
    outer_sqrt = torch.sqrt(outer_discriminant.clamp_min(0))
    denominator = 2 * a.clamp_min(torch.finfo(a.dtype).eps)
    outer_near = (-b - outer_sqrt) / denominator
    outer_far = (-b + outer_sqrt) / denominator
    distance_near = outer_near.clamp_min(0)
    distance_far = outer_far
    valid &= distance_far > distance_near

    inner_c = safe_o.square().sum(dim=-1) - inner_radius.square()
    inner_discriminant = b.square() - 4 * a * inner_c
    intersects_inner = inner_discriminant >= 0
    inner_sqrt = torch.sqrt(inner_discriminant.clamp_min(0))
    inner_near = (-b - inner_sqrt) / denominator
    inner_far = (-b + inner_sqrt) / denominator

    # For a normal remote observer, the photosphere terminates the visible front
    # shell. If a synthetic ray starts inside the inner sphere, begin after its
    # forward exit instead.
    origin_inside_inner = safe_o.square().sum(dim=-1) < inner_radius.square()
    distance_near = torch.where(
        valid & intersects_inner & origin_inside_inner & (inner_far > distance_near),
        inner_far,
        distance_near,
    )
    hits_front_surface = (
        valid & intersects_inner & ~origin_inside_inner
        & (inner_near > distance_near) & (inner_near < distance_far)
    )
    distance_far = torch.where(hits_front_surface, inner_near, distance_far)
    valid &= distance_far > distance_near

    distance_near = torch.where(valid, distance_near, torch.zeros_like(distance_near))
    distance_far = torch.where(valid, distance_far, torch.zeros_like(distance_far))
    return safe_o, safe_d, distance_near, distance_far, valid


def _perturb_interior_samples(z_vals: torch.Tensor) -> torch.Tensor:
    """Jitter interior samples while preserving the exact ray boundaries."""
    if z_vals.shape[-1] <= 2:
        return z_vals

    mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    lower = mids[..., :-1]
    upper = mids[..., 1:]
    interior = lower + (upper - lower) * torch.rand_like(lower)
    return torch.cat([z_vals[..., :1], interior, z_vals[..., -1:]], dim=-1)


class SphericalSampler(torch.nn.Module):

    def __init__(self, Rs_per_ds, min_distance=1.0, max_distance=2.0, n_samples=64, perturb=True,
                 radial_weighting=False, radial_weight_power=2.0, radial_weight_grid_size=256):
        super().__init__()
        if n_samples < 2:
            raise ValueError('n_samples must be at least 2.')
        if min_distance < 0 or max_distance <= min_distance:
            raise ValueError('Require 0 <= min_distance < max_distance.')
        self.perturb = perturb
        self.radial_weighting = radial_weighting
        self.radial_weight_power = radial_weight_power
        self.radial_weight_grid_size = radial_weight_grid_size

        self.register_buffer('max_distance', torch.tensor(max_distance / Rs_per_ds, dtype=torch.float32))
        self.register_buffer('min_distance', torch.tensor(min_distance / Rs_per_ds, dtype=torch.float32))
        self.register_buffer('t_vals', torch.linspace(0.0, 1.0, n_samples, dtype=torch.float32)[None])

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        r"""
        Sample from near to solar surface. If no points are on the solar surface this
        """

        rays_o, rays_d, dist_near, dist_far, ray_valid = _front_shell_intersection(
            rays_o, rays_d, self.min_distance, self.max_distance
        )

        if self.radial_weighting:
            t_vals = self.t_vals.to(device=rays_o.device, dtype=rays_o.dtype).expand(rays_o.shape[0], -1)
            z_vals = self._sample_radial_weighted(rays_o, rays_d, dist_near, dist_far, t_vals)
        else:
            t_vals = self.t_vals.to(device=rays_o.device, dtype=rays_o.dtype)
            z_vals = dist_near[:, None] * (1. - t_vals) + dist_far[:, None] * t_vals

        # Draw stratified training samples while retaining the exact near/far
        # boundaries needed by finite-interval quadrature. Evaluation is
        # deterministic even when perturb=True in the configuration.
        if self.perturb and self.training:
            z_vals = _perturb_interior_samples(z_vals)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return {
            'points': pts,
            'z_vals': z_vals,
            'ray_valid': ray_valid,
            'rays_o': rays_o,
            'rays_d': rays_d,
        }

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
        if n_samples < 2:
            raise ValueError('n_samples must be at least 2.')
        if max_distance <= 1:
            raise ValueError('max_distance must be greater than one solar radius.')
        self.perturb = perturb

        self.register_buffer('distance', torch.tensor(max_distance / Rs_per_ds, dtype=torch.float32))
        self.register_buffer('solar_R', torch.tensor(1 / Rs_per_ds, dtype=torch.float32))

        self.register_buffer('t_vals', torch.linspace(0.0, 1.0, n_samples, dtype=torch.float32)[None])

    def forward(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        r"""
        Sample from near to solar surface. If no points are on the solar surface this
        """

        rays_o, rays_d, dist_near, dist_far, ray_valid = _front_shell_intersection(
            rays_o, rays_d, self.solar_R, self.distance
        )
        t_vals = self.t_vals.to(device=rays_o.device, dtype=rays_o.dtype)
        z_vals = dist_near[:, None] * (1. - t_vals) + dist_far[:, None] * t_vals

        # Keep the exact integration endpoints and disable jitter in eval mode.
        if self.perturb and self.training:
            z_vals = _perturb_interior_samples(z_vals)

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        return {
            'points': pts,
            'z_vals': z_vals,
            'ray_valid': ray_valid,
            'rays_o': rays_o,
            'rays_d': rays_d,
        }


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
        z_vals_combined, sort_indices = torch.sort(
            torch.cat([z_vals, new_z_samples], dim=-1), dim=-1
        )
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        # [N_rays, N_samples + n_samples, 3]
        return {
            'points': pts,
            'z_vals': z_vals_combined,
            'new_z_samples': new_z_samples,
            'sort_indices': sort_indices,
        }

    def sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        r"""
        Apply inverse transform sampling to a weighted set of points.
        """

        # Normalize weights to get PDF.
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True)  # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
        cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not (self.perturb and self.training):
            u = torch.linspace(0., 1., self.n_samples, device=cdf.device, dtype=cdf.dtype)
            u = u.expand(list(cdf.shape[:-1]) + [self.n_samples])  # [n_rays, n_samples]
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [self.n_samples],
                device=cdf.device,
                dtype=cdf.dtype,
            )  # [n_rays, n_samples]

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
