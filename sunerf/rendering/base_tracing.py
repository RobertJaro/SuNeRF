import torch
from torch import nn

from sunerf.train.sampling import SphericalSampler, HierarchicalSampler, StratifiedSampler


class SuNeRFRendering(nn.Module):

    def __init__(self, coarse_model, fine_model, Rs_per_ds, sampling_config=None, hierarchical_sampling_config=None):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds

        # set default configurations
        hierarchical_sampling_config = {'type': 'hierarchical'} \
            if hierarchical_sampling_config is None else hierarchical_sampling_config
        sampling_config = {'type': 'stratified'} if sampling_config is None else sampling_config

        # setup sampling strategy
        sampling_type = sampling_config.pop('type')
        if sampling_type == 'spherical':
            self.sampler = SphericalSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'stratified':
            self.sampler = StratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {sampling_type}')

        # setup hierarchical sampling
        hierarchical_sampling_type = hierarchical_sampling_config.pop('type')
        if hierarchical_sampling_type == 'hierarchical':
            self.sampler_hierarchical = HierarchicalSampler(**hierarchical_sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {hierarchical_sampling_type}')

        self.coarse_model = coarse_model
        self.fine_model = fine_model

    def forward(self, rays_o, rays_d, times, **kwargs):
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

        # add time to query points
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

        # Coarse model pass.
        coarse_out = self._render(self.coarse_model, query_points_time, rays_d, rays_o, z_vals, **kwargs)

        # Fine model pass.
        # Apply hierarchical sampling for fine query points.
        hierarchical_out = self.sampler_hierarchical(rays_o, rays_d, z_vals, coarse_out['weights'])
        query_points, z_vals_combined, z_hierarch = (hierarchical_out['points'],
                                                     hierarchical_out['z_vals'],
                                                     hierarchical_out['new_z_samples'])

        # add time to query points = expand to dimensions of query points and slice one dimension
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)

        fine_out = self._render(self.fine_model, query_points_time, rays_d, rays_o, z_vals_combined, **kwargs)

        # compute regularization of absorption
        distance = query_points.pow(2).sum(-1).pow(0.5)
        height_map = (fine_out['weights'] * distance).sum(-1)
        # penalize absorption past 1.2 solar radii
        # regularization = torch.relu(distance - 1.2 / self.Rs_per_ds) * (1 - absorption)

        return {**fine_out,
                'z_vals_stratified': z_vals, 'z_vals_hierarchical': z_hierarch,
                'height_map': height_map,
                'coarse_image': coarse_out['image'], 'fine_image': fine_out['image']}

    def forward_points(self, query_points):
        flat_points = query_points.view(-1, 4)
        raw_out = self.fine_model(flat_points)
        return raw_out

    def _render(self, model, query_points, rays_d, rays_o, z_vals, **kwargs):
        raw = model(query_points)
        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        state = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d, 'rays_o': rays_o, 'query_points': query_points}
        out = self.raw2outputs(**state, **kwargs)
        return out

    def raw2outputs(self, **kwargs):
        raise NotImplementedError("This method should be implemented in a subclass")


def cumprod_exclusive(tensor: torch.Tensor, dim=1) -> torch.Tensor:
    """
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    if dim == 0:
        cumprod[0] = 1.
    elif dim == 1:
        cumprod[:, 0] = 1.
    elif dim == 2:
        cumprod[:, :, 0] = 1.
    elif dim == -1:
        cumprod[..., 0] = 1.
    else:
        raise NotImplementedError(f"cumprod_exclusive not implemented for dim={dim}")

    return cumprod
