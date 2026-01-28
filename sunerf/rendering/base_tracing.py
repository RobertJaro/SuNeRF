import torch
from torch import nn

from sunerf.train.sampling import SphericalSampler, HierarchicalSampler, StratifiedSampler
from sunerf.train.util import TimeShuffler, NormalTimeShuffler


class MultiResolutionRenderingModule(nn.Module):

    def __init__(self, coarse_model, fine_model, rendering_modules, Rs_per_ds,
                 sampling_config=None, hierarchical_sampling_config=None, shuffle_config=None):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds

        self.rendering_modules = nn.ModuleDict(rendering_modules)

        # set default configurations
        hierarchical_sampling_config = {} if hierarchical_sampling_config is None else hierarchical_sampling_config
        sampling_config = {} if sampling_config is None else sampling_config

        # setup sampling strategy
        sampling_type = sampling_config.pop('type', 'stratified')
        if sampling_type == 'spherical':
            self.sampler = SphericalSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'stratified':
            self.sampler = StratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {sampling_type}')

        # setup hierarchical sampling
        hierarchical_sampling_type = hierarchical_sampling_config.pop('type', 'hierarchical')
        if hierarchical_sampling_type == 'hierarchical':
            self.sampler_hierarchical = HierarchicalSampler(**hierarchical_sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {hierarchical_sampling_type}')

        self.shuffler = load_shuffler(shuffle_config)

        self.coarse_model = coarse_model
        self.fine_model = fine_model

    def forward(self, batch, **kwargs):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""
        batch = self.shuffler(batch) if self.shuffler else batch

        dataset_keys = batch.keys()
        instrument_keys = self.rendering_modules.keys()

        dataset_n_rays = {k: batch[k]['rays'].shape[0] for k in dataset_keys}
        dataset_instrument = {k: batch[k]['instrument'] for k in dataset_keys}

        # merge rays from all instruments
        rays = torch.cat([batch[k]['rays'] for k in dataset_keys], dim=0)
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        times = torch.cat([batch[k]['time'] for k in dataset_keys], dim=0)

        # Sample query points along each ray.
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']

        # add time to query points
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

        # Coarse model pass.
        coarse_raw = self.coarse_model(query_points_time)
        state = {**coarse_raw, 'z_vals': z_vals,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points_time}
        coarse_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        # Fine model pass.
        # Apply hierarchical sampling for fine query points.
        weights = torch.cat([coarse_out[k]['weights'] for k in dataset_keys], dim=0)
        hierarchical_out = self.sampler_hierarchical(rays_o, rays_d, z_vals, weights)
        query_points, z_vals_combined, z_hierarch = (hierarchical_out['points'],
                                                     hierarchical_out['z_vals'],
                                                     hierarchical_out['new_z_samples'])

        # add time to query points = expand to dimensions of query points and slice one dimension
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)

        fine_raw = self.fine_model(query_points_time)
        state = {**fine_raw, 'z_vals': z_vals_combined,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points_time}
        fine_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        return {'fine_out': fine_out, 'coarse_out': coarse_out,
                'z_vals_stratified': z_vals, 'z_vals_hierarchical': z_hierarch}

    def render_instruments(self, dataset_n_rays, dataset_instrument, state):
        ray_idx = 0
        render_out = {}
        # number of rays for instrument k
        for k, n_rays in dataset_n_rays.items():
            # split state for each instrument
            instrument_state = {k: v[ray_idx:ray_idx + n_rays] for k, v in state.items()}
            # render instrument output
            instrument_key = dataset_instrument[k]
            render_out[k] = self.rendering_modules[instrument_key](**instrument_state)
            ray_idx += n_rays
        return render_out


class BasicRenderingModule(nn.Module):

    def __init__(self, model, rendering_modules, Rs_per_ds,
                 sampling_config=None, hierarchical_sampling_config=None, shuffle_config=None):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds

        self.rendering_modules = nn.ModuleDict(rendering_modules)

        # set default configurations
        sampling_config = {} if sampling_config is None else sampling_config
        hierarchical_sampling_config = {} if hierarchical_sampling_config is None else hierarchical_sampling_config

        # setup sampling strategy
        sampling_type = sampling_config.pop('type', 'spherical')
        if sampling_type == 'spherical':
            self.sampler = SphericalSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        elif sampling_type == 'stratified':
            self.sampler = StratifiedSampler(Rs_per_ds=Rs_per_ds, **sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {sampling_type}')

        # setup hierarchical sampling
        hierarchical_sampling_type = hierarchical_sampling_config.pop('type', 'hierarchical')
        if hierarchical_sampling_type == 'hierarchical':
            self.sampler_hierarchical = HierarchicalSampler(**hierarchical_sampling_config)
        else:
            raise ValueError(f'Unknown sampling type {hierarchical_sampling_type}')

        self.shuffler = load_shuffler(shuffle_config)

        print('Shuffle config:', self.shuffler)

        self.model = model

    def forward(self, batch, **kwargs):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""
        batch = self.shuffler(batch) if self.shuffler else batch

        dataset_keys = batch.keys()

        dataset_n_rays = {k: batch[k]['rays'].shape[0] for k in dataset_keys}
        dataset_instrument = {k: batch[k]['instrument'] for k in dataset_keys}

        # merge rays from all instruments
        rays = torch.cat([batch[k]['rays'] for k in dataset_keys], dim=0)
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        times = torch.cat([batch[k]['time'] for k in dataset_keys], dim=0)

        # Sample query points along each ray.
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']

        # add time to query points
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

        # Get weights for hierarchical sampling
        with torch.no_grad():
            coarse_raw = self.model(query_points_time)
            state = {**coarse_raw, 'z_vals': z_vals,
                     'rays_d': rays_d, 'rays_o': rays_o,
                     'query_points': query_points_time}
            model_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        # sample hierarchical points based on initial weights
        weights = torch.cat([model_out[k]['weights'] for k in dataset_keys], dim=0)
        hierarchical_out = self.sampler_hierarchical(rays_o, rays_d, z_vals, weights)
        query_points, z_vals_combined = (hierarchical_out['points'], hierarchical_out['z_vals'])

        # add time to query points = expand to dimensions of query points and slice one dimension
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)

        fine_raw = self.model(query_points_time)
        state = {**fine_raw, 'z_vals': z_vals_combined,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points_time}
        model_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        return {'model_out': model_out, 'z_vals': z_vals_combined, 'z_vals_stratified': z_vals}

    def render_instruments(self, dataset_n_rays, dataset_instrument, state):
        ray_idx = 0
        render_out = {}
        for ds_key, n_rays in dataset_n_rays.items():
            # split state for each instrument
            instrument_state = {k: v[ray_idx:ray_idx + n_rays] for k, v in state.items()}
            # render instrument output
            instrument_key = dataset_instrument[ds_key]
            render_out[ds_key] = self.rendering_modules[instrument_key](**instrument_state)
            ray_idx += n_rays
        return render_out

    def on_train_batch_end(self, *args, **kwargs):
        if self.shuffler is not None:
            self.shuffler.on_train_batch_end(*args, **kwargs)


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


def load_shuffler(shuffle_config):
    if shuffle_config:
        shuffle_type = shuffle_config.pop('type')
        if shuffle_type == 'time':
            return TimeShuffler(**shuffle_config)
        elif shuffle_type == 'normal_time':
            return NormalTimeShuffler(**shuffle_config)
        else:
            raise NotImplementedError(f"Shuffle type {shuffle_type} not implemented.")
    else:
        return None
