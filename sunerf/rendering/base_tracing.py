import torch
from torch import nn
from astropy import constants as const
from astropy import units as u

from sunerf.train.sampling import SphericalSampler, HierarchicalSampler, StratifiedSampler
from sunerf.train.util import TimeShuffler, NormalTimeShuffler


class MultiResolutionRenderingModule(nn.Module):

    def __init__(self, coarse_model, fine_model, rendering_modules, Rs_per_ds,
                 seconds_per_dt=None, sampling_config=None, hierarchical_sampling_config=None, shuffle_config=None,
                 light_travel_time=False):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.light_travel_time = bool(light_travel_time)
        light_dt_per_model_distance = (
            0.0 if seconds_per_dt is None
            else Rs_per_ds / const.c.to_value(u.R_sun / u.s) / seconds_per_dt
        )
        self.register_buffer(
            'light_dt_per_model_distance',
            torch.tensor(float(light_dt_per_model_distance), dtype=torch.float32),
        )
        if self.light_travel_time and self.seconds_per_dt is None:
            raise ValueError('seconds_per_dt is required when light_travel_time=True')

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

        query_points_time = self.add_sample_times(query_points, rays_o, times)

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

        query_points_time = self.add_sample_times(query_points, rays_o, times)

        fine_raw = self.fine_model(query_points_time)
        state = {**fine_raw, 'z_vals': z_vals_combined,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points_time}
        fine_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        return {'fine_out': fine_out, 'coarse_out': coarse_out,
                'z_vals_stratified': z_vals, 'z_vals_hierarchical': z_hierarch}

    def add_sample_times(self, query_points, rays_o, times):
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        if self.light_travel_time:
            # Detector timestamps see each scattering point delayed by its observer distance.
            observer_to_sample = torch.linalg.norm(query_points - rays_o[:, None, :], dim=-1, keepdim=True)
            light_time = observer_to_sample * self.light_dt_per_model_distance
            exp_times = exp_times - light_time
        return torch.cat([query_points, exp_times], -1)

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
                 seconds_per_dt=None, sampling_config=None, hierarchical_sampling_config=None, shuffle_config=None,
                 light_travel_time=False):
        super().__init__()
        self.Rs_per_ds = Rs_per_ds
        self.seconds_per_dt = seconds_per_dt
        self.light_travel_time = bool(light_travel_time)
        light_dt_per_model_distance = (
            0.0 if seconds_per_dt is None
            else Rs_per_ds / const.c.to_value(u.R_sun / u.s) / seconds_per_dt
        )
        self.register_buffer(
            'light_dt_per_model_distance',
            torch.tensor(float(light_dt_per_model_distance), dtype=torch.float32),
        )
        if self.light_travel_time and self.seconds_per_dt is None:
            raise ValueError('seconds_per_dt is required when light_travel_time=True')

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

        query_points_time = self.add_sample_times(query_points, rays_o, times)

        # Evaluate the coarse points once. Keep their model graph because these
        # values are reused in the final, fine-grid integral. Only the cheap
        # rendering pass used to construct the sampling PDF is graph-free.
        coarse_raw = self.model(query_points_time)
        with torch.no_grad():
            state = {**coarse_raw, 'z_vals': z_vals,
                     'rays_d': rays_d, 'rays_o': rays_o,
                     'query_points': query_points_time}
            coarse_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        # sample hierarchical points based on initial weights
        weights = torch.cat([coarse_out[k]['weights'] for k in dataset_keys], dim=0)
        hierarchical_out = self.sampler_hierarchical(rays_o, rays_d, z_vals, weights)
        del coarse_out, weights
        query_points = hierarchical_out['points']
        z_vals_combined = hierarchical_out['z_vals']

        # Evaluate only the newly drawn samples, then merge both model outputs
        # with the same permutation that sorted the nonuniform ray grid. The
        # renderer therefore sees every coarse and fine point exactly once.
        new_z_samples = hierarchical_out['new_z_samples']
        new_query_points = rays_o[..., None, :] + rays_d[..., None, :] * new_z_samples[..., :, None]
        new_query_points_time = self.add_sample_times(new_query_points, rays_o, times)
        new_raw = self.model(new_query_points_time)
        fine_raw = _merge_sample_outputs(
            coarse_raw, new_raw, hierarchical_out['sort_indices']
        )

        state = {**fine_raw, 'z_vals': z_vals_combined,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': self.add_sample_times(query_points, rays_o, times)}
        model_out = self.render_instruments(dataset_n_rays, dataset_instrument, state)

        return {'model_out': model_out, 'z_vals': z_vals_combined, 'z_vals_stratified': z_vals}

    def add_sample_times(self, query_points, rays_o, times):
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        if self.light_travel_time:
            # Detector timestamps see each scattering point delayed by its observer distance.
            observer_to_sample = torch.linalg.norm(query_points - rays_o[:, None, :], dim=-1, keepdim=True)
            light_time = observer_to_sample * self.light_dt_per_model_distance
            exp_times = exp_times - light_time
        return torch.cat([query_points, exp_times], -1)

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


def _merge_sample_outputs(coarse_raw, new_raw, sort_indices):
    """Merge model outputs along their ray-sample dimension.

    ``sort_indices`` is the permutation returned when the coarse and newly
    sampled ray distances are concatenated and sorted. Model outputs may have
    any number of trailing feature dimensions.
    """
    if coarse_raw.keys() != new_raw.keys():
        raise ValueError('Coarse and hierarchical model outputs must have identical keys')

    merged = {}
    for key in coarse_raw:
        coarse_value = coarse_raw[key]
        new_value = new_raw[key]
        if coarse_value.ndim < 2 or new_value.ndim != coarse_value.ndim:
            raise ValueError(f"Model output '{key}' must include ray and sample dimensions")
        if coarse_value.shape[0] != new_value.shape[0] or coarse_value.shape[2:] != new_value.shape[2:]:
            raise ValueError(f"Incompatible coarse and hierarchical shapes for model output '{key}'")

        values = torch.cat([coarse_value, new_value], dim=1)
        gather_indices = sort_indices
        for _ in range(values.ndim - 2):
            gather_indices = gather_indices.unsqueeze(-1)
        gather_indices = gather_indices.expand(*sort_indices.shape, *values.shape[2:])
        merged[key] = torch.gather(values, dim=1, index=gather_indices)
    return merged


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
