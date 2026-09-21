import math
import warnings

import torch
from torch import nn
from astropy import constants as const
from astropy import units as u

from sunerf.train.sampling import SphericalSampler, HierarchicalSampler, StratifiedSampler
from sunerf.train.util import TimeShuffler, NormalTimeShuffler


def _maybe_shuffle_times(batch, shuffler, *, module_training, shuffle):
    """Apply time augmentation without modifying tensors owned by the caller."""
    should_shuffle = module_training if shuffle is None else bool(shuffle)
    if shuffler is None or not should_shuffle:
        return batch

    copied_batch = {}
    for dataset_key, dataset_batch in batch.items():
        copied_dataset = dict(dataset_batch)
        if 'time' in copied_dataset:
            copied_dataset['time'] = copied_dataset['time'].clone()
        copied_batch[dataset_key] = copied_dataset

    # The legacy implementation fails for probability=1 and creates CPU index
    # tensors for GPU batches. Keep its schedule, but perform the permutation at
    # this boundary where the cloned batch and target device are known.
    if isinstance(shuffler, TimeShuffler):
        probability = min(max(float(shuffler.prob.detach()), 0.0), 1.0)
        for dataset_key, dataset_batch in copied_batch.items():
            if shuffler.data_sets is not None and dataset_key not in shuffler.data_sets:
                continue
            times = dataset_batch['time']
            n_shuffle = min(int(times.shape[0] * probability), times.shape[0])
            if n_shuffle <= 1:
                continue
            max_start = times.shape[0] - n_shuffle
            if max_start:
                start = int(torch.randint(
                    0, max_start + 1, (), device=times.device
                ).item())
            else:
                start = 0
            permutation = torch.randperm(n_shuffle, device=times.device)
            source = times[start:start + n_shuffle].clone()
            times[start:start + n_shuffle] = source[permutation]
        return copied_batch
    return shuffler(copied_batch)


def _batch_ray_validity(batch, dataset_keys):
    validity = []
    for dataset_key in dataset_keys:
        dataset_batch = batch[dataset_key]
        rays = dataset_batch['rays']
        times = dataset_batch['time']
        valid = torch.isfinite(rays).all(dim=(-2, -1))
        valid &= torch.isfinite(times).reshape(times.shape[0], -1).all(dim=-1)
        if 'ray_valid' in dataset_batch:
            explicit = dataset_batch['ray_valid'].to(device=rays.device, dtype=torch.bool)
            valid &= explicit.reshape(explicit.shape[0], -1).all(dim=-1)
        validity.append(valid)
    return torch.cat(validity, dim=0)


def _mask_rendered_output(output, ray_valid):
    """Zero renderer products for invalid rays while retaining a validity flag."""
    masked = {}
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == len(ray_valid):
            mask = ray_valid.reshape(len(ray_valid), *([1] * (value.ndim - 1)))
            masked[key] = torch.where(mask, value, torch.zeros_like(value))
        else:
            masked[key] = value
    masked['ray_valid'] = ray_valid
    return masked


def _device_safe_normal_shuffler(**config):
    shuffler = NormalTimeShuffler(**config)
    # NormalTimeShuffler predates module device management and stores gamma as a
    # plain CPU tensor. A non-persistent buffer follows the module to GPU without
    # changing the state-dict schema expected by legacy checkpoints.
    gamma = shuffler.gamma
    del shuffler.gamma
    shuffler.register_buffer('gamma', gamma, persistent=False)
    return shuffler


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
        hierarchical_sampling_config = dict(hierarchical_sampling_config or {})
        sampling_config = dict(sampling_config or {})

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

        self.shuffler = load_shuffler(
            shuffle_config, seconds_per_dt=self.seconds_per_dt
        )

        self.coarse_model = coarse_model
        self.fine_model = fine_model

    def forward(self, batch, shuffle=None, diagnostics=False, **kwargs):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""
        batch = _maybe_shuffle_times(
            batch, self.shuffler, module_training=self.training, shuffle=shuffle
        )

        dataset_keys = tuple(batch)

        dataset_n_rays = {k: batch[k]['rays'].shape[0] for k in dataset_keys}
        dataset_instrument = {k: batch[k]['instrument'] for k in dataset_keys}

        # merge rays from all instruments
        rays = torch.cat([batch[k]['rays'] for k in dataset_keys], dim=0)
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        times = torch.cat([batch[k]['time'] for k in dataset_keys], dim=0)
        input_ray_valid = _batch_ray_validity(batch, dataset_keys)
        times = torch.where(torch.isfinite(times), times, torch.zeros_like(times))

        # Sample query points along each ray.
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']
        rays_o, rays_d = sampling_out['rays_o'], sampling_out['rays_d']
        ray_valid = input_ray_valid & sampling_out['ray_valid']

        query_points_time = self.add_sample_times(query_points, rays_o, times)

        # Coarse model pass.
        coarse_raw = self.coarse_model(query_points_time)
        state = {**coarse_raw, 'z_vals': z_vals,
                 'rays_d': rays_d, 'rays_o': rays_o,
                 'query_points': query_points_time, 'ray_valid': ray_valid}
        coarse_out = self.render_instruments(
            dataset_n_rays,
            dataset_instrument,
            state,
            renderer_kwargs={'diagnostics': False},
        )

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
                 'query_points': query_points_time, 'ray_valid': ray_valid}
        fine_out = self.render_instruments(
            dataset_n_rays,
            dataset_instrument,
            state,
            renderer_kwargs={'diagnostics': diagnostics},
        )

        return {'fine_out': fine_out, 'coarse_out': coarse_out,
                'z_vals_stratified': z_vals, 'z_vals_hierarchical': z_hierarch,
                'ray_valid': ray_valid}

    def add_sample_times(self, query_points, rays_o, times):
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        if self.light_travel_time:
            # Detector timestamps see each scattering point delayed by its observer distance.
            observer_to_sample = torch.linalg.norm(query_points - rays_o[:, None, :], dim=-1, keepdim=True)
            light_time = observer_to_sample * self.light_dt_per_model_distance
            exp_times = exp_times - light_time
        return torch.cat([query_points, exp_times], -1)

    def render_instruments(
        self, dataset_n_rays, dataset_instrument, state, renderer_kwargs=None
    ):
        renderer_kwargs = {} if renderer_kwargs is None else dict(renderer_kwargs)
        ray_idx = 0
        render_out = {}
        # number of rays for instrument k
        for k, n_rays in dataset_n_rays.items():
            # split state for each instrument
            instrument_state = {k: v[ray_idx:ray_idx + n_rays] for k, v in state.items()}
            # render instrument output
            instrument_key = dataset_instrument[k]
            instrument_out = self.rendering_modules[instrument_key](
                **instrument_state, **renderer_kwargs
            )
            render_out[k] = _mask_rendered_output(
                instrument_out, instrument_state['ray_valid']
            )
            ray_idx += n_rays
        return render_out

    def on_train_batch_end(self, *args, **kwargs):
        if self.shuffler is not None:
            self.shuffler.on_train_batch_end(*args, **kwargs)


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
        sampling_config = dict(sampling_config or {})
        hierarchical_sampling_config = dict(hierarchical_sampling_config or {})

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

        self.shuffler = load_shuffler(
            shuffle_config, seconds_per_dt=self.seconds_per_dt
        )

        print('Shuffle config:', self.shuffler)

        self.model = model

    def forward(self, batch, shuffle=None, diagnostics=False, **kwargs):
        r"""_summary_
        		Compute forward pass through model.

        		Args:
        			rays_o (tensor): Origin of rays
        			rays_d (tensor): Direction of rays
        			times (tensor): Times of maps
        		Returns:
        			outputs: Synthesized filtergrams/images.
        		"""
        batch = _maybe_shuffle_times(
            batch, self.shuffler, module_training=self.training, shuffle=shuffle
        )

        dataset_keys = tuple(batch)

        dataset_n_rays = {k: batch[k]['rays'].shape[0] for k in dataset_keys}
        dataset_instrument = {k: batch[k]['instrument'] for k in dataset_keys}

        # merge rays from all instruments
        rays = torch.cat([batch[k]['rays'] for k in dataset_keys], dim=0)
        rays_o, rays_d = rays[:, 0], rays[:, 1]
        times = torch.cat([batch[k]['time'] for k in dataset_keys], dim=0)
        input_ray_valid = _batch_ray_validity(batch, dataset_keys)
        times = torch.where(torch.isfinite(times), times, torch.zeros_like(times))

        # Sample query points along each ray.
        sampling_out = self.sampler(rays_o, rays_d)
        query_points, z_vals = sampling_out['points'], sampling_out['z_vals']
        rays_o, rays_d = sampling_out['rays_o'], sampling_out['rays_d']
        ray_valid = input_ray_valid & sampling_out['ray_valid']

        query_points_time = self.add_sample_times(query_points, rays_o, times)

        # Evaluate the coarse points once. Keep their model graph because these
        # values are reused in the final, fine-grid integral. Only the cheap
        # rendering pass used to construct the sampling PDF is graph-free.
        coarse_raw = self.model(query_points_time)
        with torch.no_grad():
            state = {**coarse_raw, 'z_vals': z_vals,
                     'rays_d': rays_d, 'rays_o': rays_o,
                     'query_points': query_points_time, 'ray_valid': ray_valid}
            coarse_out = self.render_instruments(
                dataset_n_rays,
                dataset_instrument,
                state,
                renderer_kwargs={'diagnostics': False},
            )

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
                 'query_points': self.add_sample_times(query_points, rays_o, times),
                 'ray_valid': ray_valid}
        model_out = self.render_instruments(
            dataset_n_rays,
            dataset_instrument,
            state,
            renderer_kwargs={'diagnostics': diagnostics},
        )

        return {
            'model_out': model_out,
            'z_vals': z_vals_combined,
            'z_vals_stratified': z_vals,
            'ray_valid': ray_valid,
        }

    def add_sample_times(self, query_points, rays_o, times):
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        if self.light_travel_time:
            # Detector timestamps see each scattering point delayed by its observer distance.
            observer_to_sample = torch.linalg.norm(query_points - rays_o[:, None, :], dim=-1, keepdim=True)
            light_time = observer_to_sample * self.light_dt_per_model_distance
            exp_times = exp_times - light_time
        return torch.cat([query_points, exp_times], -1)

    def render_instruments(
        self, dataset_n_rays, dataset_instrument, state, renderer_kwargs=None
    ):
        renderer_kwargs = {} if renderer_kwargs is None else dict(renderer_kwargs)
        ray_idx = 0
        render_out = {}
        for ds_key, n_rays in dataset_n_rays.items():
            # split state for each instrument
            instrument_state = {k: v[ray_idx:ray_idx + n_rays] for k, v in state.items()}
            # render instrument output
            instrument_key = dataset_instrument[ds_key]
            instrument_out = self.rendering_modules[instrument_key](
                **instrument_state, **renderer_kwargs
            )
            render_out[ds_key] = _mask_rendered_output(
                instrument_out, instrument_state['ray_valid']
            )
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


def load_shuffler(shuffle_config, *, seconds_per_dt=None):
    if shuffle_config:
        shuffle_config = dict(shuffle_config)
        shuffle_type = shuffle_config.pop('type')
        if shuffle_type == 'time':
            return TimeShuffler(**shuffle_config)
        elif shuffle_type == 'normal_time':
            seconds_keys = {'start_seconds', 'end_seconds'} & shuffle_config.keys()
            normalized_keys = {'start', 'end'} & shuffle_config.keys()
            if seconds_keys and normalized_keys:
                raise ValueError(
                    'normal_time shuffle cannot mix start_seconds/end_seconds '
                    'with deprecated normalized start/end.'
                )
            if seconds_keys:
                if 'start_seconds' not in shuffle_config:
                    raise ValueError('normal_time start_seconds is required.')
                if seconds_per_dt is None or not float(seconds_per_dt) > 0:
                    raise ValueError(
                        'A positive seconds_per_dt is required for physical time augmentation.'
                    )
                start_seconds = float(shuffle_config.pop('start_seconds'))
                end_seconds = float(shuffle_config.pop('end_seconds', 1e-2))
                if not (
                    math.isfinite(start_seconds)
                    and math.isfinite(end_seconds)
                    and start_seconds > 0
                    and 0 <= end_seconds <= start_seconds
                ):
                    raise ValueError(
                        'Require finite normal_time widths with '
                        '0 <= end_seconds <= start_seconds and start_seconds > 0.'
                    )
                if float(shuffle_config.get('iterations', 1e5)) <= 0:
                    raise ValueError('normal_time iterations must be positive.')
                shuffle_config['start'] = start_seconds / float(seconds_per_dt)
                shuffle_config['end'] = end_seconds / float(seconds_per_dt)
                shuffler = _device_safe_normal_shuffler(**shuffle_config)
                shuffler.start_seconds = start_seconds
                shuffler.end_seconds = end_seconds
                return shuffler

            warnings.warn(
                'normal_time start/end are deprecated normalized model-time units; '
                'use start_seconds/end_seconds instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            return _device_safe_normal_shuffler(**shuffle_config)
        else:
            raise NotImplementedError(f"Shuffle type {shuffle_type} not implemented.")
    else:
        return None
