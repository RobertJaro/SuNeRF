import copy

import pytest
import torch
from torch import nn

from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.train.sampling import SphericalSampler, StratifiedSampler


class _ConstantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.seen_times = []

    def forward(self, points):
        self.seen_times.append(points[..., -1].detach().clone())
        shape = (*points.shape[:-1], 1)
        return {'rho': torch.ones(shape, device=points.device) * self.scale}


class _ConstantRenderer(nn.Module):
    def forward(self, rho, z_vals, **kwargs):
        return {
            'image': rho[..., 0].mean(dim=-1),
            'weights': torch.ones_like(z_vals),
        }


def _rendering(shuffle_config=None):
    return BasicRenderingModule(
        model=_ConstantModel(),
        rendering_modules={'test': _ConstantRenderer()},
        Rs_per_ds=1,
        seconds_per_dt=10,
        sampling_config={
            'type': 'spherical',
            'min_distance': 1,
            'max_distance': 2,
            'n_samples': 4,
            'perturb': False,
        },
        hierarchical_sampling_config={'n_samples': 3, 'perturb': False},
        shuffle_config=shuffle_config,
        # These tests compare sampled times to the batch times directly.
        light_travel_time=False,
    )


def _batch():
    rays = torch.tensor([
        [[3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    ])
    return {
        'image': {
            'rays': rays,
            'time': torch.tensor([[0.0], [1.0]]),
            'instrument': 'test',
        }
    }


def test_shell_misses_are_finite_and_explicitly_invalid():
    sampler = SphericalSampler(Rs_per_ds=1, min_distance=1, max_distance=2, n_samples=4)
    sampler.eval()
    rays_o = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    output = sampler(rays_o, rays_d)

    assert output['ray_valid'].tolist() == [True, False]
    assert torch.isfinite(output['points']).all()
    assert torch.isfinite(output['z_vals']).all()
    torch.testing.assert_close(output['z_vals'][1], torch.zeros(4))


def test_shared_observer_uses_each_instruments_integration_radius():
    module = BasicRenderingModule(
        model=_ConstantModel(), rendering_modules={'L5': _ConstantRenderer(), 'PUNCH': _ConstantRenderer()},
        Rs_per_ds=15, seconds_per_dt=1, light_travel_time=False,
        sampling_config={'type': 'spherical', 'max_distance': 30.2, 'n_samples': 8,
                         'max_distance_by_instrument': {'PUNCH': 236.46}},
    ).eval()
    # A ray from L5 with a 100 R_sun impact parameter misses the coronal grid.
    rays = torch.tensor([[[215 / 15, 0., 0.], [-(1 - (100 / 215)**2)**0.5, 100 / 215, 0.]]])
    batch = {key: {'rays': rays, 'time': torch.zeros(1, 1), 'instrument': key} for key in ('L5', 'PUNCH')}
    out = module(batch)
    assert out['ray_valid'].tolist() == [False, True]
    z = out['z_vals'][1]
    torch.testing.assert_close(z[0], torch.tensor(0.))
    torch.testing.assert_close((rays[0, 0] + z[-1] * rays[0, 1]).norm() * 15, torch.tensor(236.46))


def test_stratified_sampler_uses_a_fixed_path_length_per_ray():
    sampler = StratifiedSampler(Rs_per_ds=1, max_distance=2, n_samples=4)
    sampler.eval()
    rays_o = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [float('nan'), 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]).requires_grad_()
    output = sampler(rays_o, rays_d)

    # The disk-center ray ends at the photosphere, the ray that never comes
    # close to the Sun keeps the full interval around the observer distance.
    assert output['ray_valid'].tolist() == [True, True, False]
    torch.testing.assert_close(output['z_vals'][0, [0, -1]], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(output['z_vals'][1, [0, -1]], torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(output['z_vals'][2], torch.zeros(4))
    assert torch.isfinite(output['points']).all()

    output['points'][output['ray_valid']].square().sum().backward()
    assert torch.isfinite(rays_d.grad).all()


def test_ray_gradients_are_finite_for_grazing_rays_and_shell_misses():
    # Learned pointing alignment differentiates the shell intersection with
    # respect to the rays. In float32 the discriminant of a grazing ray is
    # exactly zero, where the derivative of the square root is infinite.
    rays_o = torch.tensor([
        [215.0, 0.0, 0.0],  # disk center
        [215.0, 1.2, 0.0],  # off limb
        [215.0, 1.0, 0.0],  # tangent to the inner sphere
        [215.0, 1.5, 0.0],  # tangent to the outer sphere
        [215.0, 3.0, 0.0],  # misses the shell
    ])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0]]).repeat(len(rays_o), 1).requires_grad_()
    sampler = SphericalSampler(Rs_per_ds=1, min_distance=1, max_distance=1.5, n_samples=4, perturb=False)

    sampled = sampler(rays_o, rays_d)
    sampled['points'][sampled['ray_valid']].square().sum().backward()

    assert sampled['ray_valid'].tolist() == [True, True, True, False, False]
    assert torch.isfinite(rays_d.grad).all()
    assert torch.count_nonzero(rays_d.grad[:2]) > 0


def test_invalid_rays_are_not_sampled_at_the_solar_center():
    # A non-finite ray keeps zero path length; its samples must stay away from
    # r = 0, where the density model takes log(r).
    rays_o = torch.tensor([[215.0, 0.0, 0.0], [float('nan'), 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    sampler = SphericalSampler(Rs_per_ds=1, min_distance=1, max_distance=1.5, n_samples=4, perturb=False)

    sampled = sampler(rays_o, rays_d)

    assert sampled['ray_valid'].tolist() == [True, False]
    assert torch.isfinite(sampled['points']).all()
    assert (torch.linalg.norm(sampled['points'][1], dim=-1) > 0).all()


def test_rendering_masks_shell_misses_instead_of_forming_an_image():
    rendering = _rendering()
    rendering.eval()

    output = rendering(_batch())

    assert output['ray_valid'].tolist() == [True, False]
    assert output['model_out']['image']['ray_valid'].tolist() == [True, False]
    assert output['model_out']['image']['image'][0].item() == 1
    assert output['model_out']['image']['image'][1].item() == 0


def test_time_shuffle_is_train_only_overrideable_and_does_not_mutate_batch():
    rendering = _rendering({
        'type': 'normal_time', 'start_seconds': 50.0,
        'end_seconds': 1.0, 'iterations': 10
    })
    batch = _batch()
    original = copy.deepcopy(batch)

    rendering.train()
    rendering(batch)
    torch.testing.assert_close(batch['image']['time'], original['image']['time'])
    assert not torch.equal(
        rendering.model.seen_times[-1][:, 0], original['image']['time'][:, 0]
    )

    rendering.model.seen_times.clear()
    rendering.eval()
    rendering(batch)
    torch.testing.assert_close(
        rendering.model.seen_times[0][:, 0], original['image']['time'][:, 0]
    )

    rendering.model.seen_times.clear()
    rendering.train()
    rendering(batch, shuffle=False)
    torch.testing.assert_close(
        rendering.model.seen_times[0][:, 0], original['image']['time'][:, 0]
    )


def test_rendering_configuration_mappings_are_not_mutated():
    sampling = {'type': 'spherical', 'min_distance': 1, 'max_distance': 2}
    hierarchy = {'type': 'hierarchical', 'n_samples': 3}
    shuffle = {'type': 'time', 'probability': 0.5}
    expected = copy.deepcopy((sampling, hierarchy, shuffle))

    BasicRenderingModule(
        model=_ConstantModel(),
        rendering_modules={'test': _ConstantRenderer()},
        Rs_per_ds=1,
        seconds_per_dt=10,
        sampling_config=sampling,
        hierarchical_sampling_config=hierarchy,
        shuffle_config=shuffle,
    )

    assert (sampling, hierarchy, shuffle) == expected


def test_normal_time_augmentation_converts_physical_seconds_to_model_time():
    config = {
        'type': 'normal_time',
        'start_seconds': 1800,
        'end_seconds': 30,
        'iterations': 10,
    }
    rendering = BasicRenderingModule(
        model=_ConstantModel(),
        rendering_modules={'test': _ConstantRenderer()},
        Rs_per_ds=1,
        seconds_per_dt=3600,
        sampling_config={'min_distance': 1, 'max_distance': 2},
        hierarchical_sampling_config={'n_samples': 3},
        shuffle_config=config,
    )

    assert rendering.shuffler.scaling.item() == 0.5
    assert rendering.shuffler.end == 30 / 3600
    assert rendering.shuffler.start_seconds == 1800
    assert rendering.shuffler.end_seconds == 30
    assert 'gamma' not in rendering.shuffler.state_dict()
    rendering.shuffler.to(dtype=torch.float64)
    assert rendering.shuffler.gamma.dtype == torch.float64
    assert config['start_seconds'] == 1800


def test_normal_time_jitter_holds_during_cold_phase_then_decays():
    rendering = BasicRenderingModule(
        model=_ConstantModel(),
        rendering_modules={'test': _ConstantRenderer()},
        Rs_per_ds=1,
        seconds_per_dt=3600,
        sampling_config={'min_distance': 1, 'max_distance': 2},
        hierarchical_sampling_config={'n_samples': 3},
        shuffle_config={
            'type': 'normal_time',
            'start_seconds': 3600,
            'end_seconds': 0,
            'iterations': 10,
            'cold_iterations': 5,
        },
    )
    shuffler = rendering.shuffler
    for step, expected in [(0, 1.0), (5, 1.0), (6, 0.9), (10, 0.5), (15, 0.0), (100, 0.0)]:
        shuffler.set_step(step)
        assert shuffler.scaling.item() == pytest.approx(expected)
