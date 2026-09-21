import copy

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
    for sampler in (
        SphericalSampler(Rs_per_ds=1, min_distance=1, max_distance=2, n_samples=4),
        StratifiedSampler(Rs_per_ds=1, max_distance=2, n_samples=4),
    ):
        sampler.eval()
        rays_o = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        rays_d = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        output = sampler(rays_o, rays_d)

        assert output['ray_valid'].tolist() == [True, False]
        assert torch.isfinite(output['points']).all()
        assert torch.isfinite(output['z_vals']).all()
        torch.testing.assert_close(output['z_vals'][1], torch.zeros(4))


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
