import torch
from torch import nn

from sunerf.physics.thomson import LIMB_DARKENING_COEFF
from sunerf.rendering.base_tracing import BasicRenderingModule
from sunerf.rendering.thomson import ThomsonScattering, _van_de_hulst_coefficients
from sunerf.train.sampling import SphericalSampler


def test_renderer_uses_density_normalization_limb_darkening_coefficient():
    renderer = ThomsonScattering(Rs_per_ds=100)

    torch.testing.assert_close(
        renderer.limb_darkening_coeff,
        torch.tensor(LIMB_DARKENING_COEFF),
    )


def test_nonuniform_grid_uses_finite_interval_trapezoidal_weights():
    renderer = ThomsonScattering(Rs_per_ds=100)
    z_vals = torch.tensor([[0.0, 1.0, 3.0, 10.0]])
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])
    rays_o = torch.zeros_like(rays_d)
    query_points = torch.tensor([[[0.0, 0.5, 0.0, 0.0]]]).expand(-1, 4, -1)
    rho = torch.ones((1, 4, 1))

    output = renderer(rho, z_vals, rays_d, rays_o, query_points)

    torch.testing.assert_close(output['density'], torch.tensor([10.0]))


def test_outer_corona_kernel_remains_physically_ordered():
    renderer = ThomsonScattering(Rs_per_ds=100)
    z_vals = torch.tensor([[0.0, 1.0]])
    rays_d = torch.tensor([[0.0, 0.0, -1.0]])
    rays_o = torch.zeros_like(rays_d)
    query_points = torch.tensor([[[0.0, 0.8, 0.0, 0.0]]]).expand(-1, 2, -1)
    rho = torch.ones((1, 2, 1))

    image = renderer(rho, z_vals, rays_d, rays_o, query_points)['image']
    ratio = image[..., 1] / image[..., 0]

    assert torch.isfinite(image).all()
    assert (image >= 0).all()
    assert (image[..., 1] <= image[..., 0]).all()
    torch.testing.assert_close(ratio, torch.tensor([0.99986]), rtol=2e-5, atol=2e-5)


def test_float32_coefficients_match_float64_reference_across_corona():
    radii = torch.logspace(
        torch.log10(torch.tensor(1.01)),
        torch.log10(torch.tensor(110.0)),
        4096,
        dtype=torch.float32,
    )
    x = radii.reciprocal()
    actual = _van_de_hulst_coefficients(x)

    # Reference uses the represented float32 abscissae, isolating kernel error
    # from input quantization near the solar limb.
    x_ref = x.double()
    y_ref = x_ref.square()
    cos_ref = torch.sqrt(1.0 - y_ref)
    log_ref = torch.atanh(x_ref)
    q_ref = (1.0 - y_ref) / x_ref * log_ref
    expected = (
        cos_ref * y_ref,
        -(1.0 - 3.0 * y_ref - q_ref * (1.0 + 3.0 * y_ref)) / 8.0,
        4.0 / 3.0 - cos_ref - cos_ref.pow(3) / 3.0,
        (5.0 + y_ref - q_ref * (5.0 - y_ref)) / 8.0,
    )

    for coefficient, reference in zip(actual, expected):
        assert coefficient.dtype == torch.float32
        torch.testing.assert_close(
            coefficient.double(), reference, rtol=8e-7, atol=2e-9
        )


def test_float32_kernel_is_finite_ordered_and_differentiable():
    renderer = ThomsonScattering(Rs_per_ds=1)
    radius = torch.logspace(
        torch.log10(torch.tensor(1.01)),
        torch.log10(torch.tensor(110.0)),
        512,
    )
    z_vals = torch.stack([torch.zeros_like(radius), torch.ones_like(radius)], dim=-1)
    rays_d = torch.tensor([[0.0, 0.0, -1.0]]).expand(radius.shape[0], -1)
    rays_o = torch.zeros_like(rays_d)
    points = torch.zeros((radius.shape[0], 2, 4), dtype=torch.float32)
    points[..., 1] = radius[:, None]
    rho = torch.ones((radius.shape[0], 2, 1), requires_grad=True)

    image = renderer(rho, z_vals, rays_d, rays_o, points)['image']

    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert (image >= 0).all()
    assert (image[..., 1] <= image[..., 0]).all()
    image.sum().backward()
    assert rho.grad is not None
    assert torch.isfinite(rho.grad).all()


class _CountingLinearDensity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.sample_counts = []

    def forward(self, coords):
        self.sample_counts.append(coords.shape[0] * coords.shape[1])
        return {'rho': self.scale * coords[..., :1]}


class _TrapezoidDensity(nn.Module):
    def forward(self, rho, z_vals, rays_d, **kwargs):
        dz = z_vals[..., 1:] - z_vals[..., :-1]
        widths = torch.cat([
            0.5 * dz[..., :1],
            0.5 * (dz[..., :-1] + dz[..., 1:]),
            0.5 * dz[..., -1:],
        ], dim=-1)
        widths = widths * torch.linalg.vector_norm(rays_d, dim=-1, keepdim=True)
        density = (rho[..., 0] * widths).sum(dim=-1)
        weights = widths / widths.sum(dim=-1, keepdim=True)
        return {'density': density, 'weights': weights}


def test_hierarchical_render_reuses_coarse_values_with_exact_spacing_and_gradients():
    model = _CountingLinearDensity()
    rendering = BasicRenderingModule(
        model=model,
        rendering_modules={'test': _TrapezoidDensity()},
        Rs_per_ds=1,
        seconds_per_dt=1,
        light_travel_time=False,
        sampling_config={
            'type': 'spherical',
            'min_distance': 1,
            'max_distance': 2,
            'n_samples': 4,
            'perturb': False,
        },
        hierarchical_sampling_config={'n_samples': 3, 'perturb': False},
    )
    rays = torch.tensor([[[3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
    batch = {'image': {
        'rays': rays,
        'time': torch.zeros((1, 1)),
        'instrument': 'test',
    }}

    output = rendering(batch)
    density = output['model_out']['image']['density']

    # Integral of x=3-z over z=[1,2] is exactly 1.5. The final grid is
    # nonuniform, but composite trapezoidal integration remains exact here.
    torch.testing.assert_close(density, torch.tensor([1.5]))
    assert model.sample_counts == [4, 3]
    density.sum().backward()
    torch.testing.assert_close(model.scale.grad, torch.tensor(1.5))


def test_spherical_sampler_preserves_boundaries_and_is_deterministic_in_eval():
    sampler = SphericalSampler(
        Rs_per_ds=1,
        min_distance=1,
        max_distance=2,
        n_samples=8,
        perturb=True,
    )
    rays_o = torch.tensor([[3.0, 0.0, 0.0]])
    rays_d = torch.tensor([[-1.0, 0.0, 0.0]])

    sampler.train()
    first_train = sampler(rays_o, rays_d)['z_vals']
    second_train = sampler(rays_o, rays_d)['z_vals']
    torch.testing.assert_close(first_train[:, [0, -1]], second_train[:, [0, -1]])
    assert not torch.equal(first_train[:, 1:-1], second_train[:, 1:-1])

    sampler.eval()
    first_eval = sampler(rays_o, rays_d)['z_vals']
    second_eval = sampler(rays_o, rays_d)['z_vals']
    torch.testing.assert_close(first_eval, second_eval)
    torch.testing.assert_close(first_train[:, [0, -1]], first_eval[:, [0, -1]])
