import torch
from astropy import units as u

from sunerf.model.thomson import ThomsonSuNeRFModule


def test_scaled_continuity_residual_is_the_relative_violation_at_every_radius():
    Rs_per_ds, seconds_per_dt, v_kms, exponent = 100.0, 86400.0, 300.0, 3.0
    v0 = (v_kms * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt

    # Constant radial wind with log rho = -exponent * log r. Mass conservation
    # requires exponent 2; the residual is (2 - exponent) * v0 / r.
    radius = torch.tensor([0.015, 0.1, 1.0])
    points = torch.zeros(3, 4)
    points[:, 0] = radius
    points.requires_grad_(True)
    r = torch.norm(points[:, :3], dim=-1, keepdim=True)
    v = v0 * points[:, :3] / r
    log_rho = -exponent * torch.log(r)
    jacobian = lambda field: torch.stack([
        torch.autograd.grad(field[:, i].sum(), points, create_graph=True)[0]
        for i in range(field.shape[1])
    ], dim=1)
    _, terms = ThomsonSuNeRFModule.compute_log_continuity_loss(v, jacobian(log_rho), jacobian(v))

    class _Module:
        continuity_reference_velocity = v0
        scale_continuity_residual = ThomsonSuNeRFModule.scale_continuity_residual

    torch.testing.assert_close(terms['residual'], (2 - exponent) * v0 / radius, rtol=1e-4, atol=1e-4)
    scaled = _Module().scale_continuity_residual(terms['residual'], points)
    torch.testing.assert_close(scaled, torch.full((3,), 2 - exponent), rtol=1e-4, atol=1e-4)
