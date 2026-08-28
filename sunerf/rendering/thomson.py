import torch
from torch import nn


def _atanh_over_x_minus_one(x: torch.Tensor) -> torch.Tensor:
    """Return ``atanh(x) / x - 1`` without outer-corona cancellation.

    The direct expression loses several digits when the apparent solar radius
    is small. A Horner series is accurate in float32 below ``x = 0.45``; above
    that threshold the direct expression is well conditioned. The series is
    ``sum(x**(2k) / (2k + 1), k=1...)``.
    """
    y = x.square()
    # Eight terms make the truncation error smaller than float32 rounding at
    # the switch point while remaining cheap relative to the network.
    coefficients = (
        1.0 / 3.0, 1.0 / 5.0, 1.0 / 7.0, 1.0 / 9.0,
        1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0,
    )
    series = torch.full_like(y, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        series = series * y + coefficient
    series = y * series
    direct = torch.atanh(x) / x - 1.0
    return torch.where(x < 0.45, series, direct)


def _van_de_hulst_coefficients(sin_omega: torch.Tensor):
    """Compute the finite-Sun Thomson coefficients in the input dtype.

    These forms avoid the small-angle subtractions in the textbook equations,
    so float32 remains accurate through the outer corona without temporarily
    promoting the full ray grid to float64.
    """
    y = sin_omega.square()
    cos_omega = torch.sqrt((1.0 - y).clamp_min(0.0))

    # A is already well conditioned. Factor C and rationalize 1-cos(omega)
    # to avoid subtracting two nearly equal float32 values.
    A = cos_omega * y
    C = y * (cos_omega.square() + cos_omega + 4.0) / (3.0 * (1.0 + cos_omega))

    # q = (1-y) atanh(x)/x = 1 + delta. Rewriting B and D in terms of
    # delta removes their cancellation of O(1) terms at small x.
    h = _atanh_over_x_minus_one(sin_omega)
    delta = h - y - y * h
    B = (6.0 * y + (1.0 + 3.0 * y) * delta) / 8.0
    D = (2.0 * y - (5.0 - y) * delta) / 8.0
    return A, B, C, D


class ThomsonScattering(nn.Module):

    def __init__(self, Rs_per_ds, **kwargs):
        super().__init__(**kwargs)
        solar_radius = 1 / Rs_per_ds
        #
        self.limb_darkening_coeff = nn.Parameter(torch.tensor(0.63, dtype=torch.float32), requires_grad=False)
        self.solar_radius = nn.Parameter(torch.tensor(solar_radius, dtype=torch.float32), requires_grad=False)

    def forward(self, rho, z_vals, rays_d, rays_o, query_points, **kwargs):
        r"""
        Convert the raw NeRF output into electron density (1 model output).

        raw: output of NeRF, 2 values per sampled point
        z_vals: distance along the ray as measure from the observer
        """

        # Composite trapezoidal node weights for a finite, potentially
        # nonuniform line-of-sight grid. These sum to the full sampled chord.
        dz = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([
            0.5 * dz[..., :1],
            0.5 * (dz[..., :-1] + dz[..., 1:]),
            0.5 * dz[..., -1:],
        ], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # For total and polarised brightness need... (Howard and Tappin 2009):
        # * Omega (half angular width of Sun - depends on distance to sun)
        # * z (distance from Q to observer)
        # * chi (scattering angle between observer and S)
        # * u (limb darkening coeff based on wavelength - constant for white light?)
        # * I0 (intensity/ of the Sun - will vary with solar cycle - look up table?)
        # * sigma_e (scattering constant - eqn 3)

        # HOWARD AND TAPPIN 2009 FIG 3
        # working with units of solar radii
        # half angular width of Sun (angle between SQ and ST)
        r = query_points[..., :3]  # position of scattering electron
        s_q = torch.norm(r, dim=-1)
        s_t = self.solar_radius  # 1 in units of solar radii
        # z = distance Q to observer
        z = z_vals * torch.norm(rays_d[..., None, :], dim=-1)  # distance between observer and scattering point Q

        # chi = scattering angle between line of sight (OS) and QS
        geometry_dtype = r.dtype
        dtype_info = torch.finfo(geometry_dtype)
        d_hat = rays_d / torch.linalg.norm(
            rays_d, dim=-1, keepdim=True
        ).clamp_min(dtype_info.eps)
        sin_chi2 = torch.cross(r, d_hat[..., None, :], dim=-1).pow(2).sum(-1) \
            / r.pow(2).sum(-1).clamp_min(dtype_info.eps)
        sin_chi2 = sin_chi2.clamp(min=0.0, max=1.0)

        # Alternative angle calculation
        # sin_chi2 = torch.cross(rays_o, rays_d).pow(2).sum(-1)[:, None] / r.pow(2).sum(-1)

        u_const = self.limb_darkening_coeff

        # I0 = intensity of the source (Sun) as a power per unit area (of the photosphere) per unit solid angle
        #    = mean solar radiance ( = irradiance / 4pi)

        sin_omega = (s_t / s_q).clamp(
            min=dtype_info.eps, max=1.0 - dtype_info.eps
        )
        A, B, C, D = _van_de_hulst_coefficients(sin_omega)

        # equations 23, 24, 29
        intensity_T = ((1 - u_const) * C + u_const * D)  # I_T in paper - transverse

        intensity_pB = sin_chi2 * ((1 - u_const) * A + u_const * B)  # I_p in Paper
        intensity_tB = 2 * intensity_T - intensity_pB  # I_tot in paper

        # remove nan values (where omega close to 0)
        intensity_tB = torch.nan_to_num(intensity_tB, nan=0.0, posinf=0.0, neginf=0.0)
        intensity_pB = torch.nan_to_num(intensity_pB, nan=0.0, posinf=0.0, neginf=0.0)
        # intensity (total and polarised) from all electrons
        # for one electron * electron density * weighted by line element ds- separation between sampling points
        rho = rho[..., 0]  # squeeze last dimension
        point_tB = rho * intensity_tB
        point_pB = rho * intensity_pB

        # integrate all intensity contributions along LOS
        image_tB = (point_tB * dists).sum(1)
        image_pB = (point_pB * dists).sum(1)

        # print("pixel tB smaller than 0? - {} - Value: {}".format((image_tB < 0).any(),(image_tB < 0).nonzero()))
        # print("Intensity tB smaller than 0? - {} - Value: {}".format((intensity_tB < 0).any(),(intensity_tB < 0).nonzero()))
        # height and density maps
        # electron_density: (batch, sampling_points, 1), s_q: (batch, sampling_points, 1)
        density_weights = rho * dists
        image_density = density_weights.sum(1)
        density_weight_sum = density_weights.sum(1) + 1e-10
        distance_from_sun = (density_weights * s_q).sum(1) / density_weight_sum
        distance_from_obs = (density_weights * z).sum(1) / density_weight_sum

        # set the weigths to the intensity contributions (sample primary contributing regions)
        # need weights for sampling for fine model
        weights = (point_pB + point_tB) * dists
        weights = weights / (weights.sum(1, keepdim=True) + 1e-10)

        image = torch.stack([image_tB, image_pB], dim=-1)

        distance = r.pow(2).sum(-1).pow(0.5)

        return {'image': image, 'density': image_density, 'distance_from_sun': distance_from_sun,
                'distance_from_obs': distance_from_obs, 'weights': weights, 'rho': rho,
                'distance': distance, 'z_vals': z_vals}
