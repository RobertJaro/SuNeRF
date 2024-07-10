import torch
from astropy import units as u

from sunerf.rendering.base_tracing import SuNeRFRendering


class ThompsonScattering(SuNeRFRendering):

    def __init__(self, Rs_per_ds, **kwargs):
        super().__init__(**kwargs)
        C_0 = 1  # (8.69e-7 * u.cm ** 2).to_value(u.R_sun ** 2) / (Rs_per_ds ** 2)
        self.register_buffer('limb_darkening_coeff', torch.tensor(0.63, dtype=torch.float32))
        self.register_buffer('C_0', torch.tensor(C_0, dtype=torch.float32))
        self.register_buffer('solar_radius',
                             torch.tensor((1 * u.solRad).to_value(u.R_sun) / Rs_per_ds, dtype=torch.float32))

    def raw2outputs(self, raw, z_vals, rays_d, rays_o, query_points, **kwargs):
        r"""
        Convert the raw NeRF output into electron density (1 model output).

        raw: output of NeRF, 2 values per sampled point
        z_vals: distance along the ray as measure from the observer
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        # compute line element (dz) for integration
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists[..., :1], dists], dim=-1)

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
        log_rho = raw[..., 0]
        rho = 10 ** log_rho

        # HOWARD AND TAPPIN 2009 FIG 3
        # working with units of solar radii
        # half angular width of Sun (angle between SQ and ST)
        s_q = torch.norm(query_points, dim=-1)
        s_t = self.solar_radius  # 1 in units of solar radii
        omega = torch.asin(s_t / s_q)
        # print("Max S_q: {} - Omega Minimum: {} - Omega = 0? {}".format(torch.max(s_q), torch.min(omega), (omega == 0).any()))
        # z = distance Q to observer
        z = z_vals * torch.norm(rays_d[..., None, :], dim=-1)  # distance between observer and scattering point Q

        # chi = scattering angle between line of sight (OS) and QS (dot product)
        # chi = torch.acos((rays_d[:, None] * query_points).sum(-1) / (
        #         rays_d.pow(2).sum(-1).pow(0.5)[:, None] * query_points.pow(2).sum(-1).pow(0.5) + 1e-6))
        sin_chi2 = torch.cross(rays_o, rays_d).pow(2).sum(-1)[:, None] / query_points.pow(2).sum(-1)
        u_const = self.limb_darkening_coeff

        # I0 = intensity of the source (Sun) as a power per unit area (of the photosphere) per unit solid angle
        #    = mean solar radiance ( = irradiance / 4pi)

        ln = torch.log((1 + torch.sin(omega)) / torch.cos(omega))
        cos2_sin = torch.cos(omega) ** 2 / (torch.sin(omega))
        A = torch.cos(omega) * torch.sin(omega) ** 2
        B = - (1 / 8) * (1 - 3 * torch.sin(omega) ** 2 - cos2_sin * (1 + 3 * torch.sin(omega) ** 2) * ln)
        C = (4 / 3) - torch.cos(omega) - torch.cos(omega) ** 3 / 3
        D = (1 / 8) * (5 + torch.sin(omega) ** 2 - cos2_sin * (5 - torch.sin(omega) ** 2) * ln)

        # equations 23, 24, 29
        intensity_T = ((1 - u_const) * C + u_const * D)  # I_T in paper - transverse
        intensity_pB = sin_chi2 * ((1 - u_const) * A + u_const * B)  # I_p in Paper

        intensity_tB = 2 * intensity_T - intensity_pB  # I_tot in paper
        # Intensities being negative is unphysical
        intensity_T = torch.abs(intensity_T)
        intensity_pB = torch.abs(intensity_pB)
        intensity_tB = torch.abs(intensity_tB)
        if torch.isnan(intensity_tB).any() or torch.isnan(intensity_pB).any():
            cond = torch.isnan(intensity_tB) | torch.isnan(intensity_pB)
            print(f'Invalid values in intensity_tB or intensity_pB: query points {query_points[cond]}')
            # remove nan values (where omega close to 0)
            intensity_tB = torch.nan_to_num(intensity_tB, nan=0.0, posinf=0.0, neginf=0.0)
            intensity_pB = torch.nan_to_num(intensity_pB, nan=0.0, posinf=0.0, neginf=0.0)

        # intensity (total and polarised) from all electrons
        # for one electron * electron density * weighted by line element ds- separation between sampling points
        point_tB = (self.C_0 * rho) * intensity_tB * dists
        point_pB = (self.C_0 * rho) * intensity_pB * dists

        # sum all intensity contributions along LOS
        pixel_tB = point_tB.sum(-1)[..., None]
        pixel_pB = point_pB.sum(-1)[..., None]
        # print("pixel tB smaller than 0? - {} - Value: {}".format((pixel_tB < 0).any(),(pixel_tB < 0).nonzero()))
        # print("Intensity tB smaller than 0? - {} - Value: {}".format((intensity_tB < 0).any(),(intensity_tB < 0).nonzero()))
        # height and density maps
        # electron_density: (batch, sampling_points, 1), s_q: (batch, sampling_points, 1)
        pixel_density = (rho * dists).sum(1)
        distance_from_sun = (rho * s_q).sum(1) / (rho.sum(1) + 1e-10)
        distance_from_obs = (rho * z).sum(1) / (rho.sum(1) + 1e-10)

        # set the weigths to the intensity contributions (sample primary contributing regions)
        # need weights for sampling for fine model
        weights = rho / (rho.sum(1, keepdim=True) + 1e-10)

        pixel_B = torch.cat([pixel_tB, pixel_pB], dim=-1)

        return {'pixel_B': pixel_B, 'pixel_density': pixel_density, 'distance_from_sun': distance_from_sun,
                'distance_from_obs': distance_from_obs, 'weights': weights}
