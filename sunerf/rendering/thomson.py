import torch
from torch import nn

from sunerf.train.util import asin_safe


class ThomsonScattering(nn.Module):

    def __init__(self, Rs_per_ds, scaling_config=None, **kwargs):
        super().__init__(**kwargs)
        c_0 = 1.0  # (8.69e-7 * u.cm ** 2).to_value(u.R_sun ** 2) / (Rs_per_ds ** 2)
        solar_radius = 1 / Rs_per_ds
        #
        self.limb_darkening_coeff = nn.Parameter(torch.tensor(0.63, dtype=torch.float32), requires_grad=False)
        self.C_0 = nn.Parameter(torch.tensor(c_0, dtype=torch.float32), requires_grad=False)
        self.solar_radius = nn.Parameter(torch.tensor(solar_radius, dtype=torch.float32), requires_grad=False)
        scaling_config = {'type': 'constant', 'value': 1.0} if scaling_config is None else scaling_config
        if scaling_config['type'] == 'constant':
            self.scaling = nn.Parameter(torch.tensor(scaling_config['value'], dtype=torch.float32), requires_grad=False)
        else:
            raise NotImplementedError(f"Scaling type {scaling_config['type']} not implemented.")

    def forward(self, rho, z_vals, rays_d, rays_o, query_points, **kwargs):
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

        # HOWARD AND TAPPIN 2009 FIG 3
        # working with units of solar radii
        # half angular width of Sun (angle between SQ and ST)
        r = query_points[..., :3]  # position of scattering electron
        s_q = torch.norm(r, dim=-1)
        s_t = self.solar_radius  # 1 in units of solar radii
        omega = asin_safe(s_t / s_q)
        # print("Max S_q: {} - Omega Minimum: {} - Omega = 0? {}".format(torch.max(s_q), torch.min(omega), (omega == 0).any()))
        # z = distance Q to observer
        z = z_vals * torch.norm(rays_d[..., None, :], dim=-1)  # distance between observer and scattering point Q

        # chi = scattering angle between line of sight (OS) and QS
        norm = r.pow(2).sum(-1) + 1e-8
        sin_chi2 = torch.cross(r, rays_d[..., None, :], dim=-1).pow(2).sum(-1) / norm

        # Alternative angle calculation
        # sin_chi2 = torch.cross(rays_o, rays_d).pow(2).sum(-1)[:, None] / r.pow(2).sum(-1)

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
        intensity_pB[intensity_pB < 0] = 0
        intensity_tB[intensity_tB < 0] = 0

        if torch.isnan(intensity_tB).any() or torch.isnan(intensity_pB).any():
            cond = torch.isnan(intensity_tB) | torch.isnan(intensity_pB)
            # print(f'Invalid values in intensity_tB or intensity_pB: query points {query_points[cond]}')
            # remove nan values (where omega close to 0)
            intensity_tB = torch.nan_to_num(intensity_tB, nan=0.0, posinf=0.0, neginf=0.0)
            intensity_pB = torch.nan_to_num(intensity_pB, nan=0.0, posinf=0.0, neginf=0.0)
            # raise ValueError('Invalid values in intensity_tB or intensity_pB')

        # intensity (total and polarised) from all electrons
        # for one electron * electron density * weighted by line element ds- separation between sampling points
        rho = rho[..., 0]  # squeeze last dimension
        # TODO clarify z ** -2
        point_tB = self.C_0 * rho * intensity_tB  #* (z ** -2)
        point_pB = self.C_0 * rho * intensity_pB  #* (z ** -2)

        # integrate all intensity contributions along LOS
        image_tB = (point_tB * dists).sum(-1)
        image_pB = (point_pB * dists).sum(-1)

        # print("pixel tB smaller than 0? - {} - Value: {}".format((image_tB < 0).any(),(image_tB < 0).nonzero()))
        # print("Intensity tB smaller than 0? - {} - Value: {}".format((intensity_tB < 0).any(),(intensity_tB < 0).nonzero()))
        # height and density maps
        # electron_density: (batch, sampling_points, 1), s_q: (batch, sampling_points, 1)
        image_density = (rho * dists).sum(1)
        distance_from_sun = (rho * s_q).sum(1) / (rho.sum(1) + 1e-10)
        distance_from_obs = (rho * z).sum(1) / (rho.sum(1) + 1e-10)

        # set the weigths to the intensity contributions (sample primary contributing regions)
        # need weights for sampling for fine model
        weights = rho / (rho.sum(1, keepdim=True) + 1e-10)

        image = torch.stack([image_tB, image_pB], dim=-1)
        image = image * self.scaling

        distance = r.pow(2).sum(-1).pow(0.5)

        return {'image': image, 'density': image_density, 'distance_from_sun': distance_from_sun,
                'distance_from_obs': distance_from_obs, 'weights': weights, 'rho': rho,
                'distance': distance, 'z_vals': z_vals}
