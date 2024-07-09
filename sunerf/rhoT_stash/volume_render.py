from typing import Tuple, Callable

import astropy.units as u
import numpy as np
import torch
from astropy.visualization import ImageNormalize, AsinhStretch
from scipy import interpolate
from sunpy.io.special import read_genx
from torch import nn

from sunerf.train.sampling import sample_hierarchical
from sunerf.train.sampling import sample_non_uniform_box


class VolumeRender():
    def __init__(self, aia_response_path='', maximum_AIA_intensity=1, density_normalization=1, aia_exp_time=2.9):
        """Class to make NeRF volume renders that also can read AIA temperature response functions

        Parameters
        ----------
        aia_response_path : string
            path to file containing SDO/AIA temperature resp. functions
        density_normalization : float
            density normalization factor
        aia__time: float
            typical aia exposure time
        """

        # AIA temperature resp. functions.
        aia_resp = read_genx(aia_response_path + "aia_temp_resp.genx")

        # Exposure time
        # = s_map.meta["exptime"]

        temperature = np.power(10, aia_resp['A193']['LOGTE'])  # temperature
        response = aia_resp['A193'][
                       'TRESP'] * u.cm ** 5 * aia_exp_time  # multiply response by typical AIA exposure time
        response = response.to(u.um ** 5).value

        self.tres193 = interpolate.interp1d(temperature, response)
        self.density_normalization = density_normalization
        self.maximum_AIA_intensity = maximum_AIA_intensity

        # Normalization of the images (0 to 1)
        self.sdo_norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=AsinhStretch(0.005), clip=True),
                          193: ImageNormalize(vmin=0, vmax=9800, stretch=AsinhStretch(0.005), clip=True),
                          211: ImageNormalize(vmin=0, vmax=5800, stretch=AsinhStretch(0.005), clip=True),
                          304: ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True)}

    def raw2outputs_density_temp(self,
                                 raw: torch.Tensor,
                                 z_vals: torch.Tensor,
                                 rays_d: torch.Tensor,
                                 raw_noise_std: float = 0.0
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the raw NeRF output into emission and absorption.

        raw: output of NeRF, 2 values per sampled point
        z_vals: distance along the ray as measure from the origin
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        # compute line element (dz) for integration
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists[..., :1], dists], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., -1].shape) * raw_noise_std

        # density ([..., 0]; epsilon(z)) and temperature ([..., 1]; kappa(z))
        # dI = density^2*temperature_response(T) dz ;

        # Get density in units of um^-3
        density = nn.functional.relu(raw[..., 0]) / self.density_normalization
        # Get temperature
        temperature = nn.functional.relu(raw[..., 1])
        # Get the AIA temperature function in units of um^-5
        temperature_response = self.tres193(temperature)  # TODO: Modify for arbritrary channels

        # Calculate intensity as density^2*temperature_response(T)  dz
        emerging_intensity = density * density * temperature_response

        # Get absorption coefficient
        absortpion_coefficient = nn.functional.relu(raw[..., 2])

        # Calculate absorption
        absorption = torch.exp(-absortpion_coefficient * density * dists)

        # compute total absorption for each light ray (intensity)
        # how much light is transmitted from each sampled point
        total_absorption = cumprod_exclusive(
            absorption + 1e-10)  # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)

        emerging_intensity = emerging_intensity * total_absorption  # integrate total intensity [n_rays, n_samples - 1]
        emerging_intensity = 0.5 * (emerging_intensity[:, :-1] + emerging_intensity[:, 1:]) * dists[:, :-1]

        # sum all intensity contributions
        pixel_intensity = self.sdo_norms[193](emerging_intensity.sum(1)[:, None]) / self.maximum_aia_intensity

        # set the weigths to the intensity contributions
        weights = emerging_intensity
        weights = torch.cat([weights, weights[..., -1][:, None]], dim=-1)

        # Estimated depth map is predicted distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        # disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
        #                          depth_map / torch.sum(weights, -1))

        # Sum of weights along each ray. In [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        return pixel_intensity, depth_map, acc_map, weights


def raw2outputs(raw: torch.Tensor,
                z_vals: torch.Tensor,
                rays_d: torch.Tensor,
                raw_noise_std: float = 0.0
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into emission and absorption.

    raw: output of NeRF, 2 values per sampled point
    z_vals: distance along the ray as measure from the origin
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    # compute line element (dz) for integration
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists[..., :1], dists], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., -1].shape) * raw_noise_std

    # emission ([..., 0]; epsilon(z)) and absorption ([..., 1]; kappa(z)) coefficient per unit volume
    # dtau = - kappa dz
    # I' / I = - kappa dz --> I' emerging intensity; I incident intensity;
    intensity = torch.exp(raw[..., 0]) * dists  # emission per sampled point [n_rays, n_samples]

    absorption = torch.exp(
        -nn.functional.relu(raw[..., 1] + noise) * dists)  # transmission per sampled point [n_rays, n_samples]
    # [1, .9, 1, 0, 0, 1] --> less dense objects transmit light (1); dense objects absorbe light (0)

    # compute total absorption for each light ray (intensity)
    # how much light is transmitted from each sampled point
    total_absorption = cumprod_exclusive(
        absorption + 1e-10)  # first intensity has no absorption (1, t[0], t[0] * t[1], t[0] * t[1] * t[2], ...)
    # [(1), 1, .9, .9, 0, 0] --> total absorption for each point along the ray
    # apply absorption to intensities
    emerging_intensity = intensity * total_absorption  # integrate total intensity [n_rays, n_samples - 1]
    # sum all intensity contributions
    pixel_intensity = emerging_intensity.sum(1)[:, None]

    # stretch value range to images --> ASINH
    pixel_intensity = torch.asinh(pixel_intensity / 0.005) / 5.991471  # normalization

    # set the weigths to the intensity contributions (sample primary contributing regions)
    weights = emerging_intensity
    weights = weights / (weights.sum(1)[:, None] + 1e-10)

    return pixel_intensity, weights, absorption


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
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
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.

    return cumprod


def nerf_forward(rays_o: torch.Tensor,
                 rays_d: torch.Tensor,
                 times: torch.Tensor,
                 coarse_model: nn.Module,
                 fine_model: nn.Module,
                 near: float,
                 far: float,
                 encoding_fn: Callable[[torch.Tensor], torch.Tensor],
                 sample_stratified: Callable[[torch.Tensor, torch.Tensor, float, float, int], Tuple[
                     torch.Tensor, torch.Tensor]] = sample_non_uniform_box,
                 kwargs_sample_stratified: dict = None,
                 n_samples_hierarchical: int = 0,
                 kwargs_sample_hierarchical: dict = None):
    r"""_summary_

        Compute forward pass through model.

        Args:
            rays_o (tensor): Origin of rays
            rays_d (tensor): Direction of rays
            times (tensor): Times of maps
            near (float): Beginning point of ray sampling (in solar radii, converted to pixels)
            far (float): Ending point of ray sampling (in solar radii, converted to pixels)
            encoding_fn (Callable[[torch.Tensor], torch.Tensor]): Encoding function for the inputs
            coarse_model (nn.Module): Function sampling coarse points along rays.
            sample_stratified: function that determines the points that will be sampled by the coarse model
            kwargs_sample_stratified (dict): Number of samples per ray, along with perturbation boolean.
            n_samples_hierarchical (int): Number of samples per ray.
            kwargs_sample_hierarchical: Contains the perturbation boolean for sampling (If set, applies noise to sample positions)
            fine_model (nn.Module): Function to refine the sampling along rays.
        Returns:
            outputs: Synthesized filtergrams/images.
        """

    # vol_render = volume_render()

    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}

    # Sample query points along each ray.
    query_points, z_vals = sample_stratified(
        rays_o, rays_d, near, far, **kwargs_sample_stratified)

    # add time to query points
    exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
    query_points_time = torch.cat([query_points, exp_times], -1)  # --> (x, y, z, t)

    # Prepare points --> encoding.
    enc_query_points = encoding_fn(query_points_time.view(-1, 4))

    # Coarse model pass.
    raw = coarse_model(enc_query_points)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the filtergrams.
    channel_map, weights, absorption = raw2outputs(raw, z_vals, rays_d)
    outputs = {'z_vals_stratified': z_vals}

    # Fine model pass.
    if n_samples_hierarchical > 0:
        # Save previous outputs to return.
        channel_map_0 = channel_map

        # Apply hierarchical sampling for fine query points.
        query_points, z_vals_combined, z_hierarch = sample_hierarchical(
            rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
            **kwargs_sample_hierarchical)

        # add time to query points = expand to dimensions of query points and slice one dimension
        exp_times = times[:, None].repeat(1, query_points.shape[1], 1)
        query_points_time = torch.cat([query_points, exp_times], -1)

        # Prepare inputs as before.
        enc_query_points = encoding_fn(query_points_time.view(-1, 4))

        # Forward pass new samples through fine model.
        fine_model = fine_model if fine_model is not None else coarse_model
        raw = fine_model(enc_query_points)
        raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the filtergrams.
        channel_map, weights, absorption = raw2outputs(raw, z_vals_combined, rays_d)

        # Store outputs.
        outputs['z_vals_hierarchical'] = z_hierarch
        outputs['channel_map_0'] = channel_map_0

    # compute image of absorption
    absorption_map = (1 - absorption).sum(-1)

    # compute regularization of absorption
    distance = query_points.pow(2).sum(-1).pow(0.5)
    regularization = torch.relu(distance - 1.2) * (1 - absorption)  # penalize absorption past 1.2 solar radii

    # compute image of height
    height_map = (weights * distance).sum(-1)

    # Store outputs.
    outputs['channel_map'] = channel_map
    outputs['weights'] = weights
    outputs['height_map'] = height_map
    outputs['absorption_map'] = absorption_map
    outputs['regularization'] = regularization
    return outputs
