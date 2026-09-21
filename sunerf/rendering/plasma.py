import copy

import numpy as np
import torch
from torch import nn

from sunerf.model.model import AbsorptionModel, ConstantAbsorptionModel
from sunerf.absorption.torch import PhotoionizationOpacity
from sunerf.physics.euv import R_SUN_CM, torch_trapezoid_node_weights
from sunerf.configuration import canonical_channel_id
from sunerf.response import load_response_artifact


class PlasmaRadiativeTransfer(nn.Module):

    def __init__(self, temperature_response_config, log_T_range, absorption_model=None):
        super().__init__()

        channels = temperature_response_config.get('channels', None)
        source_artifact = load_response_artifact(temperature_response_config['artifact'])
        source_response_id = source_artifact.response_id
        # The response keeps its native (density, temperature) axes. The
        # renderer interpolates it at the pointwise plasma state instead of
        # resampling it onto a model temperature grid.
        artifact = source_artifact.select_channels(channels)
        provenance = dict(artifact.provenance)
        provenance['source_response_id'] = source_response_id
        artifact = artifact.updated(provenance=provenance)

        # Store every response as (density, temperature, channel). A singleton
        # density dimension handles density-independent tables without a
        # separate rendering path.
        if artifact.log_density is None:
            response = artifact.response.T[None]
            log_density = np.empty(0, dtype=np.float32)
        else:
            response = np.transpose(artifact.response, (1, 2, 0))
            log_density = artifact.log_density

        # ``log_T`` is the diagnostic temperature grid of LOS emission-measure
        # histograms; it does not enter image formation.
        diagnostic_log_T = torch.as_tensor(np.asarray(log_T_range), dtype=torch.float32)
        self.register_buffer('log_T', diagnostic_log_T)
        self.register_buffer(
            'temperature_bin_weights', torch_trapezoid_node_weights(diagnostic_log_T)
        )
        self.register_buffer(
            'response_log_T',
            torch.as_tensor(artifact.log_temperature, dtype=torch.float32),
        )
        self.register_buffer('log_density_axis', torch.as_tensor(log_density, dtype=torch.float32))
        self.register_buffer('temperature_response', torch.as_tensor(response, dtype=torch.float32))

        cutoff = temperature_response_config.get('temperature_cutoff')
        if cutoff is None:
            cutoff_temperature, cutoff_width = 0.0, 1.0
        else:
            unknown = set(cutoff).difference({'T_cut_K', 'delta_T_K'})
            if unknown or set(cutoff) != {'T_cut_K', 'delta_T_K'}:
                raise ValueError('temperature_cutoff must define exactly T_cut_K and delta_T_K')
            cutoff_temperature = float(cutoff['T_cut_K'])
            cutoff_width = float(cutoff['delta_T_K'])
            if not (np.isfinite(cutoff_temperature) and cutoff_temperature > 0):
                raise ValueError('temperature_cutoff.T_cut_K must be finite and positive')
            if not (np.isfinite(cutoff_width) and cutoff_width > 0):
                raise ValueError('temperature_cutoff.delta_T_K must be finite and positive')
        self.temperature_cutoff = None if cutoff is None else {
            'T_cut_K': cutoff_temperature, 'delta_T_K': cutoff_width,
        }
        self.register_buffer(
            'temperature_cutoff_K', torch.tensor(cutoff_temperature, dtype=torch.float32)
        )
        self.register_buffer(
            'temperature_cutoff_width_K', torch.tensor(cutoff_width, dtype=torch.float32)
        )

        self.absorption = absorption_model is not None
        print('Using absorption:', self.absorption)

        learnable = bool(temperature_response_config.get('learnable', False))
        self.learnable = learnable
        if 'scaling' in temperature_response_config:
            raise ValueError(
                'temperature_response.scaling is unsupported; fixed image divisors '
                'belong to the dataset and calibration uses bounded learnable gains'
            )
        self.register_buffer(
            'instrument_scaling_center',
            torch.zeros(len(artifact.channels), dtype=torch.float32),
        )
        self.instrument_scaling = nn.Parameter(
            torch.zeros(len(artifact.channels), dtype=torch.float32),
            requires_grad=learnable,
        )
        self.global_reference = bool(
            temperature_response_config.get('global_reference', False)
        )
        if self.global_reference and not learnable:
            raise ValueError('global_reference requires learnable: true')
        self.common_instrument_scaling = nn.Parameter(
            torch.zeros((), dtype=torch.float32),
            requires_grad=learnable and not self.global_reference,
        )
        self.gain_limit_dex = float(temperature_response_config.get('gain_limit_dex', 0.3))
        if not np.isfinite(self.gain_limit_dex) or self.gain_limit_dex <= 0:
            raise ValueError('gain_limit_dex must be finite and strictly positive')
        self.gain_prior_sigma_dex = float(
            temperature_response_config.get('gain_prior_sigma_dex', self.gain_limit_dex / 2)
        )
        if not np.isfinite(self.gain_prior_sigma_dex) or self.gain_prior_sigma_dex <= 0:
            raise ValueError('gain_prior_sigma_dex must be finite and strictly positive')
        self.common_gain_limit_dex = float(
            temperature_response_config.get('common_gain_limit_dex', 1.0)
        )
        if not np.isfinite(self.common_gain_limit_dex) or self.common_gain_limit_dex <= 0:
            raise ValueError('common_gain_limit_dex must be finite and strictly positive')
        self.common_gain_prior_sigma_dex = float(
            temperature_response_config.get(
                'common_gain_prior_sigma_dex', self.common_gain_limit_dex / 2
            )
        )
        if (
            not np.isfinite(self.common_gain_prior_sigma_dex)
            or self.common_gain_prior_sigma_dex <= 0
        ):
            raise ValueError(
                'common_gain_prior_sigma_dex must be finite and strictly positive'
            )
        self.common_gain_identifiability = (
            'fixed_global_reference' if self.global_reference else 'learned_relative_to_global_reference'
        )

        if (
            'reference_channel' in temperature_response_config
            and temperature_response_config['reference_channel'] is None
        ):
            raise ValueError('reference_channel must be non-null when supplied')
        reference_channel = temperature_response_config.get('reference_channel')
        gain_constraint = temperature_response_config.get('gain_constraint')
        if reference_channel is not None and 'gain_constraint' in temperature_response_config:
            raise ValueError('reference_channel and gain_constraint are mutually exclusive')
        if reference_channel is None and gain_constraint not in {None, 'zero_mean'}:
            raise ValueError('gain_constraint must be zero_mean when reference_channel is absent')
        if reference_channel is None:
            self.reference_channel = None
            self.reference_channel_index = None
            self.gain_identifiability = 'zero_mean_log_correction'
        else:
            reference_channel = str(reference_channel)
            matching_reference_channels = [
                channel for channel in artifact.channels
                if canonical_channel_id(channel)
                == canonical_channel_id(reference_channel)
            ]
            if len(matching_reference_channels) != 1:
                raise ValueError(
                    f'reference_channel {reference_channel!r} is not in response channels '
                    f'{artifact.channels}'
                )
            reference_channel = matching_reference_channels[0]
            self.reference_channel = reference_channel
            self.reference_channel_index = artifact.channels.index(reference_channel)
            self.gain_identifiability = 'fixed_reference_channel'

        model_length_unit_cm = temperature_response_config.get('model_length_unit_cm')
        if model_length_unit_cm is None:
            model_length_unit_cm = (
                float(temperature_response_config.get('Rs_per_ds', 1.0)) * R_SUN_CM
            )
        model_length_unit_cm = float(model_length_unit_cm)
        if not np.isfinite(model_length_unit_cm) or model_length_unit_cm <= 0:
            raise ValueError('model_length_unit_cm must be finite and strictly positive')
        self.register_buffer(
            'model_length_unit_cm', torch.tensor(model_length_unit_cm, dtype=torch.float64)
        )

        self.emission_measure_convention = artifact.emission_measure_convention
        if self.emission_measure_convention == 'ne_nh':
            artifact_ratio = artifact.provenance.get('hydrogen_to_electron_ratio')
            configured_ratio = temperature_response_config.get('hydrogen_to_electron_ratio')
            if configured_ratio is None:
                configured_ratio = artifact_ratio
            if configured_ratio is None:
                raise ValueError(
                    'responses using the ne_nh convention require a provenance-bound '
                    'hydrogen_to_electron_ratio'
                )
            hydrogen_to_electron_ratio = float(configured_ratio)
            if not 0 < hydrogen_to_electron_ratio <= 1:
                raise ValueError('hydrogen_to_electron_ratio must be in (0, 1]')
            if artifact_ratio is not None and not np.isclose(
                hydrogen_to_electron_ratio, float(artifact_ratio), rtol=1e-7, atol=0.0
            ):
                raise ValueError(
                    'configured hydrogen_to_electron_ratio does not match the response '
                    'artifact abundance provenance'
                )
        else:
            hydrogen_to_electron_ratio = 1.0
        self.register_buffer(
            'emission_measure_factor',
            torch.tensor(hydrogen_to_electron_ratio, dtype=torch.float32),
        )

        self.absorption_model = absorption_model
        self.channels = artifact.channels
        self.response_unit = artifact.response_unit
        self.response_provenance = dict(artifact.provenance)
        self.response_id = source_response_id
        self.prepared_response_id = artifact.response_id

    @property
    def effective_instrument_scaling(self):
        """Constrained per-channel base-10 gain correction."""
        return (
            self.instrument_scaling_center
            + self.common_gain_delta_dex
            + self.instrument_gain_delta_dex
        )

    @property
    def common_gain_delta_dex(self):
        """Bounded instrument-wide gain, fixed to zero for the global reference."""
        if self.global_reference:
            return self.common_instrument_scaling * 0.0
        return self.common_gain_limit_dex * torch.tanh(
            self.common_instrument_scaling
        )

    @property
    def instrument_gain_delta_dex(self):
        """Identifiable bounded correction to the nominal channel calibration.

        A configured reference channel has exactly zero correction.  Otherwise
        the corrections have exactly zero mean, separating relative channel
        calibration from the instrument-wide gain. The explicit global reference
        instrument, not this channel constraint, fixes the density/gain gauge.
        """
        bounded = self.gain_limit_dex * torch.tanh(self.instrument_scaling)
        if self.reference_channel_index is not None:
            mask = torch.ones_like(bounded)
            mask[self.reference_channel_index] = 0.0
            return bounded * mask

        centered = bounded - bounded.mean()
        # Centering values already in [-limit, limit] can double their range.
        # A common rescaling preserves exact zero mean and the requested bound.
        relative_max = centered.abs().amax() / self.gain_limit_dex
        return centered / relative_max.clamp_min(1.0)

    @property
    def instrument_gain(self):
        return torch.pow(10.0, self.effective_instrument_scaling)

    def calibration_regularization(self):
        channel_penalty = (
            self.instrument_gain_delta_dex / self.gain_prior_sigma_dex
        ).square().mean()
        common_penalty = (
            self.common_gain_delta_dex / self.common_gain_prior_sigma_dex
        ).square()
        return channel_penalty + common_penalty

    @staticmethod
    def _safe_pow10(exponent: torch.Tensor) -> torch.Tensor:
        finfo = torch.finfo(exponent.dtype)
        lower = float(np.log10(finfo.tiny))
        upper = float(np.log10(finfo.max)) - 1.0
        finite = torch.nan_to_num(exponent, nan=lower, posinf=upper, neginf=lower)
        powered = torch.pow(10.0, finite.clamp(lower, upper))
        return torch.where(finite < lower, torch.zeros_like(powered), powered)

    @staticmethod
    def _bracket(axis: torch.Tensor, values: torch.Tensor):
        upper = torch.searchsorted(axis, values.contiguous(), right=True)
        upper = upper.clamp(1, axis.numel() - 1)
        lower = upper - 1
        fraction = (values - axis[lower]) / (axis[upper] - axis[lower])
        return lower, upper, fraction

    def response_at(self, log_temperature: torch.Tensor, log_density: torch.Tensor) -> torch.Tensor:
        """Bilinear response lookup at the pointwise plasma state.

        The response is interpolated linearly in its value over
        ``(log10 T, log10 n_e)``. It is zero outside the tabulated temperature
        support; the density is clamped to its axis, where the level
        populations have reached their low/high-density limits.
        """
        temperature = torch.nan_to_num(
            log_temperature.squeeze(-1), nan=float('-inf'), posinf=float('inf'), neginf=float('-inf')
        )
        axis = self.response_log_T
        inside = (temperature >= axis[0]) & (temperature <= axis[-1])
        t_lower, t_upper, t_fraction = self._bracket(axis, temperature.clamp(axis[0], axis[-1]))
        t_fraction = t_fraction[..., None]

        table = self.temperature_response
        if self.log_density_axis.numel() <= 1:
            response = torch.lerp(table[0][t_lower], table[0][t_upper], t_fraction)
        else:
            density = torch.nan_to_num(log_density.squeeze(-1), nan=0.0).clamp(
                self.log_density_axis[0], self.log_density_axis[-1]
            )
            d_lower, d_upper, d_fraction = self._bracket(self.log_density_axis, density)
            low = torch.lerp(table[d_lower, t_lower], table[d_lower, t_upper], t_fraction)
            high = torch.lerp(table[d_upper, t_lower], table[d_upper, t_upper], t_fraction)
            response = torch.lerp(low, high, d_fraction[..., None])
        return response * inside[..., None].to(response.dtype)

    def emission_cutoff(self, log_temperature: torch.Tensor) -> torch.Tensor:
        """Transition-region cutoff applied to emission only (never to opacity)."""
        if self.temperature_cutoff is None:
            return torch.ones_like(log_temperature)
        temperature = self._safe_pow10(log_temperature)
        return 0.5 * (
            1.0 + torch.tanh(
                (temperature - self.temperature_cutoff_K) / self.temperature_cutoff_width_K
            )
        )

    def _emission_measure_histogram(self, log_temperature, emission_measure):
        """Deposit LOS emission measure onto the diagnostic temperature nodes."""
        axis = self.log_T
        temperature = torch.nan_to_num(log_temperature.squeeze(-1), nan=float(axis[0]))
        inside = (temperature >= axis[0]) & (temperature <= axis[-1])
        lower, upper, fraction = self._bracket(axis, temperature.clamp(axis[0], axis[-1]))
        measure = emission_measure * inside.to(emission_measure.dtype)
        histogram = measure.new_zeros(measure.shape[0], axis.numel())
        histogram.scatter_add_(1, lower, measure * (1.0 - fraction))
        histogram.scatter_add_(1, upper, measure * fraction)
        return histogram

    def forward(self, total_ne, mean_log_T, total_log_ne, z_vals: torch.Tensor,
                rays_d: torch.Tensor, query_points: torch.Tensor,
                diagnostics=None, ray_valid=None, **kwargs):
        r"""
        Render pointwise electron density and temperature into channel images.

        total_ne, total_log_ne, mean_log_T: (ray, sample, 1) plasma state
        z_vals: distance along the ray as measured from the observer
        """

        if z_vals.ndim != 2 or z_vals.shape != total_log_ne.shape[:2]:
            raise ValueError('z_vals must have shape (ray, sample) matching the model output')
        quadrature_z_vals = z_vals
        if ray_valid is not None:
            ray_valid = torch.as_tensor(
                ray_valid, dtype=torch.bool, device=z_vals.device
            ).reshape(-1)
            if ray_valid.shape != (z_vals.shape[0],):
                raise ValueError('ray_valid must contain one boolean per ray')
            # A shell miss deliberately has a false validity flag and zero
            # placeholder distances. Give only such rows a harmless increasing
            # grid so strict quadrature can run before the caller zero-masks all
            # products from the invalid ray.
            dummy = torch.linspace(
                0.0,
                1.0,
                z_vals.shape[-1],
                dtype=z_vals.dtype,
                device=z_vals.device,
            ).expand_as(z_vals)
            quadrature_z_vals = torch.where(ray_valid[:, None], z_vals, dummy)
        ray_norm = torch.linalg.norm(rays_d, dim=-1)
        node_dl_model = (
            torch_trapezoid_node_weights(quadrature_z_vals) * ray_norm[:, None]
        )
        interval_dl_model = (
            quadrature_z_vals[:, 1:] - quadrature_z_vals[:, :-1]
        ) * ray_norm[:, None]
        length_scale = self.model_length_unit_cm.to(device=z_vals.device, dtype=z_vals.dtype)
        node_dl_cm = node_dl_model * length_scale
        interval_dl_cm = interval_dl_model * length_scale

        # Squaring in log space keeps the float32 emission measure finite.
        local_emission_measure_density = self._safe_pow10(
            2.0 * total_log_ne.clamp(-30.0, 18.0)
        )
        response = self.response_at(mean_log_T, total_log_ne)
        local_emissivity = (
            response * local_emission_measure_density * self.emission_cutoff(mean_log_T)
        )
        local_emissivity = local_emissivity * self.emission_measure_factor
        local_emissivity = local_emissivity * self.instrument_gain.reshape(1, 1, -1)
        local_emissivity = torch.nan_to_num(local_emissivity, nan=0.0, posinf=0.0, neginf=0.0)

        absorption_state = None
        if self.absorption:
            absorption_state = self.absorption_model.opacity(
                total_ne=total_ne,
                total_log_ne=total_log_ne,
                mean_log_T=mean_log_T,
                total_hydrogen_density=kwargs.get('total_hydrogen_density'),
                **(
                    {'cool_hydrogen_density': kwargs['cool_hydrogen_density']}
                    if kwargs.get('cool_hydrogen_density') is not None else {}
                ),
            )
            alpha_cm_inverse = absorption_state['alpha_cm_inverse']
            if alpha_cm_inverse.shape[-1] == 1:
                alpha_cm_inverse = alpha_cm_inverse.expand(
                    *alpha_cm_inverse.shape[:-1], len(self.channels)
                )
            elif alpha_cm_inverse.shape[-1] != len(self.channels):
                raise ValueError(
                    'absorption opacity must provide one value or one value per response channel'
                )
        else:
            alpha_cm_inverse = torch.zeros(
                *total_ne.shape[:-1], len(self.channels),
                dtype=total_ne.dtype,
                device=total_ne.device,
            )

        interval_optical_depth = (
            0.5 * (alpha_cm_inverse[:, :-1] + alpha_cm_inverse[:, 1:])
            * interval_dl_cm[..., None]
        )
        optical_depth_to_node = torch.cat(
            [
                torch.zeros_like(interval_optical_depth[:, :1]),
                torch.cumsum(interval_optical_depth, dim=1),
            ],
            dim=1,
        )
        transmission_to_observer = torch.exp(-optical_depth_to_node.clamp(max=80.0))
        emerging_intensity = local_emissivity * transmission_to_observer
        integrated_intensity = (emerging_intensity * node_dl_cm[..., None]).sum(dim=1)

        contribution = emerging_intensity.mean(dim=-1) * node_dl_cm
        if self.absorption:
            # Absorbers emit nothing, so an emission-only importance weight never
            # refines them. Add the intensity each node removes from the ray,
            # transmission * (1 - exp(-alpha dl)) of the rendered intensity.
            removed_fraction = transmission_to_observer * (
                1.0 - torch.exp(-(alpha_cm_inverse * node_dl_cm[..., None]).clamp(max=80.0))
            )
            contribution = contribution + (
                removed_fraction * integrated_intensity[:, None, :]
            ).mean(dim=-1)
        contribution_sum = contribution.sum(dim=1, keepdim=True)
        path_weights = node_dl_cm / node_dl_cm.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(node_dl_cm.dtype).eps
        )
        weights = torch.where(
            contribution_sum > 0,
            contribution / contribution_sum.clamp_min(torch.finfo(contribution.dtype).eps),
            path_weights,
        )

        total_optical_depth = interval_optical_depth.sum(dim=1)
        mean_absorption = (1.0 - torch.exp(-total_optical_depth.clamp(max=80.0))).mean(dim=-1)

        distance = query_points[..., :3].pow(2).sum(-1).pow(0.5)
        local_emission_measure_density = local_emission_measure_density.squeeze(-1)

        # The training loop only consumes these products. Avoid returning large
        # per-ray/per-sample tensors that would otherwise extend their autograd
        # lifetimes. Validation/evaluation remains diagnostic by default, and a
        # direct caller can request diagnostics explicitly while training.
        output = {
            'image': integrated_intensity,
            'weights': weights,
            'mean_absorption': mean_absorption,
            'em': local_emission_measure_density,
            'distance': distance,
            'calibration_regularization': self.calibration_regularization(),
            'instrument_gain_delta_dex': self.instrument_gain_delta_dex,
            'common_gain_delta_dex': self.common_gain_delta_dex,
        }
        cool_density = None if absorption_state is None else absorption_state.get(
            'cool_hydrogen_density_cm3'
        )
        if cool_density is not None:
            # LOS hydrogen column of the separate cool absorber; the training
            # loop applies its sparsity prior to this product.
            output['cool_hydrogen_column_cm2'] = (cool_density.squeeze(-1) * node_dl_cm).sum(dim=1)
        if diagnostics is None:
            diagnostics = not self.training
        if not diagnostics:
            return output

        electron_density = torch.nan_to_num(
            total_ne, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        density_path_weight = electron_density * node_dl_cm[..., None]
        column_electron_density = density_path_weight.sum(dim=1)
        local_temperature = self._safe_pow10(mean_log_T)
        mean_temperature = (
            (local_temperature * density_path_weight).sum(dim=1)
            / column_electron_density.clamp_min(
                torch.finfo(column_electron_density.dtype).tiny
            )
        )
        mean_log_temperature = torch.log10(
            mean_temperature.clamp_min(torch.finfo(mean_temperature.dtype).tiny)
        )
        height_map = (weights * distance).sum(-1)
        emission_measure_along_ray = local_emission_measure_density * node_dl_cm
        emission_measure = emission_measure_along_ray.sum(dim=1)
        # With one temperature per point the DEM is the LOS distribution of
        # n_e^2 dl over temperature, deposited on the diagnostic nodes.
        emission_measure_per_temperature_bin = self._emission_measure_histogram(
            mean_log_T, emission_measure_along_ray
        )
        differential_emission_measure = (
            emission_measure_per_temperature_bin / self.temperature_bin_weights
        )

        output.update({
            'mean_T': mean_log_temperature,
            'mean_log_T': mean_log_temperature,
            'total_ne': column_electron_density,
            'height_map': height_map,
            'distance': distance,
            # Explicit physical diagnostics.
            'electron_density_cm3': electron_density,
            'column_electron_density_cm2': column_electron_density,
            'emission_measure_density_cm6': local_emission_measure_density,
            'emission_measure_cm5': emission_measure,
            'differential_emission_measure_cm5_per_dex': differential_emission_measure,
            'emission_measure_per_temperature_bin_cm5': emission_measure_per_temperature_bin,
            # Compatibility alias: dem is the LOS emission measure per bin.
            'dem': emission_measure_per_temperature_bin,
            'transmission': transmission_to_observer,
            'optical_depth': total_optical_depth,
            'z_vals': z_vals,
            'query_points': query_points,
        })
        if cool_density is not None:
            output['cool_hydrogen_density_cm3'] = cool_density
            output['cool_optical_depth'] = (
                0.5 * (
                    absorption_state['cool_alpha_cm_inverse'][:, :-1]
                    + absorption_state['cool_alpha_cm_inverse'][:, 1:]
                ) * interval_dl_cm[..., None]
            ).sum(dim=1)
        if absorption_state is not None and 'total_hydrogen_density_cm3' in absorption_state:
            hydrogen_density = absorption_state['total_hydrogen_density_cm3']
            output['total_hydrogen_density_cm3'] = hydrogen_density
            output['column_hydrogen_density_cm2'] = (
                hydrogen_density * node_dl_cm[..., None]
            ).sum(dim=1)
            species_density = absorption_state['absorber_species_density_cm3']
            output['absorber_species_density_cm3'] = species_density
            output['absorber_species_column_density_cm2'] = (
                species_density * node_dl_cm[..., None]
            ).sum(dim=1)
        return output


def init_absorption_model(absorption_config, *, instrument_key=None, channels=None):
    absorption_config = copy.deepcopy(absorption_config)
    absorption_type = absorption_config.pop('type', None)
    if absorption_type == 'photoionization':
        if instrument_key is None or channels is None:
            raise ValueError(
                'photoionization absorption requires an instrument key and ordered channels'
            )
        return PhotoionizationOpacity(
            instrument_key=instrument_key,
            channels=channels,
            **absorption_config,
        )
    if absorption_type == 'constant':
        return ConstantAbsorptionModel(**absorption_config)
    elif absorption_type == 'learned':
        return AbsorptionModel(**absorption_config)
    elif absorption_type is None:
        return None
    else:
        raise NotImplementedError(f"Absorption type {absorption_type} not implemented.")
