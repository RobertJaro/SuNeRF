"""Grid-backed plasma fields for simulation-driven synthetic observations.

The reconstruction models in :mod:`sunerf.model.model` learn continuous fields
from images.  Synthetic observations already have a density and temperature
field, so fitting another neural network before ray tracing only introduces an
uncontrolled approximation.  This module exposes a simulation grid through the
same model interface used by the regular SuNeRF renderers.
"""

from __future__ import annotations

from itertools import product

import math

import torch
from torch import nn

from sunerf.train.coordinate_transformation import cartesian_to_spherical


class SphericalGridPlasmaModel(nn.Module):
    """Interpolate density and temperature on a spherical, time-dependent grid.

    Parameters
    ----------
    log_density, log_temperature
        Arrays with shape ``(time, radius, latitude, longitude)``. Values are
        base-10 logarithms. Non-finite grid cells are ignored and the remaining
        corner weights are renormalized during interpolation.
    time, radius, latitude, longitude
        Strictly increasing grid coordinates. Longitude must contain unique
        samples from one period; interpolation across its seam is periodic.
    log_T
        Base-10 temperature nodes of the renderer's diagnostic LOS
        emission-measure histogram; they do not enter image formation.

    Notes
    -----
    The field returns the interpolated pointwise ``(log10 n_e, log10 T)``.
    ``PlasmaRadiativeTransfer`` evaluates the channel response at exactly that
    state, so no temperature binning is applied here.
    """

    def __init__(
        self,
        *,
        log_density,
        log_temperature,
        time,
        radius,
        latitude,
        longitude,
        log_T,
        longitude_period=2 * torch.pi,
        fill_log_density=-30.0,
        fill_log_temperature=6.0,
        clamp_time=True,
    ):
        super().__init__()

        values = torch.stack(
            [torch.as_tensor(log_density), torch.as_tensor(log_temperature)],
            dim=-1,
        ).to(dtype=torch.float32)
        axes = {
            "time": torch.as_tensor(time, dtype=torch.float32),
            "radius": torch.as_tensor(radius, dtype=torch.float32),
            "latitude": torch.as_tensor(latitude, dtype=torch.float32),
            "longitude": torch.as_tensor(longitude, dtype=torch.float32),
        }
        expected_shape = tuple(axis.numel() for axis in axes.values())
        if values.shape != (*expected_shape, 2):
            raise ValueError(
                "log_density and log_temperature must have shape "
                f"(time, radius, latitude, longitude)={expected_shape}; "
                f"received {tuple(values.shape[:-1])}"
            )
        for name, axis in axes.items():
            if axis.ndim != 1 or axis.numel() == 0:
                raise ValueError(f"{name} must be a non-empty one-dimensional axis")
            if not torch.isfinite(axis).all():
                raise ValueError(f"{name} contains non-finite coordinates")
            if axis.numel() > 1 and not torch.all(axis[1:] > axis[:-1]):
                raise ValueError(f"{name} must be strictly increasing")

        period = float(longitude_period)
        if period <= 0:
            raise ValueError("longitude_period must be positive")
        if axes["longitude"].numel() > 1:
            longitude_span = float(axes["longitude"][-1] - axes["longitude"][0])
            if longitude_span >= period:
                raise ValueError(
                    "longitude must contain unique samples from one period; "
                    "drop a duplicated periodic endpoint"
                )

        self.register_buffer("values", values.contiguous())
        for name, axis in axes.items():
            self.register_buffer(name, axis.contiguous())
        self.register_buffer("log_T", torch.as_tensor(log_T, dtype=torch.float32))
        if self.log_T.ndim != 1 or self.log_T.numel() == 0:
            raise ValueError("log_T must be a non-empty one-dimensional axis")
        if self.log_T.numel() > 1 and not torch.all(self.log_T[1:] > self.log_T[:-1]):
            raise ValueError("log_T must be strictly increasing")

        self.longitude_period = period
        self.fill_log_density = float(fill_log_density)
        self.fill_log_temperature = float(fill_log_temperature)
        if not math.isfinite(self.fill_log_density) or self.fill_log_density > -20.0:
            raise ValueError(
                "fill_log_density must be finite and <= -20 so invalid grid cells "
                "cannot create observable EUV emission"
            )
        if not math.isfinite(self.fill_log_temperature):
            raise ValueError("fill_log_temperature must be finite")
        self.clamp_time = bool(clamp_time)

    @staticmethod
    def _linear_bracket(axis, coordinates):
        """Return lower/upper indices, upper weight, and in-bounds mask."""
        if axis.numel() == 1:
            index = torch.zeros_like(coordinates, dtype=torch.long)
            valid = torch.isclose(coordinates, axis[0], rtol=0.0, atol=1e-6)
            return index, index, torch.zeros_like(coordinates), valid

        upper = torch.searchsorted(axis, coordinates.contiguous(), right=True)
        upper = upper.clamp(1, axis.numel() - 1)
        lower = upper - 1
        lower_value = axis[lower]
        upper_value = axis[upper]
        weight = (coordinates - lower_value) / (upper_value - lower_value)
        weight = weight.clamp(0.0, 1.0)
        valid = (coordinates >= axis[0]) & (coordinates <= axis[-1])
        return lower, upper, weight, valid

    def _periodic_bracket(self, coordinates):
        axis = self.longitude
        if axis.numel() == 1:
            index = torch.zeros_like(coordinates, dtype=torch.long)
            return index, index, torch.zeros_like(coordinates)

        wrapped = torch.remainder(coordinates - axis[0], self.longitude_period) + axis[0]
        upper_unwrapped = torch.searchsorted(axis, wrapped.contiguous(), right=True)
        seam = upper_unwrapped == axis.numel()
        lower = (upper_unwrapped - 1).clamp(min=0)
        upper = torch.where(seam, torch.zeros_like(upper_unwrapped), upper_unwrapped)

        lower_value = axis[lower]
        upper_value = torch.where(seam, axis[0] + self.longitude_period, axis[upper])
        weight = (wrapped - lower_value) / (upper_value - lower_value)
        return lower, upper, weight.clamp(0.0, 1.0)

    def _interpolate_fields(self, coordinates):
        original_shape = coordinates.shape[:-1]
        flat_coordinates = coordinates.reshape(-1, 4)
        spherical = cartesian_to_spherical(flat_coordinates[:, :3], torch)

        time_coordinates = flat_coordinates[:, 3]
        if self.clamp_time:
            time_coordinates = time_coordinates.clamp(self.time[0], self.time[-1])
        t0, t1, wt, valid_time = self._linear_bracket(self.time, time_coordinates)
        r0, r1, wr, valid_radius = self._linear_bracket(self.radius, spherical[:, 0])
        a0, a1, wa, valid_latitude = self._linear_bracket(self.latitude, spherical[:, 1])
        l0, l1, wl = self._periodic_bracket(spherical[:, 2])

        if self.clamp_time:
            valid_time = torch.ones_like(valid_time)
        valid_query = valid_time & valid_radius & valid_latitude

        indices = ((t0, t1), (r0, r1), (a0, a1), (l0, l1))
        weights = ((1.0 - wt, wt), (1.0 - wr, wr), (1.0 - wa, wa), (1.0 - wl, wl))
        nt, nr, na, nl, n_fields = self.values.shape
        flat_values = self.values.reshape(nt * nr * na * nl, n_fields)

        numerator = torch.zeros(
            (flat_coordinates.shape[0], n_fields),
            dtype=self.values.dtype,
            device=flat_coordinates.device,
        )
        denominator = torch.zeros(
            (flat_coordinates.shape[0], 1),
            dtype=self.values.dtype,
            device=flat_coordinates.device,
        )
        for corner in product((0, 1), repeat=4):
            ti, ri, ai, li = (indices[dimension][side] for dimension, side in enumerate(corner))
            corner_weight = torch.ones_like(wt)
            for dimension, side in enumerate(corner):
                corner_weight = corner_weight * weights[dimension][side]
            flat_index = ((ti * nr + ri) * na + ai) * nl + li
            corner_values = flat_values[flat_index]
            # Density and temperature form one physical sample.  A partially
            # valid corner must not mix a density from one set of cells with a
            # temperature interpolated from another set (or the fill value).
            finite = torch.isfinite(corner_values).all(dim=-1, keepdim=True)
            finite_weight = corner_weight[:, None] * finite
            numerator = numerator + torch.nan_to_num(corner_values) * finite_weight
            denominator = denominator + finite_weight

        fill = torch.tensor(
            [self.fill_log_density, self.fill_log_temperature],
            dtype=self.values.dtype,
            device=flat_coordinates.device,
        )
        interpolated = torch.where(
            denominator > 0,
            numerator / denominator.clamp_min(torch.finfo(denominator.dtype).eps),
            fill,
        )
        interpolated = torch.where(valid_query[:, None], interpolated, fill)
        return interpolated.reshape(*original_shape, n_fields)

    def forward(self, coordinates):
        if coordinates.shape[-1] != 4:
            raise ValueError("coordinates must have final dimension (x, y, z, time)")

        fields = self._interpolate_fields(coordinates)
        total_log_ne = fields[..., 0:1]
        mean_log_T = fields[..., 1:2]
        return {
            "mean_log_T": mean_log_T,
            "total_ne": torch.pow(10.0, total_log_ne),
            "total_log_ne": total_log_ne,
        }
