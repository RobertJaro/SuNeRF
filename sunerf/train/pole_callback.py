"""Validation view of the reconstruction from above the solar north pole.

The EUV validation plots only show held-out frames of the observing
instruments, which all lie close to the ecliptic. This callback renders the
same model from a virtual observer at Carrington latitude +90 deg, at the time
and distance of one validation frame, and logs the predicted channel images
and plasma diagnostics next to the ``euv_tomography`` plots.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import wandb
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from matplotlib import pyplot as plt
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header

from sunerf.configuration import canonical_channel_id
from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.train.callback import BaseCallback
from sunerf.train.euv_callback import (
    _append_colorbar,
    _channel_cmap,
    _emission_measure_mask,
    _linear_norm,
    _positive_log_norm,
)

# Rendered image size; the validation frame's field of view is kept and its
# pixel scale coarsened accordingly.
NORTH_POLE_VIEW_PIXELS = 256
# Heliographic latitudes drawn on the photosphere for orientation.
_LATITUDE_CIRCLES_DEG = (60.0, 70.0, 80.0)

_DIAGNOSTICS = (
    ("mean_T", "Mean log$_{10}$(T / K)", "plasma", "linear"),
    ("emission_measure_cm5", "Emission measure [cm$^{-5}$]", "magma", "log"),
    ("column_electron_density_cm2", "Column $N_e$ [cm$^{-2}$]", "viridis", "log"),
)


def _quantity(value, unit):
    return value.to(unit) if isinstance(value, u.Quantity) else float(value) * unit


def north_pole_view_geometry(wcs, image_shape, observer, Rs_per_ds,
                             pixels=NORTH_POLE_VIEW_PIXELS):
    """Rays of a disk-centered view from Carrington latitude +90 deg.

    ``wcs`` and ``image_shape`` belong to a validation frame and set the field
    of view; ``observer`` (the frame's observer record) sets the distance and
    time. The rays are built by the same loader as the training rays.
    """
    image_shape = tuple(int(value) for value in image_shape[:2])
    scales = proj_plane_pixel_scales(wcs) * u.Unit(wcs.wcs.cunit[0])
    field_of_view = [
        (scale * size).to_value(u.arcsec) for scale, size in zip(scales[::-1], image_shape)
    ]
    shape = tuple(min(size, int(pixels)) for size in image_shape)
    scale = [fov / size for fov, size in zip(field_of_view, shape)]

    time = observer['time']
    distance = _quantity(observer['radius'], u.solRad)
    # At the pole, the image up direction is the limit along the observer
    # longitude; lon -90 puts Carrington longitude 0 right and 90 up.
    pole = SkyCoord(
        lon=-90 * u.deg, lat=90 * u.deg, radius=distance,
        frame=frames.HeliographicCarrington(observer='self', obstime=time),
    )
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec,
        frame=frames.Helioprojective(observer=pole, obstime=time),
    )
    header = make_fitswcs_header(
        shape, reference,
        scale=[scale[1], scale[0]] * u.arcsec / u.pix,
        projection_code='ARC',
    )
    geometry = MapDataLoader(Rs_per_ds=Rs_per_ds, reference_frame='carrington').load(
        Map(np.zeros(shape, dtype=np.float32), header)
    )
    return {
        'rays': geometry['rays'],
        'time': time,
        'distance_rsun': float(distance.to_value(u.solRad)),
        'scale_arcsec': scale,
        'image_shape': shape,
    }


def latitude_circle_radius_pixels(latitude_deg, distance_rsun, scale_arcsec):
    """Image radius of a photospheric latitude circle seen from the pole.

    In the azimuthal-equidistant (ARC) projection the image radius is the
    angle from the optical axis.
    """
    latitude = math.radians(latitude_deg)
    angle = math.atan2(math.cos(latitude), distance_rsun - math.sin(latitude))
    return math.degrees(angle) * 3600.0 / float(scale_arcsec)


class NorthPoleViewCallback(BaseCallback):
    """Render and log the reconstruction seen from above the north pole."""

    def __init__(self, ds_key, instrument_key, channel_metadata, selected_channels,
                 geometry, normalized_time, *, every_n_validations=1,
                 figure_dpi=150, batch_size=512):
        super().__init__(ds_key)
        self.instrument_key = str(instrument_key)
        self.every_n_validations = int(every_n_validations)
        self.figure_dpi = int(figure_dpi)
        self.batch_size = int(batch_size)
        self._validation_count = 0

        channel_metadata = tuple(dict(channel) for channel in channel_metadata)
        indices = []
        selected_metadata = []
        for requested in selected_channels:
            matches = [
                (index, channel)
                for index, channel in enumerate(channel_metadata)
                if canonical_channel_id(channel['id']) == canonical_channel_id(requested)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"North-pole channel {requested!r} must match exactly one artifact "
                    f"channel for {self.instrument_key!r}."
                )
            indices.append(matches[0][0])
            selected_metadata.append(matches[0][1])
        self.channel_indices = tuple(indices)
        self.channel_metadata = tuple(selected_metadata)

        self.image_shape = tuple(geometry['image_shape'])
        self.distance_rsun = float(geometry['distance_rsun'])
        self.scale_arcsec = float(geometry['scale_arcsec'][0])
        self.time = geometry['time']
        self.normalized_time = float(normalized_time)
        self.rays = torch.from_numpy(
            np.ascontiguousarray(geometry['rays'], dtype=np.float32)
        ).reshape(-1, 2, 3)

    def setup(self, trainer, pl_module, stage):
        # Renders its own rays; no validation tensors are consumed.
        return

    @property
    def state_key(self):
        return f"{type(self).__qualname__}:{self.ds_key}"

    def state_dict(self):
        return {"validation_count": self._validation_count}

    def load_state_dict(self, state_dict):
        self._validation_count = int(state_dict.get("validation_count", 0))

    @torch.no_grad()
    def render(self, pl_module):
        """Images in loss space and diagnostics, reshaped to the view."""
        rendering = pl_module.rendering
        scaling = pl_module.image_scaling[self.instrument_key]
        keys = ('image',) + tuple(field for field, *_ in _DIAGNOSTICS)
        collected = {key: [] for key in keys + ('ray_valid',)}
        was_training = rendering.training
        rendering.eval()
        try:
            for rays in torch.split(self.rays, self.batch_size):
                rays = rays.to(pl_module.device)
                time = torch.full(
                    (rays.shape[0], 1), self.normalized_time,
                    dtype=rays.dtype, device=rays.device,
                )
                batch = {self.ds_key: {
                    'rays': rays, 'time': time, 'instrument': self.instrument_key,
                }}
                out = rendering(batch, shuffle=False, diagnostics=True)
                out = out['model_out'][self.ds_key]
                collected['image'].append(scaling(out['image']).cpu())
                for key in keys[1:]:
                    collected[key].append(out[key].reshape(rays.shape[0], -1).cpu())
                valid = out.get('ray_valid')
                if valid is None:
                    valid = torch.ones(rays.shape[0], dtype=torch.bool)
                collected['ray_valid'].append(valid.reshape(-1).cpu())
        finally:
            rendering.train(was_training)
        outputs = {}
        for key, values in collected.items():
            array = torch.cat(values).numpy()
            outputs[key] = array.reshape(*self.image_shape, *array.shape[1:])
        return outputs

    def _draw_latitude_circles(self, axis):
        center_y = (self.image_shape[0] - 1) / 2
        center_x = (self.image_shape[1] - 1) / 2
        for latitude in _LATITUDE_CIRCLES_DEG:
            radius = latitude_circle_radius_pixels(
                latitude, self.distance_rsun, self.scale_arcsec
            )
            axis.add_patch(plt.Circle(
                (center_x, center_y), radius, fill=False, color='white',
                linestyle=':', linewidth=0.6, alpha=0.7,
            ))

    def plot(self, outputs):
        valid = outputs['ray_valid'].astype(bool)
        emission_mask = _emission_measure_mask(
            np.squeeze(outputs['emission_measure_cm5'], -1), valid
        )
        n_columns = max(len(self.channel_indices), len(_DIAGNOSTICS))
        figure, axes = plt.subplots(
            2, n_columns, figsize=(3.2 * n_columns, 6.2),
            dpi=self.figure_dpi, squeeze=False,
        )
        for axis in axes.ravel():
            axis.set_axis_off()
        for axis, channel_index, metadata in zip(
            axes[0], self.channel_indices, self.channel_metadata
        ):
            values = np.where(valid, outputs['image'][..., channel_index], np.nan)
            image = axis.imshow(
                values, origin='lower', cmap=_channel_cmap(metadata),
                norm=_linear_norm(values),
            )
            axis.set_title(f"{metadata['id']} — prediction")
            self._draw_latitude_circles(axis)
            _append_colorbar(figure, axis, image)
        for axis, (field, title, cmap, norm_kind) in zip(axes[1], _DIAGNOSTICS):
            mask = emission_mask if field == 'mean_T' else valid
            values = np.where(mask, np.squeeze(outputs[field], -1), np.nan)
            norm = _positive_log_norm(values) if norm_kind == 'log' else _linear_norm(values)
            image = axis.imshow(values, origin='lower', cmap=cmap, norm=norm)
            axis.set_title(title)
            self._draw_latitude_circles(axis)
            _append_colorbar(figure, axis, image)
        figure.suptitle(
            f"{self.instrument_key} seen from Carrington latitude +90° at "
            f"{self.time:%Y-%m-%d %H:%M} — Carrington longitude 0° to the right, "
            f"90° up; dotted: {', '.join(f'{lat:g}' for lat in _LATITUDE_CIRCLES_DEG)}° "
            "latitude",
            fontsize=9,
        )
        figure.tight_layout()
        return figure

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self._validation_count += 1
        if self._validation_count % self.every_n_validations:
            return
        figure = self.plot(self.render(pl_module))
        wandb.log({f"euv_tomography.north_pole.{self.ds_key}": wandb.Image(figure)})
        plt.close(figure)
