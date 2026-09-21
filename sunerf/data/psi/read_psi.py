import os
import re

import numpy as np
from h5py import File



def _frame_id(path):
    match = re.search(r"(\d+)$", os.path.splitext(os.path.basename(path))[0])
    if match is None:
        raise ValueError(f"Could not infer a numeric PSI frame id from {path}")
    return int(match.group(1))


def read_PSI_fields(
    rho_hdf5,
    T_hdf5,
    *,
    density_unit_scale_cm3,
    temperature_unit_scale_K,
    reference_frame_id,
    min_radius=1.0,
    max_radius=2.6,
):
    """Read PSI fields with explicit source-unit and temporal conversions.

    ``density_unit_scale_cm3`` converts one source density unit to ``cm^-3``;
    ``temperature_unit_scale_K`` converts one source temperature unit to K.
    The returned time is a frame offset relative to ``reference_frame_id``.
    """
    for name, value in (
        ("density_unit_scale_cm3", density_unit_scale_cm3),
        ("temperature_unit_scale_K", temperature_unit_scale_K),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive scalar")
    if isinstance(reference_frame_id, (bool, np.bool_)) or not isinstance(
        reference_frame_id, (int, np.integer)
    ):
        raise TypeError("reference_frame_id must be an integer")
    print(f'Reading {rho_hdf5} and {T_hdf5}')
    rho_frame = _frame_id(rho_hdf5)
    temperature_frame = _frame_id(T_hdf5)
    if rho_frame != temperature_frame:
        raise ValueError(
            f'PSI density/temperature frame mismatch: {rho_frame} != {temperature_frame}'
        )

    with File(rho_hdf5, 'r') as h5file:
        radius = np.asarray(h5file['dim1'], dtype=np.float32)
        latitude = np.asarray(h5file['dim2'], dtype=np.float32) - np.pi / 2
        longitude = np.asarray(h5file['dim3'], dtype=np.float32)
        density = np.asarray(h5file['Data'], dtype=np.float32).T
    with File(T_hdf5, 'r') as h5file:
        temperature = np.asarray(h5file['Data'], dtype=np.float32).T
        temperature_axes = tuple(
            np.asarray(h5file[key], dtype=np.float32)
            for key in ('dim1', 'dim2', 'dim3')
            if key in h5file
        )

    expected_shape = (radius.size, latitude.size, longitude.size)
    if density.shape != expected_shape or temperature.shape != expected_shape:
        raise ValueError(
            'PSI field shape does not match its coordinate axes: '
            f'expected {expected_shape}, density={density.shape}, temperature={temperature.shape}'
        )
    if temperature_axes:
        reference_axes = (radius, latitude + np.pi / 2, longitude)
        if len(temperature_axes) != 3 or any(
            source.shape != reference.shape or not np.allclose(source, reference)
            for source, reference in zip(temperature_axes, reference_axes)
        ):
            raise ValueError('PSI density and temperature coordinate grids differ')

    print(
        f'r_range: {(radius.min(), radius.max())}, '
        f'th_range: {(latitude.min(), latitude.max())}, '
        f'phi_range: {(longitude.min(), longitude.max())}'
    )
    # Keep the bracketing node on either side so interpolation covers the full
    # closed interval [min_radius, max_radius]; a strict crop leaves the layer
    # between the requested boundary and the first retained node empty.
    upper_nodes = np.flatnonzero(radius >= max_radius)
    upper_limit = radius[upper_nodes[0]] if upper_nodes.size else radius[-1]
    radial_mask = radius <= upper_limit
    if min_radius is not None:
        lower_nodes = np.flatnonzero(radius <= min_radius)
        lower_limit = radius[lower_nodes[-1]] if lower_nodes.size else radius[0]
        radial_mask &= radius >= lower_limit
    radius = radius[radial_mask]
    density = density[radial_mask]
    temperature = temperature[radial_mask]
    if radius.size == 0:
        raise ValueError('PSI radial selection is empty')

    density[density <= 0] = np.nan
    temperature[temperature <= 0] = np.nan
    density *= float(density_unit_scale_cm3)
    temperature *= float(temperature_unit_scale_K)

    return {
        'rho': density,
        'T': temperature,
        'time': rho_frame - int(reference_frame_id),
        'frame_id': rho_frame,
        'radius': radius,
        'latitude': latitude,
        'longitude': longitude,
    }
