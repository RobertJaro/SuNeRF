from typing import Tuple

import numpy as np
from astropy import units as u


def hpc_angular_separation(Tx, Ty, center_Tx=0 * u.rad, center_Ty=0 * u.rad):
    """Return the exact great-circle separation between HPC directions.

    Helioprojective longitude and latitude are spherical angles.  Treating them
    as Cartesian coordinates with ``hypot(Tx, Ty)`` is only a small-angle
    approximation and noticeably biases PUNCH's wide field of view.
    """
    tx = u.Quantity(Tx).to_value(u.rad)
    ty = u.Quantity(Ty).to_value(u.rad)
    center_tx = u.Quantity(center_Tx).to_value(u.rad)
    center_ty = u.Quantity(center_Ty).to_value(u.rad)

    delta_tx = tx - center_tx
    delta_ty = ty - center_ty
    haversine = (
        np.sin(0.5 * delta_ty) ** 2
        + np.cos(ty) * np.cos(center_ty) * np.sin(0.5 * delta_tx) ** 2
    )
    haversine = np.clip(haversine, 0.0, 1.0)
    separation = 2.0 * np.arctan2(np.sqrt(haversine), np.sqrt(1.0 - haversine))
    return separation * u.rad


def hpc_impact_parameter(Tx, Ty, observer_distance):
    """Return the Sun-centered impact parameter of each HPC line of sight."""
    separation = hpc_angular_separation(Tx, Ty).to_value(u.rad)
    return u.Quantity(observer_distance) * np.sin(separation)


def get_rays(Tx, Ty, c2w: np.array) -> Tuple[np.array, np.array]:
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """
    Tx = Tx.to_value(u.rad)
    Ty = Ty.to_value(u.rad)
    # Tx and Ty are helioprojective longitude and latitude, respectively.  They
    # are spherical angles rather than independent tangent-plane coordinates.
    # The optical axis points along negative camera z.
    cos_ty = np.cos(Ty)
    directions = np.stack([
        cos_ty * np.sin(Tx),
        np.sin(Ty),
        -cos_ty * np.cos(Tx),
    ], axis=-1).astype(np.float32, copy=False)

    # Apply camera pose to directions
    rays_d = np.einsum('ij,...j->...i', c2w[:3, :3], directions)

    # Guard against small numerical deviations from an orthonormal camera pose.
    rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # Origin is same for all directions (the optical center)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape).copy()
    return rays_o, rays_d
