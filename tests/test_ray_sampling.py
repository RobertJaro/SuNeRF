import numpy as np
from astropy import units as u

from sunerf.data.ray_sampling import (
    get_rays,
    hpc_angular_separation,
    hpc_impact_parameter,
)


def test_hpc_angles_use_spherical_camera_geometry():
    tx = np.array([[0.0, 30.0], [-25.0, 15.0]]) * u.deg
    ty = np.array([[0.0, 20.0], [12.0, -18.0]]) * u.deg
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, 3] = [3.0, -2.0, 1.0]

    rays_o, rays_d = get_rays(tx, ty, camera_to_world)

    tx_rad = tx.to_value(u.rad)
    ty_rad = ty.to_value(u.rad)
    expected = np.stack([
        np.cos(ty_rad) * np.sin(tx_rad),
        np.sin(ty_rad),
        -np.cos(ty_rad) * np.cos(tx_rad),
    ], axis=-1)

    np.testing.assert_allclose(rays_d, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(rays_d, axis=-1), 1.0, atol=1e-7)
    np.testing.assert_allclose(
        rays_o, np.broadcast_to(camera_to_world[:3, 3], rays_o.shape)
    )


def test_wide_field_separation_and_impact_parameter_are_spherical():
    tx = 30 * u.deg
    ty = 20 * u.deg
    distance = 216 * u.R_sun

    separation = hpc_angular_separation(tx, ty)
    expected_separation = np.arccos(
        np.cos(tx.to_value(u.rad)) * np.cos(ty.to_value(u.rad))
    )
    expected_impact = distance * np.sin(expected_separation)

    np.testing.assert_allclose(
        separation.to_value(u.rad), expected_separation, rtol=1e-13
    )
    np.testing.assert_allclose(
        hpc_impact_parameter(tx, ty, distance).to_value(u.R_sun),
        expected_impact.to_value(u.R_sun),
        rtol=1e-13,
    )
