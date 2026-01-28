import numpy as np
import torch

# Transformation from spherical to cartesian coordinates
trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.sin(th), 0, -np.cos(th), 0],
    [0, 1, 0, 0],
    [np.cos(th), 0, np.sin(th), 0],
    [0, 0, 0, 1]]).float()

trans_shift = lambda tx, ty, tz: torch.Tensor([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]]).float()

trans_unit = torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(longitude, latitude, r):
    # position in world
    p = r * np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(latitude) * np.sin(longitude),
        np.sin(latitude)
    ], dtype=np.float32)

    f = -p / (np.linalg.norm(p) + 1e-8)              # forward to origin
    up0 = np.array([0, 0, 1], dtype=np.float32)

    # handle near-pole degeneracy (optional)
    if abs(np.dot(f, up0)) > 0.999:
        up0 = np.array([0, 1, 0], dtype=np.float32)

    rvec = np.cross(up0, f)
    rvec /= (np.linalg.norm(rvec) + 1e-8)

    uvec = np.cross(f, rvec)

    # camera looks along -z_cam => +z_cam = -f
    R = np.stack([rvec, uvec, -f], axis=1)          # columns are camera axes in world

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3,  3] = p
    return c2w


# def pose_spherical(theta, phi, radius, shift=None):
#     """_summary_
#
#     Args:
#       theta: angle of position [rad]
#       phi: angle of position [rad]
#       radius: radius of position [pix]
#
#     Returns:
#         c2w: matrix for coordinate transformation
#     """
#     c2w = trans_unit
#     c2w = trans_t(radius) @ c2w
#     c2w = rot_phi(phi) @ c2w
#     c2w = rot_theta(theta) @ c2w
#     c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
#     if shift is not None:
#         c2w = trans_shift(*shift) @ c2w
#     return c2w


def spherical_to_cartesian(v, f=np):
    sin = f.sin
    cos = f.cos
    r, t, p = v[..., 0], v[..., 1], v[..., 2]
    x = r * cos(t) * cos(p)
    y = r * cos(t) * sin(p)
    z = r * sin(t)
    return f.stack([x, y, z], -1)

def cartesian_to_spherical(v, f):
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    eps = 1e-6
    r = f.sqrt(x*x + y*y + z*z + eps*eps)

    lon = f.atan2(y, x)

    if f is torch:
        u = (z / r).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        lat = torch.asin(u)   # latitude
        return torch.stack([r, lat, lon], dim=-1)
    elif f is np:
        u = np.clip(z / r, -1.0 + 1e-6, 1.0 - 1e-6)
        lat = np.arcsin(u)
        return np.stack([r, lat, lon], axis=-1)
    else:
        raise ValueError("Unsupported math module")



def carrington_rotation_rate(f=torch):
    """
    Get the Carrington rotation angular velocity.

    Parameters
    ----------
    f : module, optional
        Math module to use for calculations (default: torch; can also use numpy).

    Returns
    -------
    omega_carrington : float
        Carrington angular velocity (rad/s).
    """
    omega_carrington = 2 * f.pi / (25.38 * 24 * 3600)  # rad/s
    return omega_carrington

def differential_rotation_rate(latitude, f=torch):
    """
    Compute the solar differential rotation angular velocity at a given latitude.

    Parameters
    ----------
    latitude : array-like or tensor
        Heliographic latitude(s) in radians (positive northward).
    f : module, optional
        Math module to use for calculations (default: torch; can also use numpy).

    Returns
    -------
    omega : array-like or tensor
        Angular velocity (rad/s) at the given latitude(s).
    """
    A = 14.713 * (f.pi / 180) / 86400  # rad/s
    B = -2.396 * (f.pi / 180) / 86400  # rad/s
    C = -1.787 * (f.pi / 180) / 86400  # rad/s

    sin_lat = f.sin(latitude)
    omega = A + B * sin_lat**2 + C * sin_lat**4
    return omega  # in rad/s


def to_carrington_rotation_frame(coords, seconds_per_dt):
    time = coords[..., 3:4]  # 3 = reference time
    spherical_coords = cartesian_to_spherical(coords[..., :3], torch)
    lat = spherical_coords[..., 1:2]
    lon = spherical_coords[..., 2:3]
    omega_carrington = carrington_rotation_rate()
    lon_shift = omega_carrington * time * seconds_per_dt
    spherical_coords_shift = torch.cat([spherical_coords[..., 0:1], spherical_coords[..., 1:2],
                                        lon + lon_shift], dim=-1)
    cartesian_coords_shift = spherical_to_cartesian(spherical_coords_shift, torch)
    coords_shift = torch.cat([cartesian_coords_shift, time], dim=-1)
    return coords_shift

def to_differential_rotation_frame(coords, seconds_per_dt):
    time = coords[..., 3:4]  # 3 = reference time
    spherical_coords = cartesian_to_spherical(coords[..., :3], torch)
    lat = spherical_coords[..., 1:2]
    lon = spherical_coords[..., 2:3]
    omega_differential = differential_rotation_rate(lat, torch)
    lon_shift = omega_differential * time * seconds_per_dt
    spherical_coords_shift = torch.cat([spherical_coords[..., 0:1], spherical_coords[..., 1:2],
                                        lon + lon_shift], dim=-1)
    cartesian_coords_shift = spherical_to_cartesian(spherical_coords_shift, torch)
    coords_shift = torch.cat([cartesian_coords_shift, time], dim=-1)
    return coords_shift