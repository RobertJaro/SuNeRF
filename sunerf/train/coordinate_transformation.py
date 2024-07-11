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
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
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


def pose_spherical(theta, phi, radius, shift=None):
    """_summary_

    Args:
      theta: angle of position [rad]
      phi: angle of position [rad]
      radius: radius of position [pix]

    Returns:
        c2w: matrix for coordinate transformation
    """
    c2w = trans_unit
    c2w = trans_t(radius) @ c2w
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    if shift is not None:
        c2w = trans_shift(*shift) @ c2w
    return c2w


def spherical_to_cartesian(r, lat, lon):
    return np.array([r * np.cos(lat) * np.cos(lon),
                     r * np.cos(lat) * np.sin(lon),
                     r * np.sin(lat)])
