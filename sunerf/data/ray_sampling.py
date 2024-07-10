from typing import Tuple

import numpy as np
from astropy import units as u


def get_rays(img_coords, c2w: np.array) -> Tuple[np.array, np.array]:
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """
    Tx = img_coords.Tx.to_value(u.rad)
    Ty = img_coords.Ty.to_value(u.rad)
    # get direction vector --> 0,0 is the down the z axis
    # central pixel (Tx=Ty=0) = (0, 0, -1)
    x = np.sin(Tx)
    y = -np.sin(Ty) * np.cos(Tx)
    z = -np.cos(Tx) * np.cos(Ty)

    # alternative rotation (might need verification)
    # alpha = np.arctan2(Tx, Ty)
    # rho = np.sqrt(Tx ** 2 + Ty ** 2)
    # x = np.sin(alpha) * np.sin(rho)
    # y = - np.cos(alpha) * np.sin(rho)
    # z = - np.cos(rho)

    directions = np.stack([x, y, z], axis=-1, dtype=np.float32)

    # Apply camera pose to directions
    rays_d = np.sum(directions[..., None, :] * c2w[:3, :3], axis=-1)

    # rays_d should be already normalized
    # rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)

    # Origin is same for all directions (the optical center)
    rays_o = np.tile(c2w[None, :3, -1], [rays_d.shape[0], rays_d.shape[1], 1])
    return rays_o, rays_d
