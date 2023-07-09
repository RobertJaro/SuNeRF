from typing import Tuple

import numpy as np
from sunpy.map.mapbase import PixelPair


def get_rays(height: int, width: int, ref_pixel: PixelPair, focal_length: float, c2w: np.array) -> Tuple[np.array, np.array]:
    r"""
    Find origin and direction of rays through every pixel and camera origin.
    """

    # Apply pinhole camera model to gather directions at each pixel
    i, j = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing='ij')
    i, j = i.transpose(-1, -2), j.transpose(-1, -2)

    directions = np.stack([(i - ref_pixel.x.value - 0.5) / focal_length,
                           -(j - ref_pixel.y.value - 0.5) / focal_length,
                           -np.ones_like(i)], axis=-1)

    # Apply camera pose to directions
    rays_d = np.sum(directions[..., None, :] * c2w[:3, :3], axis=-1)

    # Origin is same for all directions (the optical center)
    rays_o = np.tile(c2w[None, :3, -1], [rays_d.shape[0], rays_d.shape[1], 1])
    return rays_o, rays_d
