import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, LinearStretch
from sunpy.visualization.colormaps import cm

from sunerf.baseline.reprojection import transform

sdo_img_norm = ImageNormalize(vmin=0, vmax=1, stretch=LinearStretch(), clip=True)

# !stretch is connected to NeRF!
sdo_norms = {171: ImageNormalize(vmin=0, vmax=8600, stretch=LinearStretch(), clip=False),
             193: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=False),
             195: ImageNormalize(vmin=0, vmax=9800, stretch=LinearStretch(), clip=False),
             211: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=False),
             284: ImageNormalize(vmin=0, vmax=5800, stretch=LinearStretch(), clip=False),
             304: ImageNormalize(vmin=0, vmax=8800, stretch=LinearStretch(), clip=False), }

psi_norms = {171: ImageNormalize(vmin=0, vmax=22348.267578125, stretch=LinearStretch(), clip=True),
             193: ImageNormalize(vmin=0, vmax=50000, stretch=LinearStretch(), clip=True),
             211: ImageNormalize(vmin=0, vmax=13503.1240234375, stretch=LinearStretch(), clip=True), }

so_norms = {304: ImageNormalize(vmin=0, vmax=300, stretch=LinearStretch(), clip=False),
            174: ImageNormalize(vmin=0, vmax=300, stretch=LinearStretch(), clip=False)}

sdo_cmaps = {171: cm.sdoaia171, 174: cm.sdoaia171, 193: cm.sdoaia193, 211: cm.sdoaia211, 304: cm.sdoaia304}


def get_azimuthal_equidistant_coordinates(s_map):
    data = s_map.data

    coord_grid = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    coord_grid = np.stack(coord_grid, dtype=np.float32).T
    coord_grid[..., 0] = coord_grid[..., 0] - (data.shape[0] - 1) / 2
    coord_grid[..., 1] = coord_grid[..., 1] - (data.shape[1] - 1) / 2

    coord_grid[..., 0] = coord_grid[..., 0] * s_map.scale[0].to_value(u.arcsec / u.pix)
    coord_grid[..., 1] = coord_grid[..., 1] * s_map.scale[1].to_value(u.arcsec / u.pix)

    coord_grid = coord_grid * u.arcsec

    # shift center
    coord_grid[..., 0] = coord_grid[..., 0] + s_map.reference_coordinate.Tx.to(u.arcsec)
    coord_grid[..., 1] = coord_grid[..., 1] + s_map.reference_coordinate.Ty.to(u.arcsec)

    return coord_grid