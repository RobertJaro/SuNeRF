import numpy as np
from astropy import units as u
from astropy.visualization import ImageNormalize, LinearStretch
from sunpy.coordinates import frames
from sunpy.map import all_coordinates_from_map
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
    """Return exact Helioprojective ``[Tx, Ty]`` coordinates for an ARC map.

    Pixel scale times offset is a projection-plane coordinate, not a
    Helioprojective angle away from the reference pixel. Let Astropy WCS apply
    the inverse azimuthal-equidistant projection. This also preserves NumPy's
    ``(ny, nx)`` ordering for non-square images; the former ``.T`` silently
    swapped both axes and their scales.
    """
    coordinates = all_coordinates_from_map(s_map).transform_to(frames.Helioprojective)
    tx = coordinates.Tx.to_value(u.arcsec)
    ty = coordinates.Ty.to_value(u.arcsec)
    return u.Quantity(np.stack([tx, ty], axis=-1), u.arcsec)
