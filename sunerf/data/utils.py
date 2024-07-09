from astropy import units as u
from astropy.visualization import ImageNormalize, LinearStretch
from itipy.data.editor import LoadMapEditor, NormalizeRadiusEditor, AIAPrepEditor
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


def loadAIAMap(file_path, resolution=1024, map_reproject=False):
    """Load and preprocess AIA file to make them compatible to ITI.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels of 2.2 solar radii.
    map_reproject: apply preprocessing to remove off-limb (map to heliographic map and transform back to original view).

    Returns
    -------
    the preprocessed SunPy Map
    """
    s_map, _ = LoadMapEditor().call(file_path)
    assert s_map.meta['QUALITY'] == 0, f'Invalid quality flag while loading AIA Map: {s_map.meta["QUALITY"]}'
    s_map = NormalizeRadiusEditor(resolution).call(s_map)
    s_map = AIAPrepEditor(calibration='auto').call(s_map)
    if map_reproject:
        s_map = transform(s_map, lat=s_map.heliographic_latitude,
                          lon=s_map.heliographic_longitude, distance=1 * u.AU)
    return s_map
