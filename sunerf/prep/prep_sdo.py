import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader
from itipy.data.editor import AIAPrepEditor
from sunpy.coordinates import frames
from sunpy.map import Map

from sunerf.data.utils import sdo_norms, str2bool


def _loadMLprepMap(file_path, out_path, target_scale, center_crop, subframe):
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
    # load Map
    save_path = os.path.join(out_path, os.path.basename(file_path))
    # skip existing files
    if os.path.exists(save_path):
        return save_path
    s_map = Map(file_path)
    s_map = AIAPrepEditor(calibration='auto').call(s_map)
    # adjust image size
    scale_factor = s_map.scale[0].value / target_scale
    s_map = s_map.rotate(recenter=True, scale=scale_factor, missing=0, order=4)
    if center_crop:
        bl = SkyCoord(-1000 * u.arcsec, -1000 * u.arcsec, frame=s_map.coordinate_frame)
        tr = SkyCoord(1000 * u.arcsec, 1000 * u.arcsec, frame=s_map.coordinate_frame)
        s_map = s_map.submap(bottom_left=bl, top_right=tr)
    if subframe is not None:
        coord = SkyCoord(subframe['hgc_lon'] * u.deg, subframe['hgc_lat'] * u.deg, frame=frames.HeliographicCarrington,
                         observer=s_map.observer_coordinate)
        x, y = s_map.world_to_pixel(coord)
        x = int(x.value)
        y = int(y.value)

        w = subframe['width'] // 2
        h = subframe['height'] // 2
        s_map = s_map.submap(bottom_left=[x - w, y - h] * u.pixel, top_right=[x + w, y + h] * u.pixel)
    # normalize image data
    data = s_map.data
    data = sdo_norms[int(s_map.wavelength.value)](data)
    data[data < 0] = 0  # remove negative values
    data = np.nan_to_num(data, nan=0)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    s_map.save(save_path)
    return save_path


if __name__ == '__main__':
    # fix write delay bug
    AIAPrepEditor(calibration='auto')

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--sdo_file_path', type=str,
                   default='/mnt/nerf-data/sdo_2012_08/1h_193/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--scale', type=float,
                   default=1.2,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/nerf-data/prep_2012_08/193',
                   help='path to save the converted maps.')
    p.add_argument('--center_crop', type=str2bool,
                   default=False,
                   help='apply a crop of the center for initial training.')
    subparsers = p.add_subparsers()
    group = subparsers.add_parser('subframe', help='optional crop a subframe from each map.')
    group.add_argument('--hgc_lon', dest='hgc_lon', type=float, help='carrington longitude of the frame center.', required=True)
    group.add_argument('--hgc_lat', dest='hgc_lat', type=float, help='carrington latitude of the frame center.', required=True)
    group.add_argument('--width', type=int, help='width of the frame in pixels.', required=True)
    group.add_argument('--height', type=int, help='height of the frame in pixels.', required=True)

    args = p.parse_args()

    subframe = {'hgc_lon': args.hgc_lon, 'hgc_lat': args.hgc_lat, 'width': args.width, 'height': args.height} \
        if 'hgc_lon' in args else None

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    sdo_paths = sorted(glob(args.sdo_file_path))

    # remove invalid AIA files
    sdo_paths = [f for f in sdo_paths if getheader(f, 1)['QUALITY'] == 0]

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap,
                  zip(sdo_paths, repeat(args.output_path), repeat(args.scale),
                      repeat(args.center_crop), repeat(subframe)))
