import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from astropy.io.fits import getheader
from sunpy.map import Map

from sunerf.data.utils import psi_norms


def _loadMLprepMap(file_path, out_path, resolution):
    """Load and preprocess PSI file.


    Parameters
    ----------
    file_path: path to the FTIS file.
    resolution: target resolution in pixels.

    Returns
    -------
    the preprocessed SunPy Map
    """
    # load Map
    s_map = Map(file_path)
    if np.abs(s_map.carrington_latitude.value) > 7:
        return
    # adjust image size
    s_map = s_map.resample((resolution, resolution) * u.pix)

    # normalize image data
    data = s_map.data
    data = psi_norms[int(s_map.wavelength.value)](data)
    data[data < 0] = 0  # remove negative values
    data = np.nan_to_num(data, nan=0)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    s_map.save(os.path.join(out_path, os.path.basename(file_path)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--psi_path', type=str,
                   default='/mnt/psi-data/PSI/AIA_171/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--resolution', type=float,
                   default=1024,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/psi-data/prep_psi/171',
                   help='path to save the converted maps.')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    psi_paths = sorted(glob(args.psi_path))

    assert len(psi_paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(psi_paths, repeat(args.output_path), repeat(args.resolution)))
