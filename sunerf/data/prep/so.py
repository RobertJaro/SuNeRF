import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from sunpy.map import Map

from sunerf.data.utils import so_norms


def _loadMLprepMap(file_path, out_path):
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

    # normalize image data
    data = s_map.data
    data = data / s_map.exposure_time.to(u.s).value
    data = so_norms[int(s_map.wavelength.value)](data)
    data[data < 0] = 0  # remove negative values
    data = np.nan_to_num(data, nan=0)
    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    s_map.save(os.path.join(out_path, os.path.basename(file_path)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--path', type=str,
                   default='/mnt/nerf-data/so_2022_02/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/nerf-data/prep_so_2022_02',
                   help='path to save the converted maps.')
    args = p.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    paths = sorted(glob(args.path))

    assert len(paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(paths, repeat(args.output_path)))
