import argparse
import glob
import multiprocessing
import os
from itertools import repeat

import numpy as np
from astropy import units as u
from sunpy.map import Map
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)


    def _convert_map(d):
        map_path, out_path, resolution = d

        s_map = Map(map_path)
        # exposure_time = s_map.meta['EXPTIME']
        s_map = s_map.resample((resolution, resolution) * u.pixel)
        # s_map.data[:] /= exposure_time
        # mask missing blocks
        s_map.data[s_map.data <= 0] = np.nan

        # s_map.data[:] = s_map.data / 2000
        s_map.save(out_path, overwrite=True)


    files = glob.glob(args.data_path)
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]

    with multiprocessing.Pool(os.cpu_count()) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
