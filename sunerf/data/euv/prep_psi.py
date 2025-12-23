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
    parser.add_argument('--resolution', type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)


    def _convert_map(d):
        map_path, out_path, resolution = d
        if os.path.exists(out_path):
            return

        s_map = Map(map_path)
        if np.abs(s_map.carrington_latitude.value) > 7:
            return

        s_map = s_map.resample((resolution, resolution) * u.pixel)
        s_map.save(out_path, overwrite=True)


    files = glob.glob(args.data_path, recursive=True)
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]

    with multiprocessing.Pool(32) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
