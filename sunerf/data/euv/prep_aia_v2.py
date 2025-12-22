import argparse
import glob
import multiprocessing
import os
from itertools import repeat

import numpy as np
from aiapy.calibrate import correct_degradation
from aiapy.calibrate import register
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument("--nproc", type=int, default=None, help="Number of parallel processes (default: 16)")
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)


    def _convert_map(d):
        map_path, out_path, resolution, correction_table = d

        s_map = Map(map_path)
        exposure_time = s_map.meta['EXPTIME']

        if s_map.meta['QUALITY'] != 0 or exposure_time <= 0:
            print(f"Map {map_path} has quality issues, skipping.")
            return

        # north up
        s_map = register(s_map)

        # photometric correction
        s_map = correct_degradation(s_map, correction_table=correction_table)
        # normalize by exposure time
        s_map.data[:] /= exposure_time

        # remove negative values
        s_map.data[s_map.data <= 0] = 0

        target_radius = 1.3 * s_map.rsun_obs
        bottom_left = SkyCoord(-target_radius, -target_radius, observer=s_map.observer_coordinate, frame=frames.Helioprojective)
        top_right = SkyCoord(target_radius, target_radius, observer=s_map.observer_coordinate, frame=frames.Helioprojective)
        s_map = s_map.submap(bottom_left=bottom_left, top_right=top_right)

        # exposure_time = s_map.meta['EXPTIME']
        s_map = s_map.resample((resolution, resolution) * u.pixel)

        coords = all_coordinates_from_map(s_map)
        radius = np.sqrt(coords.Tx ** 2 + coords.Ty ** 2)
        mask = radius > s_map.rsun_obs * 1.3
        s_map.data[mask] = np.nan

        # s_map.data[:] = s_map.data / 2000
        s_map.save(out_path, overwrite=True)


    files = glob.glob(args.data_path)
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]
    correction_table = get_correction_table()

    nproc = os.cpu_count() if args.nproc is None else args.nproc
    with multiprocessing.Pool(nproc) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution), repeat(correction_table))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
