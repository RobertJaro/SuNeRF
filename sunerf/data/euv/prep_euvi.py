import argparse
import glob
import multiprocessing
import os
from itertools import repeat

import numpy as np
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
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)


    def _convert_map(d):
        map_path, out_path, resolution = d

        s_map = Map(map_path)

        # mask missing blocks
        s_map.data[s_map.data <= 0] = np.nan

        # north up
        s_map = s_map.rotate(recenter=True)

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

        # quality check
        if np.isnan(s_map.data).sum() / np.prod(s_map.data.shape) > 0.50:
            print(f"Map {map_path} has too many NaNs, skipping.")
            return

        s_map.save(out_path, overwrite=True)


    files = glob.glob(args.data_path)
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]

    with multiprocessing.Pool(os.cpu_count()) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
