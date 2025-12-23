import argparse
import glob
import multiprocessing
import os
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from dateutil.parser import parse
from matplotlib.colors import LogNorm
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--lat', type=float, default=None)
    parser.add_argument('--lon', type=float, default=None)
    parser.add_argument('--hpc_width', type=float, default=None)
    parser.add_argument('--hpc_height', type=float, default=None)
    parser.add_argument('--date_range', type=str, nargs=2, default=None)
    parser.add_argument('--max_radius', type=float, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'img'), exist_ok=True)

    if args.lat is not None and args.lon is not None and args.hpc_width is not None and args.hpc_height is not None:
        subframe_config = {
            'lat': args.lat * u.deg,
            'lon': args.lon * u.deg,
            'width': args.hpc_width * u.arcsec,
            'height': args.hpc_height * u.arcsec
        }
    else:
        subframe_config = None


    def _convert_map(d):
        map_path, out_path, resolution, correction_table, date_range, max_radius, overwrite = d

        if os.path.exists(out_path) and not overwrite:
            return

        s_map = Map(map_path)


        # skip SOOPs
        if s_map.meta['SOOPTYPE'] != 'none':
            print(f'Skipping SOOP: {s_map.meta["SOOPTYPE"]}')
            return

        # north up
        s_map = s_map.rotate(recenter=True)

        if date_range is not None:
            s_map_date = s_map.date
            if s_map_date < date_range[0] or s_map_date > date_range[1]:
                return

        # extract subframe
        if subframe_config is not None:
            coord = SkyCoord(lon=subframe_config['lon'], lat=subframe_config['lat'],
                             frame=frames.HeliographicCarrington,
                             observer=s_map.observer_coordinate)
            coord_pix = s_map.world_to_pixel(coord)
            pixel_width = (subframe_config['width'] / s_map.scale[0]).to_value(u.pixel).astype(int)
            pixel_height = (subframe_config['height'] / s_map.scale[1]).to_value(u.pixel).astype(int)

            bottom_left = u.Quantity([coord_pix.x.to_value(u.pix) - pixel_width // 2,
                                      coord_pix.y.to_value(u.pix) - pixel_height // 2] * u.pix)
            top_right = u.Quantity([coord_pix.x.to_value(u.pix) + pixel_width // 2 - 1,
                                    coord_pix.y.to_value(u.pix) + pixel_height // 2 - 1] * u.pix)

            s_map = s_map.submap(bottom_left=bottom_left, top_right=top_right)

        if resolution is not None:
            s_map = s_map.resample((resolution, resolution) * u.pixel)

        s_map.data[s_map.data <= 0] = 0

        if max_radius is not None:
            coords = all_coordinates_from_map(s_map)
            map_radius = (coords.Tx ** 2 + coords.Ty ** 2) ** 0.5 / s_map.rsun_obs
            s_map.data[map_radius > max_radius] = np.nan

        s_map.save(out_path, overwrite=True)

        fig, ax =  plt.subplots(figsize=(5, 5), dpi=100, subplot_kw={'projection': s_map})
        s_map.plot(axes=ax, norm=LogNorm(vmin=1, vmax=1e3))
        fig.savefig(os.path.join(os.path.dirname(out_path), 'img', os.path.basename(out_path) + '.png'))
        plt.close(fig)

    files = sorted(glob.glob(args.data_path))
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]
    correction_table = get_correction_table()
    date_range = None if args.date_range is None else [parse(d) for d in args.date_range]

    with multiprocessing.Pool(os.cpu_count()) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution), repeat(correction_table), repeat(date_range),
                     repeat(args.max_radius), repeat(args.overwrite))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
