import argparse
import glob
import multiprocessing
import os
from itertools import repeat

import aiapy.calibrate
from aiapy.calibrate import register
from aiapy.calibrate.util import get_correction_table
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map
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
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

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
        map_path, out_path, resolution, correction_table = d

        s_map = Map(map_path)
        s_map = register(s_map)
        # s_map = deconvolve(s_map)

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

        exposure_time = s_map.meta['EXPTIME']
        if s_map.meta['QUALITY'] != 0 or exposure_time <= 0:
            print(f'invalid image: {map_path}; quality={s_map.meta["QUALITY"]}, exposure_time={exposure_time}')
            return
        if resolution is not None:
            s_map = s_map.resample((resolution, resolution) * u.pixel)

        s_map = aiapy.calibrate.correct_degradation(s_map, correction_table=correction_table)
        s_map.data[:] /= exposure_time
        s_map.data[s_map.data <= 0] = 0
        # s_map.data[:] = s_map.data / 5000
        s_map.save(out_path, overwrite=True)


    files = sorted(glob.glob(args.data_path))
    out_paths = [os.path.join(args.out_path, os.path.basename(f)) for f in files]
    correction_table = get_correction_table()

    with multiprocessing.Pool(os.cpu_count()) as p:
        zip_in = zip(files, out_paths, repeat(args.resolution), repeat(correction_table))
        [_ for _ in tqdm(p.imap_unordered(_convert_map, zip_in), total=len(files))]
