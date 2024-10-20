import argparse
import glob
import os

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map

from astropy import units as u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    # lat / lon
    parser.add_argument('--lat', type=float, required=False, default=0)
    parser.add_argument('--lon', type=float, required=False, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    files = sorted(glob.glob(args.data_path))

    ref_map = Map(files[0])
    coords = all_coordinates_from_map(ref_map).transform_to(frames.HeliographicCarrington)

    target_coord = SkyCoord(lon=args.lon * u.deg, lat=args.lat * u.deg, frame=frames.HeliographicCarrington,
                            observer=ref_map.observer_coordinate)

    fig = plt.figure(figsize=(15, 5))

    ax = plt.subplot(131, projection=ref_map)
    im = ax.imshow(ref_map.data, **ref_map.plot_settings)
    ref_map.draw_grid(axes=ax, system='carrington', grid_spacing=10 * u.deg)

    ax.set_title(f'Map - {ref_map.date.to_datetime().isoformat(" ", timespec="minutes")}')
    plt.colorbar(im, ax=ax)
    ax.plot_coord(target_coord, 'x', color='red')

    ax = plt.subplot(132)
    im = ax.imshow(coords.lat.to_value(u.deg), origin='lower')
    ax.set_title('Latitude')
    plt.colorbar(im, ax=ax)

    ax = plt.subplot(133)
    im = ax.imshow(coords.lon.to_value(u.deg), origin='lower')
    ax.set_title('Longitude')
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    plt.savefig(os.path.join(args.out_path, 'coordinates.jpg'), dpi=300)
    plt.close()
