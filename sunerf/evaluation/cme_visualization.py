import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def _get_mask(s_map, occ_rad=0.1 * u.AU, outer_rad=130 * u.R_sun):
    # mask occultor
    img_coords = get_azimuthal_equidistant_coordinates(s_map)
    x = img_coords[..., 0]
    y = img_coords[..., 1]
    solar_center = SkyCoord(0 * u.deg, 0 * u.deg, frame=s_map.coordinate_frame)

    pixel_radii = np.sqrt((x - solar_center.Tx) ** 2 + (y - solar_center.Ty) ** 2)

    mask = ((pixel_radii < s_map.rsun_obs * occ_rad.to_value(u.R_sun)) |
            (pixel_radii > s_map.rsun_obs * outer_rad.to_value(u.R_sun)))

    return mask


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory')

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'cme')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    observer_longitudes = set([o['longitude'] % (360 * u.deg) for o in observers])
    observer_times = set([o['time'] for o in observers])

    min_time = min(observer_times)
    max_time = max(observer_times)
    mid_time = (max_time - min_time) / 2 + min_time

    n_points = 20
    points_0 = zip(np.ones(n_points) * u.AU,
                   np.zeros(n_points) * u.deg,
                   np.linspace(135 - 90, 135 - 90, n_points) * u.deg,
                   pd.date_range(start=min_time, end=mid_time, periods=n_points))
    points_1 = zip(np.ones(n_points) * u.AU,
                   np.zeros(n_points) * u.deg,
                   np.linspace(135 - 90, 135 + 90, n_points) * u.deg,
                   [mid_time] * n_points)
    points_2 = zip(np.ones(n_points) * u.AU,
                   np.linspace(0, 90, n_points) * u.deg,
                   np.linspace(135 + 90, 135 + 90, n_points) * u.deg,
                   [mid_time] * n_points)
    points_3 = zip(np.ones(n_points) * u.AU,
                   np.linspace(90, 90, n_points) * u.deg,
                   np.linspace(135 + 90, 135 + 90, n_points) * u.deg,
                   pd.date_range(start=mid_time, end=max_time, periods=n_points))
    points = list(points_0) + list(points_1) + list(points_2) + list(points_3)

    brightness_norm = LogNorm(vmin=1e-3, vmax=1)

    for i, (d, lat, lon, time) in tqdm(enumerate(points), total=len(points)):
        model_out = sunerf_loader.load_observer_image(lat, lon, time, distance=d,
                                                      resolution=(256, 256) * u.pix, progress=False)

        tB_map = model_out['maps'][0]
        pB_map = model_out['maps'][1]
        density = model_out['density']

        mask = _get_mask(tB_map)

        tB_map.data[mask] = np.nan
        pB_map.data[mask] = np.nan
        density[mask] = np.nan

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        ax = axs[0]
        im = ax.imshow(tB_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('Total Brightness')

        ax = axs[1]
        im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('Polarized Brightness')

        ax = axs[2]
        im = ax.imshow(model_out['density'], cmap='inferno', origin='lower', norm='log', vmax=1e1, vmin=1e-2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_title('Density')

        fig.suptitle(f'Latitude: {lat:02.0f}, Longitude: {lon:03.0f}, Time: {time.isoformat(" ", timespec="minutes")}', fontsize=16)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_path, f"cme_frame{i:03d}.jpg"), dpi=300)
        plt.close('all')
