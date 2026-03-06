import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--occ_range', type=float, nargs='+', help='Occulting range in solar radii', default=[2.5, 15.0])
    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'video_polar')
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
    mid_time = min_time + (max_time - min_time) / 2

    # time = datetime(2024, 9, 16, 15, 8)
    lon = 0 * u.deg
    lat = 90 * u.deg

    occ_min = args.occ_range[0] * u.R_sun
    occ_max = args.occ_range[1] * u.R_sun

    n_points = 50
    points_1 = zip(np.ones(n_points) * u.AU,
                   np.ones(n_points) * lat,
                   np.ones(n_points) * lon,
                   pd.date_range(start=min_time, end=max_time, periods=n_points))
    points = list(points_1)

    brightness_norm = LogNorm()
    density_norm = LogNorm()

    for i, (d, lat, lon, time) in tqdm(enumerate(points), total=len(points)):
        obs_coord = SkyCoord(radius=d, lat=lat, lon=lon, frame=frames.HeliographicCarrington, obstime=time, observer='self')
        lon = obs_coord.transform_to(frames.HeliocentricInertial).lon
        model_out = sunerf_loader.load_image(lat, lon, time,
                                             distance=d, resolution=(256, 256) * u.pix,
                                             occ_min=occ_min, occ_max=occ_max,
                                             progress=False)

        tB_map = model_out['tB_map']
        pB_map = model_out['pB_map']
        density_map = model_out['density_map']

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': tB_map})

        ax = axs[0]
        im = ax.imshow(tB_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax)
        ax.set_title('Total Brightness')
        tB_map.draw_grid(ax, color='blue')

        ax = axs[1]
        im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=brightness_norm, origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax)
        ax.set_title('Polarized Brightness')
        tB_map.draw_grid(ax, color='blue')

        ax = axs[2]
        im = ax.imshow(density_map.data, cmap='inferno', origin='lower', norm=density_norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
        fig.colorbar(im, cax=cax)
        ax.set_title('Density')
        tB_map.draw_grid(ax, color='blue')

        fig.suptitle(f'Latitude: {lat:02.0f}, Longitude: {lon:03.0f}, Time: {time.isoformat(" ", timespec="minutes")}',
                     fontsize=16)

        fig.tight_layout()
        fig.savefig(os.path.join(args.out_path, f"cme_frame{i:03d}.jpg"), dpi=300)
        plt.close('all')
