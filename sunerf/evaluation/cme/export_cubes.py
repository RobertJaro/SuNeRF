import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from tqdm import tqdm

from sunerf.convert.vtk import save_vtk
from sunerf.evaluation.loader import ThomsonSuNeRFLoader

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)
    parser.add_argument('--radius_range', type=float, nargs=2, help='Radius range to plot in solar radii',
                        default=[3, 15])
    parser.add_argument('--n_points', type=int, default=10,
                        help='Number of time points to sample between min and max observer time')
    parser.add_argument('--time_range', type=str, nargs=2, metavar=('START', 'END'), default=None,
                        help='Optional start and end time (ISO format), e.g. 2023-01-01T00:00:00 '
                             '2023-01-02T00:00:00')
    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'cubes')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    observers = sunerf_loader.observers

    observer_times = set([o['time'] for o in observers])
    min_time = min(observer_times)
    max_time = max(observer_times)

    if args.n_points <= 0:
        raise ValueError('--n_points must be a positive integer.')

    time_start, time_end = min_time, max_time
    if args.time_range is not None:
        time_start = pd.to_datetime(args.time_range[0])
        time_end = pd.to_datetime(args.time_range[1])
        if time_start > time_end:
            raise ValueError('--time_range start must be earlier than or equal to end.')

    times = pd.date_range(start=time_start, end=time_end, periods=args.n_points)

    for i, time in tqdm(enumerate(times), total=len(times)):
        out = sunerf_loader.load_cube(radius_range=args.radius_range * u.R_sun, time=time, pixel_per_Rs=64 // 15)

        rho = out['rho'][:, :, :, 0]  # x, y, z, time, 1
        cartesian_coords = out['cartesian_coords'][:, :, :, 0, :3]
        radial_distance = np.linalg.norm(cartesian_coords, axis=-1)

        normalized_rho = rho * radial_distance ** 2
        normalized_rho = np.nan_to_num(normalized_rho, nan=0.0, posinf=0.0, neginf=0.0)

        # save to vtk format
        save_vtk(os.path.join(args.out_path, f'cube_{i:03d}.vtk'), coords=cartesian_coords,
                 scalars={'rho': normalized_rho})
