import argparse
import os

import numpy as np
from astropy import units as u

from sunerf.evaluation.loader import SuNeRFLoader

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser('Load data cube from SuNeRF checkpoint')
    parser.add_argument('--chk_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    # Initialize loader
    loader = SuNeRFLoader(args.chk_path)

    times = loader.times()
    ref_time = times[0]

    latitude_range = np.arange(-90, 90, 1) * u.deg
    longitude_range = np.arange(0, 360, 1) * u.deg
    radius = np.linspace(1, 1.4, 100) * u.solRad

    # Load data
    out = loader.load_spherical(longitude_range=longitude_range,
                                radius_range=radius,
                                latitude_range=latitude_range,
                                time=ref_time)
    total_log_ne = out['total_log_ne']
    log_T = out['mean_log_T']

    # save as npz file
    np.savez(args.out_path, log_T=log_T, total_log_ne=total_log_ne,
             latitude=latitude_range, longitude=longitude_range, radius=radius)
