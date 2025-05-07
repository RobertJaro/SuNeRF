import argparse
import os.path
from datetime import datetime
from threading import Thread
from typing import Iterable

import numpy as np
from tqdm import tqdm

from sunerf.convert.vtk import save_vtk
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


class _SaveFileTask(Thread):

    def __init__(self, out_path, output, metrics=None):
        super().__init__()
        self.out_path = out_path
        self.output = output
        self.metrics = metrics if metrics is not None else []

    def run(self):
        Rs_per_pixel = self.output['Rs_per_pixel']

        # split output into vectors and scalars
        vectors = {k: v for k, v in self.output['data'].items() if len(v.shape) == 4 and v.shape[-1] == 3}
        scalars = {k: v for k, v in self.output['data'].items() if len(v.shape) == 3}

        save_vtk(self.out_path, coords=self.output['coords'], vectors=vectors, scalars=scalars,
                 Rs_per_pixel=Rs_per_pixel)


def convert(sunerf_path, out_path=None, pixel_per_Rs=None, times=None, radius_range=None, r_scaling=False, **kwargs):
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(sunerf_path), 'vtk')
    os.makedirs(out_path, exist_ok=True)
    radius_range = radius_range if radius_range is not None else [20, 120]

    model = ThomsonSuNeRFLoader(sunerf_path)
    # create box with [-max_radius, max_radius] in all directions
    if times is None:
        times = sorted(list(set(model.times())))
        time_indices = range(len(times))
    elif isinstance(times, datetime):
        times = [times]
        time_indices = [0]
    elif isinstance(times, Iterable) and isinstance(times[0], int):
        time_indices = times
        times = np.array(sorted(list(set(model.times()))))
    else:
        times = np.array(times)
        time_indices = range(len(times))

    for i in tqdm(time_indices, total=len(time_indices), desc='Converting'):
        t = times[i]
        output = model.load_cube(radius_range=radius_range, time=t, pixel_per_Rs=pixel_per_Rs, progress=False, **kwargs)
        rho = output['rho']
        # scale rho by r^2 if r_scaling is True
        if r_scaling:
            coords = output['cartesian_coords']
            r = np.linalg.norm(coords[..., :3], axis=-1)
            rho = rho * r ** 2
        # squeeze the time dimension, only use spatial coordinates
        rho = rho[:, :, :, 0]
        coords = output['cartesian_coords'][:, :, :, 0, :3]
        # async save
        out_file = os.path.join(out_path, f'time_{i:03d}.vtk')
        data = {'rho': rho, 'v': output['v']}
        task = _SaveFileTask(out_file, output={'data': data, 'coords': coords, 'Rs_per_pixel': 1 / pixel_per_Rs})
        task.start()


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--sunerf_path', type=str, help='path to the source SuNeRF file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--pixel_per_Rs', type=float, help='pixel size in solar radii', required=False, default=1)
    parser.add_argument('--radius_range', type=float, nargs=2, help='radius range in solar radii', required=False,
                        default=[20, 120])
    parser.add_argument('--times', type=int, nargs='*', help='times to be converted', required=False, default=None)

    args = parser.parse_args()
    sunerf_path = args.sunerf_path

    pixel_per_Rs = args.pixel_per_Rs
    out_path = args.out_path
    radius_range = args.radius_range

    convert(sunerf_path, out_path, pixel_per_Rs=pixel_per_Rs, radius_range=radius_range,
            times=args.times, r_scaling=False)


if __name__ == '__main__':
    main()
