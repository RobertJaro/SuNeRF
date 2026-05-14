import argparse
import os.path
from collections.abc import Iterable
from datetime import datetime
from threading import Thread

import astropy.units as u
import numpy as np
from dateutil.parser import parse
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
        # split output into vectors and scalars
        vectors = {k: v for k, v in self.output['data'].items() if len(v.shape) == 4 and v.shape[-1] == 3}
        scalars = {k: v for k, v in self.output['data'].items() if len(v.shape) == 3}

        save_vtk(self.out_path, coords=self.output['coords'], vectors=vectors, scalars=scalars)


def _parse_time(time):
    if isinstance(time, datetime):
        return time
    if isinstance(time, str):
        return parse(time)
    return time


def _is_time_index(time):
    return isinstance(time, (int, np.integer)) or (isinstance(time, str) and time.strip().lstrip('+-').isdigit())


def _parse_time_index(time):
    return int(time)


def _normalize_times(times, available_times):
    if times is None:
        times = sorted(list(set(available_times)))
        return times, range(len(times))

    if _is_time_index(times):
        return np.array(sorted(list(set(available_times)))), [int(times)]

    if isinstance(times, (str, datetime)):
        return [_parse_time(times)], [0]

    if isinstance(times, Iterable):
        times = list(times)
        if len(times) == 0:
            return [], []
        if all(_is_time_index(t) for t in times):
            return np.array(sorted(list(set(available_times)))), [_parse_time_index(t) for t in times]
        if any(_is_time_index(t) for t in times):
            raise ValueError('times must be all indices or all datetime strings, not a mix of both')
        return [_parse_time(t) for t in times], range(len(times))

    raise TypeError(f'Unsupported times type: {type(times).__name__}')


def convert(sunerf_path, out_path=None, pixel_per_Rs=None, times=None, radius_range=None, r_scaling=False, **kwargs):
    out_path = out_path if out_path is not None else os.path.join(os.path.dirname(sunerf_path), 'vtk')
    os.makedirs(out_path, exist_ok=True)
    radius_range = radius_range if radius_range is not None else [20, 120]
    if not hasattr(radius_range, 'unit'):
        radius_range = radius_range * u.R_sun

    model = ThomsonSuNeRFLoader(sunerf_path)
    # create box with [-max_radius, max_radius] in all directions
    times, time_indices = _normalize_times(times, model.times())

    for i in tqdm(time_indices, total=len(time_indices), desc='Converting'):
        t = times[i]
        output = model.load_cube(radius_range=radius_range, time=t, pixel_per_Rs=pixel_per_Rs, progress=False, **kwargs)
        # squeeze the time dimension, only use spatial coordinates
        rho = output['rho'][:, :, :, 0]
        coords = output['cartesian_coords'][:, :, :, 0, :3]
        v = output['v'][:, :, :, 0, :]
        r = np.linalg.norm(coords, axis=-1)
        scaled_rho = rho * r ** 2
        scaled_rho = np.nan_to_num(scaled_rho, nan=0.0, posinf=0.0, neginf=0.0)

        # async save
        out_file = os.path.join(out_path, f'time_{i:03d}.vtk')
        data = {'rho': rho, 'scaled_rho': scaled_rho, 'v': v}
        task = _SaveFileTask(out_file, output={'data': data, 'coords': coords})
        task.start()


def main():
    parser = argparse.ArgumentParser(description='Convert NF2 file to VTK.')
    parser.add_argument('--sunerf_path', type=str, help='path to the source SuNeRF file')
    parser.add_argument('--out_path', type=str, help='path to the target VTK file', required=False, default=None)
    parser.add_argument('--pixel_per_Rs', type=float, help='pixel size in solar radii', required=False, default=1)
    parser.add_argument('--radius_range', type=float, nargs=2, help='radius range in solar radii', required=False,
                        default=[20, 120])
    parser.add_argument('--times', nargs='*', help='time indices or ISO datetime strings to convert', required=False,
                        default=None)

    args = parser.parse_args()
    sunerf_path = args.sunerf_path

    pixel_per_Rs = args.pixel_per_Rs
    out_path = args.out_path
    radius_range = args.radius_range

    convert(sunerf_path, out_path, pixel_per_Rs=pixel_per_Rs, radius_range=radius_range,
            times=args.times, r_scaling=False)


if __name__ == '__main__':
    main()
