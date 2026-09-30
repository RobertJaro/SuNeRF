"""Export the density of a SuNeRF reconstruction on the grid of the PSI/MAS reference.

The SuNeRF snapshot is queried at the grid points and the dump time of every
PSI cube (``load_psi_frame``).  The density is written as electron density in
cm^-3 with the axis order ``rho[radius, latitude, longitude]``.  The coordinates
are written for every dump because the PSI mesh changes within a series.
``--stride`` exports every n-th grid point of all three axes for quick looks::

    <out_path>/density/rho000050.npy        reconstruction
    <out_path>/coordinates/rho000050.npz    radius [R_sun], latitude [rad], longitude [rad, Carrington],
                                            time [datetime64[s]]

Example::

    python -m sunerf.evaluation.cme.export_density \
        --sunerf_path /path/save_state.snf --data_path "/path/rho/rho*.hdf" \
        --dump_times /path/dump_times.json --out_path /path/export
"""
import argparse
import os
from dataclasses import replace

import numpy as np
from tqdm import tqdm

from sunerf.data.psi.dump_times import load_dump_times
from sunerf.data.psi.render_psi_thomson import carrington_to_hci_longitude
from sunerf.evaluation.cme.density_series import list_series_files, load_psi_frame
from sunerf.evaluation.cme.export_layout import coordinates_file, density_file
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--data_path', required=True, help='Glob of PSI density cubes (rho*.hdf)')
    parser.add_argument('--out_path', required=True, help='Output directory')
    parser.add_argument('--dump_times', help='PSI dump_times.json (default: 5 min cadence from 2021-10-28T15:30)')
    parser.add_argument('--frames', default=None, help='Slice of the sorted cube files, e.g. 12: or ::2')
    parser.add_argument('--stride', type=int, default=1, help='Subsampling of the three grid axes')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--trusted', action='store_true', help='Load a pickled (non weights-only) save state')
    return parser


def model_reference_frame(loader):
    """Reference frame of the model coordinates; all data sets of a run share it."""
    reference_frames = {config['reference_frame'] for config in loader.config.values()}
    if len(reference_frames) != 1:
        raise ValueError(f'The save state mixes reference frames: {sorted(reference_frames)}')
    return reference_frames.pop()


def model_longitude(longitude, time, reference_frame):
    """Longitude of the ground-truth grid (rad) in the reference frame of the model."""
    if reference_frame == 'carrington':
        return longitude
    if reference_frame == 'heliographic':
        # HAO runs: the observer longitudes of the synthetic images are given on the grid of the cubes
        return longitude
    if reference_frame == 'inertial':
        # same rotation as the rendering of the PSI images
        return (longitude + carrington_to_hci_longitude(time)) % (2 * np.pi)
    raise ValueError(f'Unsupported reference frame {reference_frame}; use carrington, heliographic or inertial.')


def model_query_points(loader, frame):
    """Model query points of the grid of ``frame`` rotated into the reference frame of the model."""
    longitude = model_longitude(frame.longitude, frame.time, model_reference_frame(loader))
    return replace(frame, longitude=longitude).query_points(
        loader.Rs_per_ds, loader.normalize_datetime(frame.time))


def load_sunerf_density(loader, frame, batch_size=8192):
    """Electron density of the reconstruction in cm^-3 at the grid points and time of ``frame``."""
    return loader.load_coords(model_query_points(loader, frame), batch_size=batch_size)['rho'][..., 0]


def main(argv=None):
    args = build_parser().parse_args(argv)
    for name in ('density', 'coordinates'):
        os.makedirs(os.path.join(args.out_path, name), exist_ok=True)

    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device, trusted=args.trusted)
    dump_times = load_dump_times(args.dump_times) if args.dump_times else None

    for file in tqdm(list_series_files(args.data_path, args.frames), desc='frames'):
        frame = load_psi_frame(file, dump_times)
        step = slice(None, None, args.stride)
        frame = replace(frame, rho=frame.rho[step, step, step], radius=frame.radius[step],
                        latitude=frame.latitude[step], longitude=frame.longitude[step])
        rho = load_sunerf_density(loader, frame, batch_size=args.batch_size)

        name = os.path.splitext(os.path.basename(file))[0]
        np.save(density_file(args.out_path, name), rho.astype(np.float32))
        np.savez(coordinates_file(args.out_path, name), radius=frame.radius,
                 latitude=frame.latitude, longitude=frame.longitude, time=np.datetime64(frame.time, 's'))


if __name__ == '__main__':
    main()
