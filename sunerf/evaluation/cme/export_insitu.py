"""Export the SuNeRF reconstruction along the trajectory of in-situ products.

Every ``--insitu_path`` is a prepared in-situ file (real PSP/Solar Orbiter
products of ``sunerf.data.prep`` or the synthetic probes of
``sunerf.data.psi.synthetic_insitu``).  The model is queried at the sample
positions and times; the electron density [cm^-3] and the RTN-radial velocity
[km/s] of the reconstruction are written next to the sample times::

    <out_path>/insitu/<probe>.npz    time_unix, radius_rsun, density_cm3, velocity_radial_kms

``<probe>`` is the stem of the input file.

Example::

    python -m sunerf.evaluation.cme.export_insitu \\
        --sunerf_path /path/save_state.snf --insitu_path /path/PSP_synth.npz \\
        --out_path /path/export
"""
import argparse
import os
from datetime import datetime, timezone

import numpy as np

from sunerf.data.prep.base_insitu import cartesian_from_inertial_coords
from sunerf.evaluation.cme.export_layout import insitu_file
from sunerf.evaluation.loader import ThomsonSuNeRFLoader


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--insitu_path', required=True, nargs='+', help='Prepared in-situ files (npz)')
    parser.add_argument('--out_path', required=True, help='Export directory')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=8192)
    return parser


def sample_model(loader, data, batch_size=8192):
    """Density [cm^-3] and radial velocity [km/s] of the model at the in-situ samples."""
    position = cartesian_from_inertial_coords(data['inertial_coords_r_lat_lon'])
    times = [datetime.fromtimestamp(float(t), timezone.utc).replace(tzinfo=None) for t in data['time_unix']]
    time_norm = np.asarray(loader.normalize_datetime(times), dtype=np.float32)
    query_points = np.concatenate([position.astype(np.float32) / float(loader.Rs_per_ds), time_norm[:, None]], -1)
    out = loader.load_coords(query_points, batch_size=batch_size)
    radius = np.linalg.norm(position, axis=-1)
    r_hat = position / np.clip(radius[:, None], 1e-12, None)
    return {
        'time_unix': data['time_unix'].astype(np.float64),
        'radius_rsun': radius.astype(np.float32),
        'density_cm3': np.asarray(out['rho'], dtype=np.float32).reshape(-1),
        'velocity_radial_kms': np.sum(np.asarray(out['v'], dtype=np.float64).reshape(-1, 3) * r_hat, -1).astype(np.float32),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device)
    for path in args.insitu_path:
        probe = os.path.splitext(os.path.basename(path))[0]
        result = sample_model(loader, np.load(path), batch_size=args.batch_size)
        out_file = insitu_file(args.out_path, probe)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        np.savez(out_file, **result)
        print(f'Exported {result["time_unix"].size} samples of {probe} to {out_file}')


if __name__ == '__main__':
    main()
