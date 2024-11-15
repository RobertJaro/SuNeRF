import argparse
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from dateutil.parser import parse
from sunpy.net import Fido
from sunpy.net import attrs as a


# rename files
def _rename_euvi(f, t):
    header = fits.getheader(f)
    source = header['OBSRVTRY']
    obs_time = t.strftime('%Y%m%d_%H%M%S')
    wl = header['WAVELNTH']

    new_filename = f'{source}_{obs_time}_{wl}.fts'
    base_path = os.path.dirname(f)
    os.rename(f, os.path.join(base_path, new_filename))
    # print(f, 'to', new_filename)


def _download_suvi(time_range, channels):
    target_wl = Fido.search(time_range, a.Instrument('suvi'),a.Level.two, a.goes.SatelliteNumber(16),
                            a.Wavelength(304 * u.AA))
    target_wl = target_wl[:, ::15]
    target_times = target_wl['vso']['Start Time']

    fetch_list = []
    for wl in channels:
        result = Fido.search(time_range, a.Instrument('suvi'),a.Level.two, a.goes.SatelliteNumber(16), a.Wavelength(wl * u.AA))
        start_times = np.array(result['vso']['Start Time'])
        indices = [np.argmin(np.abs(t - start_times)) for t in target_times]
        fetch_list.append(result['vso'][indices])

    download_files = Fido.fetch(*fetch_list, path=args.download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--channels', type=int, nargs='+', required=False, default=[94, 131, 171, 195, 284, 304])
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    start_time = parse(args.t_start)
    end_time = parse(args.t_end) if args.t_end is not None else None
    cadence = args.cadence

    time_range = a.Time(start_time, end_time)

    _download_suvi(time_range, args.channels)
