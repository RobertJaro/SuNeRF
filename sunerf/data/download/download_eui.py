import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from dateutil.parser import parse
from sunpy.net import Fido
from sunpy.net import attrs as a


def _download_eui(time_range, channels, cadence):
    target_times = pd.date_range(start=time_range.start.to_datetime(), end=time_range.end.to_datetime(),
                                 freq=f'{cadence}H')

    fetch_list = []
    for wl in channels:
        result = Fido.search(time_range, a.Instrument.eui, a.Wavelength(wl * u.AA))
        start_times = np.array(result['vso']['Start Time'])
        indices = [np.argmin(np.abs(Time(t) - start_times)) for t in target_times]
        fetch_list.append(result['vso'][indices])

    print(fetch_list)

    download_files = Fido.fetch(*fetch_list, path=args.download_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--cadence', type=float, required=False, default=1, help='Cadence in hours')
    parser.add_argument('--channels', type=int, nargs='+', required=False, default=[171, 284])
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    start_time = parse(args.t_start)
    end_time = parse(args.t_end) if args.t_end is not None else None
    cadence = args.cadence

    time_range = a.Time(start_time, end_time)

    _download_eui(time_range, args.channels, cadence)
