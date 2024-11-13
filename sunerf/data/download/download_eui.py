import argparse
import os
from datetime import timedelta

import numpy as np
from astropy import units as u
from dateutil.parser import parse
from sunpy.net import Fido
from sunpy.net import attrs as a

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)

    start_time = parse(args.t_start)
    end_time = parse(args.t_end) if args.t_end is not None else None

    time_range = a.Time(start_time, end_time)

    result_174 = Fido.search(time_range, a.Instrument.eui, a.Wavelength(174 * u.AA))
    result_304 = Fido.search(time_range, a.Instrument.eui, a.Wavelength(304 * u.AA))

    start_times_174 = np.array([t.to_datetime() for t in result_174['vso']['Start Time']])
    start_times_304 = np.array([t.to_datetime() for t in result_304['vso']['Start Time']])

    condition = [np.argmin(np.abs((start_times_304 - d))) for d in start_times_174]

    result_304 = result_304['vso'][condition]

    start_times_174 = np.array([t.to_datetime() for t in result_174['vso']['Start Time']])
    start_times_304 = np.array([t.to_datetime() for t in result_304['vso']['Start Time']])

    paired_condition = [np.abs(d1 - d2) < timedelta(minutes=1) for d1, d2 in zip(start_times_174, start_times_304)]

    result_174 = result_174['vso'][paired_condition]
    result_304 = result_304['vso'][paired_condition]

    # subsample list
    result_174 = result_174['vso'][::5]
    result_304 = result_304['vso'][::5]

    Fido.fetch(result_174, result_304, path=args.download_dir)
