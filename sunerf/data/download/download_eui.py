import argparse
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.parser import parse
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy_soar import Product, SOOP

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

    #########################################################################
    # find all observations of 174 and 304

    result_174 = Fido.search(time_range, a.Instrument.eui, Product('eui-fsi174-image'), a.Level(2), SOOP('none')) # CHECK SOOP download
    result_304 = Fido.search(time_range, a.Instrument.eui, Product('eui-fsi304-image'), a.Level(2), SOOP('none'))

    #########################################################################
    # find pairs
    start_times_174 = np.array([parse(t) for t in result_174['soar']['Start time']])
    start_times_304 = np.array([parse(t) for t in result_304['soar']['Start time']])

    closest_dates_cond = [np.argmin(np.abs(start_times_304 - t)) for t in start_times_174]
    min_date_diff_cond = np.abs(start_times_304[closest_dates_cond] - start_times_174) < timedelta(minutes=1)

    # filter pairs
    paired_result_304 = result_304['soar'][closest_dates_cond][min_date_diff_cond]
    paired_result_174 = result_174['soar'][min_date_diff_cond]
    #########################################################################
    # sample every hour
    start_times_174 = np.array([parse(t) for t in paired_result_174['Start time']])

    start_time = np.min(start_times_174)
    end_time = np.max(start_times_174)
    target_times = pd.date_range(start_time, end_time, freq='1h')

    condition = [np.argmin(np.abs((start_times_174 - d.to_pydatetime()))) for d in target_times if
                 np.min(np.abs(start_times_174 - d.to_pydatetime())) < timedelta(minutes=20)]

    # filter condition
    sampled_result_304 = paired_result_304[condition]
    sampled_result_174 = paired_result_174[condition]

    #########################################################################
    # print pretty table of available dates
    print('Downloading available dates:')
    with pd.option_context('display.max_rows', None):
        print(pd.DataFrame({'174': [parse(t) for t in sampled_result_174['Start time']],
                            '304': [parse(t) for t in sampled_result_304['Start time']]}))

    #########################################################################
    # download
    Fido.fetch(sampled_result_304, sampled_result_174, path=args.download_dir)
