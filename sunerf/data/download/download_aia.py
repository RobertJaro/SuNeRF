import argparse
import os

import drms
from dateutil.parser import parse

from sunerf.data.download.download_jsoc import donwload_ds


def download_euv(start_time, dir, client, end_time=None, cadence='1h', channel=None):
    channel_str = f'[{channel}]' if channel is not None else ''
    if end_time is None:
        time_str = f'[{start_time.isoformat("_", timespec="seconds")}]'
    else:
        time_str = f'[{start_time.isoformat("_", timespec="seconds")} / {(end_time - start_time).total_seconds()}s@{cadence}]'
    ds = f'aia.lev1_euv_12s{time_str}{channel_str}{{image}}'
    euv_files = donwload_ds(ds, dir, client).download
    return euv_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download_dir', type=str, required=True)
    parser.add_argument('--email', type=str, required=True)
    parser.add_argument('--t_start', type=str, required=True)
    parser.add_argument('--t_end', type=str, required=False, default=None)
    parser.add_argument('--cadence', type=str, required=False, default='1h')
    parser.add_argument('--channel', type=str, required=False, default=None)
    args = parser.parse_args()

    os.makedirs(args.download_dir, exist_ok=True)
    client = drms.Client(email=args.email)

    start_time = parse(args.t_start)
    end_time = parse(args.t_end) if args.t_end is not None else None

    download_euv(start_time=start_time, end_time=end_time,
                 cadence=args.cadence, channel=args.channel,
                 dir=args.download_dir, client=client)
