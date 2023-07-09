"""
This module is used to download concurrent Stereo A, B and AIA data.'
"""

import argparse
from collections import defaultdict
import glob
import logging
import time

from grpc import stream_stream_rpc_method_handler

import dask
import numpy as np
from tqdm import tqdm
import pandas as pd

from datetime import datetime
from astropy import units as u
from sunpy.net import Fido
from sunpy.net import attrs as a


# Initialize Python Logger
logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-start_date',
    help='Starting date of data download. YYYY-MM-DD',
    type=str,
    default='2011-01-01'
)

parser.add_argument(
    '-end_date',
    help='Ending date of data download. YYYY-MM-DD',
    type=str,
    default='2013-01-01'
)

parser.add_argument(
    '-cadence',
    help='Cadence of data download in minutes.',
    type=int,
    default=60
)

parser.add_argument(
    '-data_path',
    help='Path to store data.',
    type=str,
    default='/mnt/data/'
)

parser.add_argument(
    '-wavelength_tol',
    help='Wavelength tolerance',
    type=int,
    default=10
)

def download_concurrent_data(
    start_date=None, 
    end_date=None, 
    cadence=None,
    data_path=None,
    wavelength_tol=10
    ):
    """Download concurrent data from several helio instruments.

    Parameters
    ----------
    start_date : str
        Starting date of data download

    end_date : str
        Ending date of data download

    cadence : int
        Cadence of data download in minutes

    data_path : str
        Path to store data

    sources : list
        Sources to process. Sources and instruments need to have the same number of elements

    instruments : list
        Sources to process. Sources and instruments need to have the same number of elements

    wavelengths : list
        Wavelengths to process in Angstroms. Sources, instruments, and wavelenghts need to have the same number of elements

    wavelength_tol : int
        Wavelenght tolerance in Angstroms.

    """
    # Convert dates to datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')


    dateRange = pd.date_range(start=start_date, end=end_date, freq='D')

    for date in tqdm(dateRange):

        aia171 = (a.Instrument('AIA') &
                a.Sample(cadence * u.minute) &
                a.Wavelength(17 * u.nm, 18 * u.nm) &
                a.Time(date, date + pd.Timedelta(1, "d")))

        aia193 = (a.Instrument('AIA') &
                a.Sample(cadence * u.minute) &
                a.Wavelength(19 * u.nm, 20 * u.nm) &
                a.Time(date, date + pd.Timedelta(1, "d")))

        aia211 = (a.Instrument('AIA') &
                a.Sample(cadence * u.minute) &
                a.Wavelength(20 * u.nm, 22 * u.nm) &
                a.Time(date, date + pd.Timedelta(1, "d")))                

        aia304 = (a.Instrument('AIA') &
                a.Sample(cadence * u.minute) &
                a.Wavelength(30 * u.nm, 31 * u.nm) &
                a.Time(date, date + pd.Timedelta(1, "d")))

        stereoa = (a.Source('STEREO_A') &
                a.Instrument('EUVI') &
                a.Sample(60 * u.minute) &
                a.Time(date, date + pd.Timedelta(1, "d")))

        stereob = (a.Source('STEREO_B') &
                a.Instrument('EUVI') &
                a.Sample(60 * u.minute) &
                a.Time(date, date + pd.Timedelta(1, "d")))

        results = Fido.search(stereoa | stereob | aia171 | aia193 | aia304 | aia211)
        downloaded_files = Fido.fetch(results, path=data_path+'{source}/{instrument}/{file}')
        # while len(downloaded_files.errors)>0:
        #     downloaded_files = Fido.fetch(downloaded_files)
             

if __name__ == "__main__":
    args = parser.parse_args()
    args = vars(args) 
    LOG.info(f'Downloading data for: {args}')
    download_concurrent_data(**args)
