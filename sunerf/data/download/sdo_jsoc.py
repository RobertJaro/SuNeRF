import glob
import os
import shutil
from datetime import timedelta, datetime

import drms
from dateutil.parser import parse

download_dir = '/mnt/aia-jsoc'
channels = ['171', '193', '211', '304']
euv_channels = [c for c in channels if c != '6173']

[os.makedirs(os.path.join(download_dir, str(c)), exist_ok=True) for c in channels]

client = drms.Client(verbose=True, email='robert.jarolim@uni-graz.at')


# client = drms.Client(verbose=True, email='betr9863@colorado.edu')


def round_date(t):
    """Round data to closest hour

    Parameters
    ----------
    t : datetime
        datetime to round

    Returns
    -------
    t : datetime
        rounded datetime
    """

    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0) + timedelta(hours=1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)


def download(ds, round_date_sw=False):
    """Round data to closest hour

    Parameters
    ----------
    ds : string
        JSOC download string
    round_date_sw : bool
        whether to round dates to the closest hour
    """
    r = client.export(ds, method='url-tar', protocol='fits')
    r.wait()
    download_result = r.download(download_dir)
    for f in download_result.download:
        shutil.unpack_archive(f, os.path.join(download_dir))
        os.remove(f)
    for f in glob.glob(os.path.join(download_dir, '*.fits')):
        f_info = os.path.basename(f).split('.')
        channel = f_info[3]

        if f_info[0] == 'hmi':
            channel = '6173'
            date = parse(f_info[2][:-4].replace('_', 'T'))
            file_prefix = 'hmi_'
        else:
            date = parse(f_info[2][:-1])
            file_prefix = 'aia' + channel + '_'

        if round_date_sw:
            date = round_date(date)
            shutil.move(f, os.path.join(download_dir, str(channel),
                                        file_prefix + date.isoformat('T', timespec='hours') + '.fits'))
        else:
            shutil.move(f, os.path.join(download_dir, str(channel),
                                        file_prefix + date.isoformat('T', timespec='seconds') + '.fits'))
    [os.remove(f) for f in glob.glob(os.path.join(download_dir, '*.*'))]


def download_date_range(tstart, tend, cadence='6h', round_date_sw=False):
    """Round data to closest hour

    Parameters
    ----------
    tstart : datetime
        Start date.
    tend : datetime 
        End date.
    cadence: String 
        The cadence in hours, minutes, or seconds (i.e. 6h).
    round_date_sw : bool
        whether to round dates to the closest hour

    Returns
    -------
    None (downloads files).
        
    """
    tstart, tend = tstart.isoformat('_', timespec='seconds'), tend.isoformat('_', timespec='seconds')
    print(f'Download AIA: {tstart} -- {tend}')
    download(f'aia.lev1_euv_12s[{tstart}Z-{tend}Z@{cadence}][{",".join(euv_channels)}]{{image}}',
             round_date_sw=round_date_sw)
    if '6173' in channels:
        print(f'Download HMI: {tstart} -- {tend}')
        download(f'hmi.M_720s[{tstart}Z-{tend}Z@{cadence}]{{magnetogram}}', round_date_sw=round_date_sw)


tstart = datetime(2010, 5, 1)
tend = datetime(2014, 12, 31)
td = timedelta(days=30)
cadence = '24h'
round_date_sw = False

dates = [tstart + i * td for i in range((tend - tstart) // td)]
for d in dates:
    download_date_range(d, d + td, cadence=cadence, round_date_sw=round_date_sw)
