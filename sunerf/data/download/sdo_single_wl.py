import os
from datetime import datetime

import drms

download_dir = '/mnt/nerf-data/sdo_2012_08/1m_193'
os.makedirs(download_dir, exist_ok=True)
client = drms.Client(verbose=True, email='robert.jarolim@uni-graz.at')

tstart = datetime(2012, 8, 31, 18)
tend = datetime(2012, 8, 31, 23)
wl = 193
cadence = '1m'

tstart, tend = tstart.isoformat('_', timespec='seconds'), tend.isoformat('_', timespec='seconds')
ds = f'aia.lev1_euv_12s[{tstart}Z-{tend}Z@{cadence}][{wl}]{{image}}'
r = client.export(ds, protocol='fits')
r.wait()
download_result = r.download(download_dir)
