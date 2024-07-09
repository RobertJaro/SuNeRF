import datetime
import os

import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs as a

download_dir = '/mnt/nerf-data/so_2022_02'
os.makedirs(download_dir, exist_ok=True)

# shutil.rmtree(download_dir)
t_start = datetime.datetime(2022, 2, 17, 23, 55)
t_end = datetime.datetime(2022, 2, 18, 0, 5)

query = Fido.search(a.Instrument("EUI"), a.Source('SO'),
                    a.Time(t_start, t_end),
                    a.Wavelength(304 * u.AA))

so, = query
so = so[so['Info'] == 'L2']

files = Fido.fetch(so, path=download_dir, progress=True)
