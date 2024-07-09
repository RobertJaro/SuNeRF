import glob
import os
import shutil

import numpy as np
from dateutil.parser import parse


def get_intersecting_files(path, dirs, months=None, years=None, ext=None, parse_f=parse):
    """Group files from different folders by basename.

    Parameters
    ----------
    path: base directory
    dirs: subfolders
    months: filter months
    years: filter years
    ext: extension for file search
    parse_f: function for parsing dates from basenames (required for filter months/years)

    Returns
    -------
    list of folders and files (shape: n_folders, n_files)
    """
    pattern = '*' if ext is None else '*' + ext
    basenames = [
        [os.path.basename(path) for path in glob.glob(os.path.join(path, str(d), '**', pattern), recursive=True)] for d
        in dirs]
    basenames = list(set(basenames[0]).intersection(*basenames))
    if months:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse_f(bn.split('.')[0]).month in months]
    if years:  # assuming filename is parsable datetime
        basenames = [bn for bn in basenames if parse_f(bn.split('.')[0]).year in years]
    basenames = sorted(list(basenames))
    return [[os.path.join(path, str(dir), b) for b in basenames] for dir in dirs]


# find all files from SDO, STEREO A and STEREO B
sdo_files = get_intersecting_files('/mnt/data/sdo_jsoc', ['171', '193', '211', '304'], years=[2013], months=[1])
stereo_a_files = get_intersecting_files('/mnt/data/stereo_iti_converted', ['171', '195', '284', '304'], years=[2013],
                                        months=[1], ext='_A.fits', parse_f=lambda s: parse(s[:-2]))
stereo_b_files = get_intersecting_files('/mnt/data/stereo_iti_converted', ['171', '195', '284', '304'], years=[2013],
                                        months=[1], ext='_B.fits', parse_f=lambda s: parse(s[:-2]))

# create directories for copy
sdo_dirs = {os.path.dirname(f.replace('/mnt/data/sdo_jsoc', '/mnt/data/aligned/SDO')) for f in np.ravel(sdo_files)}
stereo_a_dirs = {os.path.dirname(f.replace('/mnt/data/stereo_iti_converted', '/mnt/data/aligned/STEREO_A')) for f in
                 np.ravel(stereo_a_files)}
stereo_b_dirs = {os.path.dirname(f.replace('/mnt/data/stereo_iti_converted', '/mnt/data/aligned/STEREO_B')) for f in
                 np.ravel(stereo_b_files)}

[os.makedirs(dir, exist_ok=True) for dir in sdo_dirs]
[os.makedirs(dir, exist_ok=True) for dir in stereo_a_dirs]
[os.makedirs(dir, exist_ok=True) for dir in stereo_b_dirs]

# copy files to new folder
[shutil.copy(f, f.replace('/mnt/data/sdo_jsoc', '/mnt/data/aligned/SDO')) for f in np.ravel(sdo_files)]
[shutil.copy(f, f.replace('/mnt/data/stereo_iti_converted', '/mnt/data/aligned/STEREO_A')) for f in
 np.ravel(stereo_a_files)]
[shutil.copy(f, f.replace('/mnt/data/stereo_iti_converted', '/mnt/data/aligned/STEREO_B')) for f in
 np.ravel(stereo_b_files)]
