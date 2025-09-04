import glob
import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sunpy.map import Map

out_path = '/glade/work/rjarolim/data/sunerf/2023_04/prep/imgs_aia_fd'
files = sorted(glob.glob('/glade/work/rjarolim/data/sunerf/2023_04/prep/aia_fd/*.fits'))

# out_path = '/glade/work/rjarolim/data/sunerf/2012_01_prep/imgs_EUVI_A'
# files = sorted(glob.glob('/glade/work/rjarolim/data/sunerf/2012_01_prep/euvi/*A.fts'))

# out_path = '/glade/work/rjarolim/data/sunerf/2012_01_prep/imgs_EUVI_B'
# files = sorted(glob.glob('/glade/work/rjarolim/data/sunerf/2012_01_prep/euvi/*B.fts'))

os.makedirs(out_path, exist_ok=True)

norm = LogNorm(vmin=1, vmax=1e4)

def _plot(f):
    s_map = Map(f)
    channel = s_map.meta['WAVELNTH']
    date_obs = s_map.meta['DATE-OBS']
    out_file = os.path.join(out_path, f'{channel}_{date_obs}.jpg')
    if os.path.exists(out_file):
        print(f'Skipping {out_file}, already exists.')
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(s_map.data, cmap='gray', norm=norm)
    ax.set_title(f'{s_map.meta["WAVELNTH"]} - {s_map.meta["DATE-OBS"]}')
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

with Pool(os.cpu_count()) as p:
    p.map(_plot, files)