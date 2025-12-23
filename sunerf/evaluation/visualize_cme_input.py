import glob
import os.path

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from astropy import units as u

files = sorted(glob.glob('/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_040_100/*.fits'))

norm = LogNorm(vmin=1e-7, vmax=2e-5)

for file in tqdm(files):
    s_map = Map(file)
    coords = get_azimuthal_equidistant_coordinates(s_map)
    img = s_map.data

    print(s_map.heliographic_latitude, s_map.heliographic_longitude)

    extent = [coords[..., 0].min().to_value(u.deg), coords[..., 0].max().to_value(u.deg),
              coords[..., 1].min().to_value(u.deg), coords[..., 1].max().to_value(u.deg)]
    cmap = cm.soholasco2.copy()

    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    im = ax.imshow(img, norm=norm, cmap=cmap, origin='lower', extent=extent)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    label = 'Total Brightness' if 'tB' in os.path.basename(file) else 'Polarized Brightness'
    cbar.set_label(f'{label} [DN]', rotation=270, labelpad=15)

    ax.set_xlabel('Helioprojective Longitude [deg]')
    ax.set_ylabel('Helioprojective Latitude [deg]')
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)

    fig.savefig(f'/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/{os.path.basename(file)}.png', dpi=300, bbox_inches='tight',
                transparent=True)
    plt.close(fig)
