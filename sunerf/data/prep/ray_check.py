import numpy as np
from matplotlib import pyplot as plt
from sunpy.map import Map, all_coordinates_from_map

from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from astropy import units as u

if __name__ == '__main__':
    s_map = Map('/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/prep_HAO_2/dcmer_020W_bang_0000_pB_stepnum_005.fits')

    a_coords = all_coordinates_from_map(s_map)
    ax, ay = a_coords.Tx.to_value(u.deg), a_coords.Ty.to_value(u.deg)
    b_coords = get_azimuthal_equidistant_coordinates(s_map)
    bx, by = b_coords[..., 0].to_value(u.deg), b_coords[..., 1].to_value(u.deg)

    v_min_max = np.nanmax(np.abs(bx))

    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    im = axs[0, 0].imshow(ax, vmin=-v_min_max, vmax=v_min_max, cmap='seismic')
    axs[0, 0].set_title('SunPy')
    fig.colorbar(im, ax=axs[0, 0])

    im = axs[0, 1].imshow(bx, vmin=-v_min_max, vmax=v_min_max, cmap='seismic')
    axs[0, 1].set_title('SuNeRf')
    fig.colorbar(im, ax=axs[0, 1])

    im = axs[1, 0].imshow(ay, vmin=-v_min_max, vmax=v_min_max, cmap='seismic')
    axs[1, 0].set_title('SunPy')
    fig.colorbar(im, ax=axs[1, 0])

    im = axs[1, 1].imshow(by, vmin=-v_min_max, vmax=v_min_max, cmap='seismic')
    axs[1, 1].set_title('SuNeRf')
    fig.colorbar(im, ax=axs[1, 1])

    fig.tight_layout()
    fig.savefig('/glade/work/rjarolim/data/sunerf-cme/hao/prep-data-v2/demo.jpg', dpi=300)
    plt.close(fig)

    print('---------------------')
    print(f'SunPy-X: {np.nanmin(ax)}, {np.nanmax(ax)}')
    print(f'SuNeRf-X: {np.nanmin(bx)}, {np.nanmax(bx)}')
    print(f'------------------------')
    print(f'SunPy-Y: {np.nanmin(ay)}, {np.nanmax(ay)}')
    print(f'SuNeRf-Y: {np.nanmin(by)}, {np.nanmax(by)}')