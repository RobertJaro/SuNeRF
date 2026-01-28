import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.map import Map
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.data.loader.base_loader import MapDataLoader
from sunerf.evaluation.loader import ThomsonSuNeRFLoader

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Visualize CME')
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--ref_map', type=str, required=False, help='Path to reference map')
    parser.add_argument('--out_path', type=str, help='Path to output directory', default=None)

    args = parser.parse_args()

    # set default path
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'ref_map')
    os.makedirs(args.out_path, exist_ok=True)

    ##########################################################
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    seconds_per_dt = sunerf_loader.seconds_per_dt
    ref_date = sunerf_loader.ref_date
    observers = sunerf_loader.observers

    ref_map = Map(args.ref_map)

    model_out = sunerf_loader.load_map(ref_map, progress=False)

    tB_map = model_out['tB_map']
    pB_map = model_out['pB_map']
    density_map = model_out['density_map']
    ##########################################################

    fig, axs = plt.subplots(1, 4, figsize=(15, 5), subplot_kw={'projection': tB_map})

    ax = axs[0]
    im = ax.imshow(ref_map.data, cmap=cm.soholasco2, norm='log', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax)
    ax.set_title('Polarized Brightness (Reference)')
    tB_map.draw_grid(ax, color='blue')

    ax = axs[1]
    im = ax.imshow(tB_map.data, cmap=cm.soholasco2, norm='log', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax)
    ax.set_title('Total Brightness')
    tB_map.draw_grid(ax, color='blue')

    ax = axs[2]
    im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm='log', origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax)
    ax.set_title('Polarized Brightness')
    tB_map.draw_grid(ax, color='blue')

    ax = axs[3]
    im = ax.imshow(density_map.data, cmap='inferno', origin='lower', norm='log')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax)
    ax.set_title('Density')
    tB_map.draw_grid(ax, color='blue')

    fig.tight_layout()
    fig.savefig(os.path.join(args.out_path, f"ref_map_visualization.png"), dpi=300, transparent=True)
    plt.close('all')
