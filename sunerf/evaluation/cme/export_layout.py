"""File layout of the SuNeRF exports read by the evaluation plots.

::

    <export>/density/rho000050.npy              export_density
    <export>/coordinates/rho000050.npz          export_density
    <export>/images/<view>/pb/<view>_pb0150.npy export_images
    <export>/cme_series.npz                     export_cme_series
"""
import os
from pathlib import Path

SERIES_FILE = 'cme_series.npz'


def view_frame(text):
    """Parse ``<view>=<frame>`` (view: instrument key of the save state)."""
    view, frame = text.rsplit('=', 1)
    return view, int(frame)


def labelled_path(text):
    """Parse ``<label>=<path>``."""
    label, path = text.split('=', 1)
    return label, Path(path)


def dump_name(dump):
    """Name of the PSI cube of a simulation dump."""
    return f'rho{int(dump):06d}'


def density_file(export_path, name):
    return os.path.join(export_path, 'density', f'{name}.npy')


def coordinates_file(export_path, name):
    return os.path.join(export_path, 'coordinates', f'{name}.npz')


def image_file(export_path, view, product, frame):
    """Exported image of the reference file ``<view>/<product>/*<frame>.fts``."""
    return os.path.join(export_path, 'images', view, product, f'{view}_{product}{frame:04d}.npy')
