import logging
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, AsinhStretch
from astropy.wcs import WCS
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from sunpy.coordinates import frames
from sunpy.map import Map, make_fitswcs_header
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

import sys, os


def create_new_observer(s_map, lat, lon, distance = 1. * u.AU):
    """Create new observer FITS header.

    Args:
        s_map (sunpy.map.Map): reference SunPy Map.
        lat (astropy.unit): latitude in Carrington frame.
        lon (astropy.unit): longitude in Carrington frame.
        distance (astropy.unit, optional): distance from Sun. Defaults to 1.*u.AU.

    Returns:
        sunpy.util.MetaDict: FITS header.
    """
    # define new observer
    # stonyhurst is earth-centered perspective
    new_observer = SkyCoord(lon, lat, distance, obstime=s_map.date, frame='heliographic_stonyhurst') 
    # (longitude of sun from earth's frame,latitude of sun from earth's frame, distance from earth, observer time, coordinate frame)

    out_shape = s_map.data.shape
    out_ref_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=new_observer.obstime,
                             frame='helioprojective', observer=new_observer,
                             rsun=s_map.coordinate_frame.rsun)
    out_header = make_fitswcs_header(  # generates new fits file with wcs
        out_shape,
        out_ref_coord,
        scale=u.Quantity(s_map.scale),
        rotation_matrix=s_map.rotation_matrix,
        instrument=s_map.instrument,
        wavelength=s_map.wavelength
    )

    return out_header

def create_heliographic_map(*maps, shape_out=(1024, 2048), obstime=None, synoptic=False):
    """If synoptic == true. Make a synoptic map that can be reprojected by sunpy.
       If synoptic == false. Project any number of SunPy maps to the heliographic frame.

    Parameters
    ----------
    *maps: List Sunpy Map objects
        Sunpy map object of a synoptic map or full-disk maps to merge.
    shape_out : int tuple
        desired output shape in pixels
    obstime: str
        Date used to construct the output map
    synoptic: bool
        Whether to use the synoptic or synchronic functionality

    Outputs
    -------
    outMap:  Sunpy Map object
        Sunpy heliographic map
    """
    if synoptic:
        shape_out = maps[0].data.shape
    else:
        shape_out = shape_out

    if obstime is None:
        obstime = maps[0].date
    coord = SkyCoord(0, 0, unit=u.deg, frame="heliographic_carrington", observer='earth', obstime=obstime)
    scale = [360 / shape_out[1], 180 / shape_out[0]] * u.deg / u.pix
    header = make_fitswcs_header(shape_out, coord, scale=scale, projection_code="CAR")

    if synoptic:
        return Map(maps[0].data, header)

    out_wcs = WCS(header)
    # create map
    array, footprint = reproject_and_coadd(maps, out_wcs, shape_out,
                                           reproject_function=reproject_interp, match_background=False)
    if np.isnan(array).mean() > 0.5:
        logging.warning('More than 50 percent of the heliographic map are NaNs!')
    array = np.nan_to_num(array, nan=np.nanmean(array))
    outmap = Map((array, header))
    
    return outmap

def transform(*maps, lat, lon, distance, obstime=None, synoptic=False):
    """Transform maps to new viewpoint.

    Args:
        lat (astropy.unit): latitude.
        lon (astropy.unit): longitude.
        distance (astropy.unit): distance to solar center.
        obstime (string): Time of observation. Use map object value or insert value.
        synoptic (bool): Specify if input data is a synoptic map or not.

    Returns:
        sunpy.map.Map: Map at new observer location.
    """
    # Check for observation time and create map
    if obstime is None:
        h_map = create_heliographic_map(*maps, synoptic=synoptic)
    else:
        h_map = create_heliographic_map(*maps, obstime=obstime, synoptic=synoptic)

    # Create new observer
    observer = create_new_observer(maps[0], lat, lon, distance)
    # Project
    sdo_new_view = h_map.reproject_to(observer)
    # fix header information
    sdo_new_view.meta['wavelnth'] = maps[0].wavelength.value

    # Return reprojected map
    return sdo_new_view

def load_views(*maps, n_workers=None, strides=10, resample=False):
    """Generate Maps from sampled viewpoints.

    Args:
        maps: list of input maps.
        n_workers: number of parallel threads. If None, workers will be set to maximum.
        strides: sampling in degrees.
        resample: optional resampling.

    Yields:
        (float, float), sunpy.map.Map: pair of viewpoint and Map.
    """
    coords = np.stack(np.mgrid[-90:91:strides, :361:strides], -1).astype(np.float32)
    coords = coords.reshape((-1, 2))
    h_map = create_heliographic_map(*maps)

    # parallel run reprojections
    inp_data = [(c, maps[0], h_map) for c in coords]
    n_workers = n_workers if n_workers is not None else os.cpu_count()
    with multiprocessing.Pool(n_workers) as p:
        for (lat, lon), s_map in zip(coords, p.imap(_process, inp_data)):
            if resample:
                s_map = s_map.resample(resample)
            coord = s_map.center.transform_to(frames.HeliographicCarrington)
            yield [(lat, lon), s_map]

def _process(d):
    """Helper function for parallel execution.

    Args:
        d: (lat, lon), ref_map, heliographic_map

    Returns:
        sunpy.map.Map: reprojected Map
    """
    (lat, lon), ref_map, heliographic_map = d
    observer = create_new_observer(ref_map, lat * u.deg, lon * u.deg)
    new_view = heliographic_map.reproject_to(observer)
    new_view.meta['wavelnth'] = ref_map.wavelength.value
    return new_view

if __name__ == '__main__':
    # Example view for synchronic maps

    sdo_path = "/mnt/data/SDO/AIA/aia_lev1_304a_2011_01_01t00_00_08_12z_image_lev1.fits"
    iti_a_path = "/mnt/data/stereo_iti_converted/304/2011-01-01T00:00:00_A.fits"
    iti_b_path = "/mnt/data/stereo_iti_converted/304/2011-01-01T00:00:00_B.fits"

    save_path = '/home/robert_jarolim/4piuvsun/results/reprojection_overview.jpg'

    # prevent circular import
    from sunerf.data.utils import loadAIAMap
    sdo_map = loadAIAMap(sdo_path)
    iti_a_map = Map(iti_a_path)
    iti_b_map = Map(iti_b_path)

    s_map = transform(sdo_map, iti_a_map, iti_b_map, lat=0 * u.deg, lon=0 * u.deg, distance=1 * u.AU)

    views = load_views(sdo_map, iti_a_map, iti_b_map, resample=(256, 256) * u.pix, strides=60)

    norm = ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True)
    cmap = cm.sdoaia193

    plt.figure(figsize=(6 * 5, 3 * 5))
    for i, (coords, s_map) in tqdm(enumerate(views), total=3 * 6):
        ax = plt.subplot(3, 6, i + 1, projection=s_map)
        s_map.plot(norm=norm, cmap=cmap)
        plt.title('lat %d lon %d' % coords)
        s_map.draw_grid()
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
    plt.close()

    '''
    #Example view for synoptic maps

    syn_database = nasa_database
    syn_wavelength = sdoaia193
    syn_car_nb = #carrington rotation number 

    save_path = "../../results/"
    
    syn_map = download_synoptic_map(syn_car_nb, syn_database = nasa_database, syn_wavelength=sdoaia193)
    
    x_resolution = 256
    y_resolution = 256

    views = load_views(syn_map, resample(x_resolution,y_resolution)*u.pix, strides = 60)
    
    norm = ImageNormalize(vmin=0, vmax=8800, stretch=AsinhStretch(0.001), clip=True)
    cmap = cm.sdoaia193

    plt.figure(figsize=(6 * 5, 3 * 5))
    for i, (coords, s_map) in tqdm(enumerate(views), total=3 * 6):
        ax = plt.subplot(3, 6, i + 1, projection=s_map)
        s_map.plot(norm=norm, cmap=cmap)
        plt.title('lat %d lon %d' % coords)
        s_map.draw_grid()
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
    plt.close()
    '''






