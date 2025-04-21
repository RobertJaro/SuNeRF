import argparse
import multiprocessing
import os
from glob import glob
from itertools import repeat

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io.fits import getheader, getdata
from sunpy.map import Map

from sunerf.data.utils import get_azimuthal_equidistant_coordinates


def _load_HAO(file_path):
    """
    20 solar radii (0.1 AU) -->
    Dynamic model
    CMEs included
    """
    # assumes dsun is 1AU
    data, header = getdata(file_path), getheader(file_path)

    # initialise
    header['cunit1'] = 'arcsec'
    header['cunit2'] = 'arcsec'

    header['HGLN_OBS'] = np.rad2deg(header["OBS_LON"])  # rad to deg
    header['HGLT_OBS'] = 90 - np.rad2deg(header["OBS_LAT"])

    header['DSUN_OBS'] = (header["OBS_R0"] * u.Rsun).to("m").value  # solar radii to m
    header['RSUN'] = np.rad2deg(header["RSUN"]) * 3600  # rad to arcsec

    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"

    header["CDELT1"] = np.rad2deg(header["CDELT1"]) * 3600  # rad -- arcsec
    header["CDELT2"] = np.rad2deg(header["CDELT2"]) * 3600  # rad -- arcsec

    # manual set of centre value to 0
    header['CRVAL1'] = 0
    header['CRVAL2'] = 0

    header["CRPIX1"] = header["NAXIS1"] * 0.5 + 0.5
    header["CRPIX2"] = header["NAXIS2"] * 0.5 + 0.5

    header['wavelnth'] = 5200  # dummy value for wavelength
    header['rsun_ref'] = (1 * u.R_sun).to_value(u.m)  # add solar radius

    return Map(data, header)


def _loadMLprepMap(file_path, out_path, resolution, occ_rad=0.1 * u.AU, outer_rad=130 * u.R_sun):
    """Load and preprocess PSI file.

    Parameters
    ----------
    file_path: path to the FITS file.
    resolution: target resolution in pixels.

    Returns
    -------
    the preprocessed SunPy Map
    """
    # load Map
    s_map = _load_HAO(file_path)

    # adjust image size
    s_map = s_map.resample((resolution, resolution) * u.pix)

    # mask occultor
    img_coords = get_azimuthal_equidistant_coordinates(s_map)
    x = img_coords[..., 0]
    y = img_coords[..., 1]
    solar_center = SkyCoord(0 * u.deg, 0 * u.deg, frame=s_map.coordinate_frame)

    pixel_radii = np.sqrt((x - solar_center.Tx) ** 2 + (y - solar_center.Ty) ** 2)

    mask = ((pixel_radii < s_map.rsun_obs * occ_rad.to_value(u.R_sun)) |
            (pixel_radii > s_map.rsun_obs * outer_rad.to_value(u.R_sun)))

    data = s_map.data
    data[mask] = np.nan

    data = data.astype(np.float32)

    s_map = Map(data, s_map.meta)
    dir_name = file_path.split(os.sep)[-2]
    s_map.save(os.path.join(out_path, dir_name + '_' + os.path.basename(file_path)), overwrite=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('--hao_path', type=str,
                   default='/mnt/ground-data/data_fits/**/*.fits',
                   help='search path for AIA maps.')
    p.add_argument('--resolution', type=float,
                   default=512,
                   help='target image scale in arcsec per pixel.')
    p.add_argument('--output_path', type=str,
                   default='/mnt/ground-data/prep_HAO',
                   help='path to save the converted maps.')
    p.add_argument('--check_matching', dest='check_matching', action='store_true')
    args = p.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # Load paths
    hao_paths = sorted(glob(args.hao_path))

    if args.check_matching:
        common_tB_fnames = []
        common_pB_fnames = []

        # HAO: more pB files than tBs
        s_maps_pB = [p for p in hao_paths if 'pB' in p]
        s_maps_tB = [p for p in hao_paths if 'tB' in p]

        for fname_pB in s_maps_pB:
            corresponding_fname_tB = str(fname_pB).replace("pB", "tB")
            if corresponding_fname_tB in s_maps_tB:
                common_tB_fnames.append(corresponding_fname_tB)
                common_pB_fnames.append(fname_pB)

        hao_paths = common_pB_fnames + common_tB_fnames

    assert len(hao_paths) > 0, 'No files found.'

    # Load maps
    with multiprocessing.Pool(os.cpu_count()) as p:
        p.starmap(_loadMLprepMap, zip(hao_paths, repeat(args.output_path), repeat(args.resolution)))
