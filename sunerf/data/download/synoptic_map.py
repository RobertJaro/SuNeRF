from astropy.io import fits
from astropy.utils.data import download_file
from sunpy.map import Map

# import astropy.units as u


# Databases for synoptic maps
nasa_database = 'NASA'
gmu_database = 'GMU'
oulu_database = 'Oulu'
# SDO/AIA wavelengths (Angstroms)
sdoaia171 = 171
sdoaia193 = 193
sdoaia211 = 211
sdoaia304 = 304
# Longitude/Latitude range (degrees)
max_lg = 360.
max_lt = 180.


def download_synoptic_map(syn_car_nb, syn_database=nasa_database, syn_wavelength=sdoaia193):
    """Download synoptic map and create Sunpy Map object.

    Args:
        syn_database: Database from which to download synoptic maps.
                      Options: NASA, GMU, Oulu.
        syn_wavelength: Wavelength of the observations.
                      Options: 171, 193, 211, and 304 Angstroms.
        syn_car_nb: Carrington rotation number.

    Returns:
        sunpy.map.Map: synoptic Map with fixed header.
    """

    # Choice of database:
    if syn_database == nasa_database:
        syn_url = 'https://sdo.gsfc.nasa.gov/assets/img/synoptic/AIA0' + str(syn_wavelength) + '/CR' + str(
            syn_car_nb) + '.fits'
    elif syn_database == gmu_database:
        syn_url = 'http://spaceweather.gmu.edu/projects/synop/AIA' + str(syn_wavelength) + 'fits/AIA' + str(
            syn_wavelength) + 'fitsnew/CR' + str(syn_car_nb) + '.fits'
    elif syn_database == oulu_database:
        syn_url = 'http://satdat.oulu.fi/solar_data/Synoptic_Map_EIT_AIA/EIT_AIA_Synop_Maps/AIA/' + str(
            syn_wavelength) + 'A/' + str(syn_wavelength) + 'A_Lat/' + str(syn_wavelength) + 'A_Lat_fits/AIA' + str(
            syn_wavelength) + '_synop_CR' + str(syn_car_nb) + '.fits'

    # Download synoptic map:
    syn_filename = download_file(syn_url, cache=True)

    # Extract header/data from .fits file:
    f = fits.open(syn_filename, memmap=False)
    f.verify('fix')
    # Load header
    syn_header = f[0].header
    # Load data
    syn_data = f[0].data
    f.close()

    # Basic fixes to header information
    syn_header['cunit1'] = 'deg'
    syn_header['cunit2'] = 'deg'
    syn_header['WAVELNTH'] = syn_wavelength
    syn_header['CDELT1'] = max_lg / syn_data.shape[1]
    syn_header['CDELT2'] = max_lt / syn_data.shape[0]

    # Additional fixes may be required depending on the database:
    if syn_database == oulu_database:
        syn_header['CTYPE1'] = 'Carrington Time'
        syn_header['CTYPE2'] = 'Latitude'

    # Create Sunpy object:
    syn_map = Map(syn_data, syn_header)

    return syn_map


# def download_disk_map(start_time, end_time, instr, wavelenth, smple):
'''
if __name__ == '__main__':
    syn_map =  download_synoptic_map(2105, syn_database=nasa_database, syn_wavelength=sdoaia193)
'''
