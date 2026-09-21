import numpy as np
from astropy.io import fits

from sunerf.data.coronagraph.prep_punch_pam import PunchPamPrep


def test_punch_pb_masks_nonpositive_pb_and_clips_nonpositive_pbp(tmp_path):
    data = np.ones((3, 2, 3), dtype=np.float32)
    data[0] = 10.0
    data[1] = [[3.0, -3.0, 0.0], [3.0, 3.0, 3.0]]
    data[2] = [[4.0, 4.0, 4.0], [-4.0, 0.0, 4.0]]
    uncertainty = np.ones_like(data)

    path = tmp_path / 'pam.fits'
    data_hdu = fits.ImageHDU(data)
    data_hdu.header.update({
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CRPIX1': 2.0,
        'CRPIX2': 1.5,
        'CRVAL1': 0.0,
        'CRVAL2': 0.0,
        'CDELT1': 1.0,
        'CDELT2': 1.0,
        'DATE-OBS': '2025-09-01T00:00:00',
    })
    fits.HDUList([
        fits.PrimaryHDU(),
        data_hdu,
        fits.ImageHDU(uncertainty),
    ]).writeto(path)

    tb_map, pb_map = PunchPamPrep._load_punch_pam_maps(path)

    np.testing.assert_array_equal(tb_map.data, np.full((2, 3), 10.0))
    # pB <= 0 is missing; pB' <= 0 is clipped to zero before the quadrature sum.
    expected = np.array([[5.0, np.nan, np.nan], [3.0, 3.0, 5.0]])
    np.testing.assert_allclose(pb_map.data, expected, equal_nan=True)
