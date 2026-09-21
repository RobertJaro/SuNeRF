import numpy as np
from astropy.io import fits

from sunerf.data.loader.thomson_instrument import COR2Dataset, PSICMEDataset


def _write(path, data):
    header = fits.Header({
        'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
        'CDELT1': 2000.0, 'CDELT2': 2000.0, 'CRPIX1': 2.5, 'CRPIX2': 2.5, 'CRVAL1': 0.0, 'CRVAL2': 0.0,
        'DATE-OBS': '2025-09-01T00:00:00', 'DSUN_OBS': 1.5e11, 'HGLN_OBS': 0.0, 'HGLT_OBS': 0.0,
        'RSUN_REF': 6.957e8,
    })
    fits.writeto(path, data.astype(np.float32), header)


def _images(dataset_class, tmp_path):
    source = tmp_path / dataset_class.__name__
    source.mkdir()
    tB = np.full((4, 4), 2.0e-9)
    pB = np.full((4, 4), 1.0e-9)
    tB[0, 0] = -1.0e-9
    pB[1, 1] = 0.0
    _write(source / 'tB.fits', tB)
    _write(source / 'pB.fits', pB)
    dataset = dataset_class(
        data_path_tB=str(source / 'tB.fits'), data_path_pB=str(source / 'pB.fits'),
        ds_key='view', instrument_key='view', Rs_per_ds=100, seconds_per_dt=86400,
        work_directory=str(source / 'cache'), shuffle=False, filter_nans=False,
        log_data_overview=False,
    )
    try:
        return np.concatenate([dataset[i]['image'].numpy() for i in range(len(dataset))])
    finally:
        dataset.clear()


def test_observational_datasets_mask_non_positive_brightness_per_channel(tmp_path):
    image = _images(COR2Dataset, tmp_path).reshape(4, 4, 2)
    assert np.isnan(image[0, 0, 0]) and np.isfinite(image[0, 0, 1])
    assert np.isnan(image[1, 1, 1]) and np.isfinite(image[1, 1, 0])
    assert np.count_nonzero(np.isnan(image)) == 2


def test_synthetic_psi_dataset_keeps_non_positive_brightness(tmp_path):
    image = _images(PSICMEDataset, tmp_path)
    assert np.isfinite(image).all()
    assert np.count_nonzero(image <= 0) == 2
