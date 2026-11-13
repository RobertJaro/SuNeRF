import os
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


def test_training_cache_interleaves_frames_and_tiles_times(tmp_path):
    source = tmp_path / 'series'
    source.mkdir()
    for index in range(3):
        tB = np.full((4, 4), 2.0e-9 + index * 1.0e-10)
        pB = np.full((4, 4), 1.0e-9)
        tB[1, index] = -1.0e-9  # non-positive in tB only: the row stays valid
        tB[0, index] = -1.0e-9
        pB[0, index] = 0.0  # both channels invalid: the row is dropped
        header_time = f'2025-09-01T0{index}:00:00'
        for name, data in (('tB', tB), ('pB', pB)):
            header = fits.Header({
                'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec',
                'CDELT1': 2000.0, 'CDELT2': 2000.0, 'CRPIX1': 2.5, 'CRPIX2': 2.5,
                'CRVAL1': 0.0, 'CRVAL2': 0.0, 'DATE-OBS': header_time, 'DSUN_OBS': 1.5e11,
                'HGLN_OBS': 0.0, 'HGLT_OBS': 0.0, 'RSUN_REF': 6.957e8,
            })
            fits.writeto(source / f'{name}_{index}.fits', data.astype(np.float32), header)
    dataset = COR2Dataset(
        data_path_tB=str(source / 'tB_*.fits'), data_path_pB=str(source / 'pB_*.fits'),
        ds_key='view', instrument_key='view', Rs_per_ds=100, seconds_per_dt=3600,
        work_directory=str(source / 'cache'), log_data_overview=False, batch_size=3,
    )
    try:
        assert sorted(path.name for path in (source / 'cache').iterdir()) == sorted(
            os.path.basename(path) for path in dataset.batches_file_paths.values()
        )
        time = np.load(dataset.batches_file_paths['time'])[:, 0]
        image = np.load(dataset.batches_file_paths['image'])
        assert len(time) == 3 * 15  # one dropped pixel per frame, equal slots per frame
        # Every batch of three rows holds one random pixel of every frame.
        assert all(sorted(group) == [0.0, 1.0, 2.0] for group in time.reshape(-1, 3))
        assert np.isfinite(image).any(axis=-1).all()
        for frame in range(3):
            frame_tB = image[time == frame, 0]
            assert np.count_nonzero(np.isnan(frame_tB)) == 1
            np.testing.assert_allclose(frame_tB[np.isfinite(frame_tB)], 2.0 + frame * 0.1)
        batch = dataset[0]
        assert sorted(batch['time'][:, 0].tolist()) == [0.0, 1.0, 2.0]
    finally:
        dataset.clear()
