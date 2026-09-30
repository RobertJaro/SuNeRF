from datetime import datetime

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header

from sunerf.data.euv.prepare import (
    GeometryConfig,
    PREPARED_EUV_SCHEMA,
    _bounded_workers,
    apply_common_geometry,
    prepare_aia_map,
    prepare_eui_map,
    _exposure_seconds,
    prepare_euv_files,
    prepare_euvi_map,
)


def _euv_map(
    data,
    *,
    instrument='EUI',
    channel=174,
    bunit='DN / s',
    level='L2',
    calibration_id='cal-v1',
    spacecraft=None,
):
    obstime = datetime(2024, 1, 2, 3, 4, 5)
    observer = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        1 * u.AU,
        frame=frames.HeliographicStonyhurst,
        obstime=obstime,
    )
    reference = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        frame=frames.Helioprojective,
        observer=observer,
        obstime=obstime,
    )
    header = make_fitswcs_header(
        np.asarray(data).shape,
        reference,
        scale=np.array([2.0, 2.0]) * u.arcsec / u.pix,
    )
    header['instrume'] = instrument
    header['wavelnth'] = channel
    header['waveunit'] = 'angstrom'
    header['bunit'] = bunit
    header['exptime'] = 2.0
    header['level'] = level
    header['cal_id'] = calibration_id
    if instrument == 'EUI':
        header['telescop'] = 'Solar Orbiter'
    elif instrument == 'AIA':
        header['telescop'] = 'SDO'
    if spacecraft:
        header['obsrvtry'] = f'STEREO-{spacecraft}'
    return Map(np.asarray(data, dtype=np.float32), header)


def test_eui_file_writes_strict_lossless_product(tmp_path):
    data = np.array([[3.0, -2.0], [np.nan, 0.0]], dtype=np.float32)
    source = tmp_path / 'eui_l2.fits'
    _euv_map(data).save(source)
    results = prepare_euv_files(
        [source],
        tmp_path / 'prepared',
        adapter='eui',
        calibration_id='cal-v1',
        sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False),
        workers=64,
    )

    assert len(results) == 1
    output = results[0].output_path
    with fits.open(output) as hdul:
        assert hdul[0].header['PREPSCHM'] == PREPARED_EUV_SCHEMA
        assert hdul[0].header['SCHEMAV'] == 2
        assert 'RESP_ID' not in hdul[0].header
        assert hdul[0].header['RADSEM'] == 'per_native_pixel'
        assert hdul[0].header['CAL_ID'] == 'cal-v1'
        assert len(hdul[0].header['SRC_SHA']) == 64
        assert hdul[0].header['BUNIT'] == 'DN / s'
        np.testing.assert_allclose(hdul[0].data[0], [3.0, -2.0])
        assert np.isnan(hdul[0].data[1, 0])
        assert hdul['VALID_MASK'].data.tolist() == [[1, 1], [0, 1]]

    assert not list((tmp_path / 'prepared').glob('*.jsonl'))


def test_eui_rejects_non_l2_and_never_exposure_normalizes():
    raw = _euv_map([[4.0]], level='L1')
    with pytest.raises(ValueError, match='level 2'):
        prepare_eui_map(
            raw,
            calibration_id='cal-v1',
            sensitivity_convention='reference_epoch',
        )

    calibrated = _euv_map([[4.0]], level='L2')
    prepared, mask, provenance = prepare_eui_map(
        calibrated,
        calibration_id='cal-v1',
        sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False),
    )
    assert prepared.data.item() == 4.0
    assert mask.item()
    assert provenance['calibration_steps'] == [
        'validated_EUI_L2',
        'geometry_only_in_SuNeRF',
    ]


def test_eui_accepts_solar_orbiter_xposure_and_header_calibration_id():
    """SOAR level-2 products carry XPOSURE and CAL_ID, never EXPTIME."""
    calibrated = _euv_map([[4.0]], level='L2', calibration_id='cal-v9')
    del calibrated.meta['exptime']
    calibrated.meta['xposure'] = 10.0

    prepared, _, provenance = prepare_eui_map(
        calibrated,
        sensitivity_convention='static_assumed',
        geometry=GeometryConfig(north_up=False),
    )
    # Exposure is validated but never divided out for EUI.
    assert prepared.data.item() == 4.0
    assert provenance['calibration_id'] == 'cal-v9'

    # Reprojection to a fresh WCS must not strip the exposure keyword, which
    # the prepared-product reader requires.
    reprojected = _euv_map(np.ones((64, 64), dtype=np.float32), level='L2')
    del reprojected.meta['exptime']
    reprojected.meta['xposure'] = 10.0
    prepared, _, _ = prepare_eui_map(
        reprojected,
        sensitivity_convention='static_assumed',
        geometry=GeometryConfig(shape=(32, 32), hpc_bounds_arcsec=(-1560, -1560, 1560, 1560)),
    )
    assert float(prepared.meta['xposure']) == 10.0


def test_exposure_keyword_precedence_and_errors():
    both = _euv_map([[4.0]], level='L2')
    both.meta['exptime'] = 2.0
    both.meta['xposure'] = 99.0
    assert _exposure_seconds(both) == 2.0

    neither = _euv_map([[4.0]], level='L2')
    del neither.meta['exptime']
    with pytest.raises(ValueError, match='EXPTIME/XPOSURE'):
        _exposure_seconds(neither)

    non_positive = _euv_map([[4.0]], level='L2')
    del non_positive.meta['exptime']
    non_positive.meta['xposure'] = -1.0
    with pytest.raises(ValueError, match='XPOSURE must be positive'):
        _exposure_seconds(non_positive)


def test_euvi_requires_external_calibration_attestation_and_spacecraft_match():
    source = _euv_map(
        [[5.0]], instrument='SECCHI/EUVI', channel=195,
        spacecraft='B', level='SECCHI_PREP_L1',
    )
    with pytest.raises(ValueError, match='does not positively identify STEREO-A'):
        prepare_euvi_map(
            source,
            spacecraft='A',
            product_level='SECCHI_PREP_L1',
            sensitivity_convention='reference_epoch',
        )
    with pytest.raises(ValueError, match='documented SECCHI'):
        prepare_euvi_map(
            source,
            spacecraft='B',
            product_level='unknown',
            sensitivity_convention='reference_epoch',
        )

    photon_flux = _euv_map(
        [[5.0]], instrument='SECCHI/EUVI', channel=195, bunit='PhotonFlux',
        spacecraft='B', level='SECCHI_PREP_L1',
    )
    prepared, _, _ = prepare_euvi_map(
        photon_flux,
        spacecraft='B',
        product_level='SECCHI_PREP_L1',
        sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False),
    )
    assert prepared.meta['bunit'] == 'ph / s'


def test_aia_uses_pinned_calibration_in_documented_order_without_clipping(
    tmp_path, monkeypatch
):
    correction = tmp_path / 'correction.tbl'
    pointing = tmp_path / 'pointing.ecsv'
    correction.write_text('pinned correction input')
    pointing.write_text('pinned pointing input')
    source = _euv_map(
        [[2.0, -2.0], [np.nan, 4.0]],
        instrument='AIA',
        channel=171,
        bunit='DN',
        level='L1',
    )
    source.meta['quality'] = 0
    calls = []
    pinned_pointing = object()

    def update_pointing(s_map, *, pointing_table):
        assert pointing_table is pinned_pointing
        calls.append('pointing')
        return s_map

    def register(s_map, *, missing, order):
        calls.append(f'register-{order}')
        if order == 3:
            assert np.isnan(missing)
        else:
            assert missing == 0
        return Map(np.roll(s_map.data, 1, axis=1), s_map.meta)

    def correct_degradation(s_map, *, correction_table):
        assert correction_table.meta['fixture'] == 'pinned-correction'
        calls.append('degradation')
        return Map(s_map.data * 2, s_map.meta)

    monkeypatch.setattr(
        'sunerf.data.euv.prepare._load_aiapy_calibration',
        lambda: (
            update_pointing,
            register,
            correct_degradation,
        ),
    )
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._load_pointing_table', lambda path: pinned_pointing
    )
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._cached_correction_table',
        lambda *args: QTable(meta={'fixture': 'pinned-correction'}),
    )

    prepared, mask, provenance = prepare_aia_map(
        source,
        correction_table_path=correction,
        pointing_table_path=pointing,
        geometry=GeometryConfig(north_up=False),
        valid_mask=np.array([[False, True], [True, True]]),
    )

    assert calls == [
        'pointing', 'pointing', 'register-3', 'register-0', 'degradation'
    ]
    assert prepared.data[0, 0] == -2.0
    assert np.isnan(prepared.data[0, 1])
    assert prepared.data[1, 0] == 4.0
    assert np.isnan(prepared.data[1, 1])
    # The explicitly invalid finite raw pixel moved with aiapy registration.
    assert mask.tolist() == [[True, False], [True, False]]
    assert prepared.meta['bunit'] == 'DN / s'
    assert provenance['correction_table_sha256']
    assert provenance['pointing_table_sha256']

    already_normalized = _euv_map(
        [[1.0]], instrument='AIA', channel=171, bunit='DN / s', level='L1'
    )
    already_normalized.meta['quality'] = 0
    with pytest.raises(ValueError, match='already rate-normalized'):
        prepare_aia_map(
            already_normalized,
            correction_table_path=correction,
            pointing_table_path=pointing,
        )


def test_aia_level1_without_bunit_is_interpreted_as_detector_dn(tmp_path, monkeypatch):
    source = _euv_map([[1.0]], instrument='AIA', channel=171, bunit='DN', level='L1')
    source.meta['quality'] = 0
    source.meta.pop('bunit')
    correction = tmp_path / 'correction.ecsv'
    pointing = tmp_path / 'pointing.ecsv'
    correction.write_text('fixture')
    pointing.write_text('fixture')

    monkeypatch.setattr(
        'sunerf.data.euv.prepare._load_aiapy_calibration',
        lambda: (
            lambda value, **kwargs: value,
            lambda value, **kwargs: value,
            lambda value, **kwargs: value,
        ),
    )
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._load_pointing_table', lambda path: 'pointing'
    )
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._cached_correction_table',
        lambda *args: QTable({'DATE': [], 'WAVELNTH': [], 'EFF_AREA': []}),
    )

    prepared, _, _ = prepare_aia_map(
        source,
        correction_table_path=correction,
        pointing_table_path=pointing,
        geometry=GeometryConfig(north_up=False),
    )
    assert prepared.meta['bunit'] == 'DN / s'


def test_worker_budget_is_bounded():
    assert _bounded_workers(32, 1000) == 32
    assert _bounded_workers(32, 5) == 5
    assert _bounded_workers(0, 5) == 1


def test_exact_hpc_grid_is_independent_of_input_pixel_rounding():
    with pytest.raises(ValueError, match='at least two pixels'):
        GeometryConfig(
            shape=(1, 5), hpc_bounds_arcsec=(-1.0, -1.0, 1.0, 1.0)
        )
    first = _euv_map(np.ones((8, 8), dtype=np.float32))
    shifted_meta = first.meta.copy()
    shifted_meta['crpix1'] += 0.37
    shifted_meta['crpix2'] -= 0.23
    second = Map(np.ones((8, 8), dtype=np.float32), shifted_meta)
    geometry = GeometryConfig(
        shape=(5, 7),
        hpc_bounds_arcsec=(-5.0, -4.0, 5.0, 4.0),
        north_up=False,
    )

    first_out, _ = apply_common_geometry(first, geometry=geometry)
    second_out, _ = apply_common_geometry(second, geometry=geometry)
    assert first_out.meta['bunit'] == first.meta['bunit']
    assert first_out.meta['instrume'] == first.meta['instrume']
    assert first_out.wavelength == first.wavelength
    y, x = np.indices(first_out.data.shape)
    first_grid = first_out.pixel_to_world(x * u.pix, y * u.pix)
    second_grid = second_out.pixel_to_world(x * u.pix, y * u.pix)

    np.testing.assert_allclose(
        first_grid.Tx.to_value(u.arcsec), second_grid.Tx.to_value(u.arcsec), atol=1e-9
    )
    np.testing.assert_allclose(
        first_grid.Ty.to_value(u.arcsec), second_grid.Ty.to_value(u.arcsec), atol=1e-9
    )
    np.testing.assert_allclose(
        first_grid.Tx[0, [0, -1]].to_value(u.arcsec), [-5.0, 5.0], atol=1e-6
    )
    np.testing.assert_allclose(
        first_grid.Ty[[0, -1], 0].to_value(u.arcsec), [-4.0, 4.0], atol=1e-6
    )



def _observer_map(distance_au, *, shape=(64, 64), scale_arcsec=40.0):
    """A disk-centered image whose detector covers +-1.28 solar radii."""
    obstime = datetime(2024, 1, 2, 3, 4, 5)
    observer = SkyCoord(
        0 * u.deg, 0 * u.deg, distance_au * u.AU,
        frame=frames.HeliographicStonyhurst, obstime=obstime,
    )
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec, frame=frames.Helioprojective,
        observer=observer, obstime=obstime,
    )
    header = make_fitswcs_header(
        shape, reference, scale=np.array([scale_arcsec, scale_arcsec]) * u.arcsec / u.pix
    )
    header['instrume'] = 'EUI'
    header['telescop'] = 'Solar Orbiter'
    header['wavelnth'] = 174
    header['waveunit'] = 'angstrom'
    header['bunit'] = 'DN / s'
    header['exptime'] = 2.0
    return Map(np.ones(shape, dtype=np.float32), header)


def _field_of_view_rsun(prepared):
    y, x = np.indices(prepared.data.shape)
    grid = prepared.pixel_to_world(x * u.pix, y * u.pix)
    return (
        (grid.Tx[0, -1] - grid.Tx[0, 0]).to_value(u.arcsec)
        / prepared.rsun_obs.to_value(u.arcsec)
    )


def test_solar_radii_field_of_view_is_identical_for_every_observer():
    geometry = GeometryConfig(shape=(32, 32), max_radius_rsun=1.5, north_up=False)

    near, _ = apply_common_geometry(_observer_map(0.5), geometry=geometry)
    far, _ = apply_common_geometry(_observer_map(1.0), geometry=geometry)

    # The angular scale differs with distance; the solar-radii extent does not.
    # The residual is the gnomonic projection's nonlinearity over the field.
    assert near.rsun_obs > 1.9 * far.rsun_obs
    assert _field_of_view_rsun(near) == pytest.approx(3.0, rel=1e-3)
    assert _field_of_view_rsun(far) == pytest.approx(3.0, rel=1e-3)
    assert _field_of_view_rsun(near) == pytest.approx(_field_of_view_rsun(far), rel=1e-3)
    assert near.data.shape == far.data.shape == (32, 32)


def test_pixels_beyond_the_detector_and_the_radius_are_nan():
    # The detector reaches 1.28 R_sun, the requested view 1.5 R_sun.
    prepared, valid_mask = apply_common_geometry(
        _observer_map(1.0), geometry=GeometryConfig(shape=(64, 64), max_radius_rsun=1.5,
                                                    north_up=False)
    )
    coordinates = all_coordinates_from_map(prepared)
    radius = np.hypot(coordinates.Tx, coordinates.Ty) / prepared.rsun_obs

    outside_view = np.asarray(radius.to_value(u.one)) > 1.5
    assert outside_view.any()
    assert np.all(np.isnan(prepared.data[outside_view]))
    assert not valid_mask[outside_view].any()
    # The detector edge is padded with NaN instead of being silently clipped.
    assert np.isnan(prepared.data).any()
    assert np.all(np.isnan(prepared.data) == ~valid_mask)
    assert valid_mask[prepared.data.shape[0] // 2, prepared.data.shape[1] // 2]


def _write_eui_source(path, *, level='L2'):
    _euv_map(np.ones((4, 4), dtype=np.float32), level=level).save(path, overwrite=True)
    return path


def test_unusable_inputs_are_skipped_and_the_rest_are_prepared(tmp_path, capsys):
    good = _write_eui_source(tmp_path / 'good.fits')
    # An EUI level-1 product is rejected by the adapter: a data defect.
    bad = _write_eui_source(tmp_path / 'bad.fits', level='L1')

    results = prepare_euv_files(
        [good, bad],
        tmp_path / 'prepared',
        adapter='eui',
        calibration_id='cal-v1',
        sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False),
        workers=1,
    )

    assert [result.output_path.name for result in results] == ['good.prepared.fits']
    # Progress and diagnostics share stderr so a redirected log stays ordered.
    report = capsys.readouterr().err
    assert 'Skipped bad.fits' in report
    assert 'Skipped 1 of 2 unusable input(s).' in report


def test_preparation_fails_when_no_input_can_be_prepared(tmp_path):
    sources = [
        _write_eui_source(tmp_path / 'first.fits', level='L1'),
        _write_eui_source(tmp_path / 'second.fits', level='L1'),
    ]

    with pytest.raises(ValueError, match='None of the 2 EUV inputs'):
        prepare_euv_files(
            sources,
            tmp_path / 'prepared',
            adapter='eui',
            calibration_id='cal-v1',
            sensitivity_convention='reference_epoch',
            geometry=GeometryConfig(north_up=False),
            workers=1,
        )


def test_existing_products_are_skipped_without_overwrite(tmp_path, capsys):
    source = _write_eui_source(tmp_path / 'good.fits')
    arguments = dict(
        adapter='eui',
        calibration_id='cal-v1',
        sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False),
        workers=1,
    )
    prepare_euv_files([source], tmp_path / 'prepared', **arguments)

    results = prepare_euv_files([source], tmp_path / 'prepared', **arguments)

    assert results == []
    assert 'already prepared' in capsys.readouterr().err
