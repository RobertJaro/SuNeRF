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
    peak_dn_layer,
    BLEED_PIXELS,
    INTERPOLATION_PIXELS,
    build_parser,
)
from sunerf.data.euv.observation import load_prepared_observation


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
        # SECCHI_PREP saturation value, in the units of the data.
        header['dsatval'] = 1.0e4
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

    # Data, mask and peak-DN layer share the pointing update and registration.
    assert calls == [
        'pointing', 'pointing', 'pointing', 'register-3', 'register-0', 'register-0',
        'degradation'
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


def test_external_calibrations_must_be_exposure_normalized():
    """EUVI and EUI are never divided by exposure in SuNeRF, so their BUNIT must already be a rate."""
    counts = _euv_map(
        [[5.0]], instrument='SECCHI/EUVI', channel=195, bunit='DN',
        spacecraft='B', level='SECCHI_PREP_L1',
    )
    with pytest.raises(ValueError, match='not exposure-normalized'):
        prepare_euvi_map(
            counts,
            spacecraft='B',
            product_level='SECCHI_PREP_L1',
            sensitivity_convention='static_assumed',
            geometry=GeometryConfig(north_up=False),
        )

    eui_counts = _euv_map([[4.0]], level='L2', bunit='DN')
    with pytest.raises(ValueError, match='not exposure-normalized'):
        prepare_eui_map(
            eui_counts,
            calibration_id='cal-v1',
            sensitivity_convention='static_assumed',
            geometry=GeometryConfig(north_up=False),
        )


def _identity_aiapy(monkeypatch):
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._load_aiapy_calibration',
        lambda: (
            lambda value, **kwargs: value,
            lambda value, **kwargs: value,
            lambda value, **kwargs: value,
        ),
    )
    monkeypatch.setattr('sunerf.data.euv.prepare._load_pointing_table', lambda path: 'pointing')
    monkeypatch.setattr(
        'sunerf.data.euv.prepare._cached_correction_table',
        lambda *args: QTable({'DATE': [], 'WAVELNTH': [], 'EFF_AREA': []}),
    )


ROWS = BLEED_PIXELS + INTERPOLATION_PIXELS  # reach along the readout columns
COLS = 1 + INTERPOLATION_PIXELS            # reach across them


def test_peak_dn_layer_covers_the_bleed_and_interpolation_reach():
    dn = np.zeros((40, 40))
    dn[20, 20] = 16000.0
    dn[0, 0] = np.nan
    layer = peak_dn_layer(dn)
    expected = np.zeros(dn.shape, dtype=bool)
    expected[20 - ROWS:21 + ROWS, 20 - COLS:21 + COLS] = True
    assert np.array_equal(layer >= 16000.0, expected)
    assert layer[0, 0] == 0.0 and layer.dtype == np.float32


def test_aia_records_the_peak_detector_dn_and_masks_nothing(tmp_path, monkeypatch):
    _identity_aiapy(monkeypatch)
    correction = tmp_path / 'correction.ecsv'
    pointing = tmp_path / 'pointing.ecsv'
    correction.write_text('fixture')
    pointing.write_text('fixture')
    data = np.full((32, 32), 100.0)
    data[12:16, 10] = 16383.0  # a clipped bleed trail along a readout column
    source = _euv_map(data, instrument='AIA', channel=193, bunit='DN', level='L1')
    source.meta['quality'] = 0

    prepared, mask, provenance = prepare_aia_map(
        source, correction_table_path=correction, pointing_table_path=pointing,
        geometry=GeometryConfig(north_up=False),
    )
    assert mask.all() and np.isfinite(prepared.data).all()
    assert (provenance['frame_peak_dn'], provenance['frame_peak_pixels']) == (16383.0, 4)
    assert provenance['peak_dn_conversion'] == 'AIA level-1 DN'
    expected = np.zeros(data.shape, dtype=bool)
    expected[12 - ROWS:16 + ROWS, 10 - COLS:11 + COLS] = True
    assert np.array_equal(provenance['peak_dn_layer'] >= 13000.0, expected)


def _euvi(data, **header):
    source = _euv_map(
        data, instrument='SECCHI/EUVI', channel=195, spacecraft='A', level='SECCHI_PREP_L1'
    )
    for key, value in header.items():
        source.meta[key] = value
    return source


def _prepare_euvi(source):
    return prepare_euvi_map(
        source, spacecraft='A', product_level='SECCHI_PREP_L1',
        sensitivity_convention='reference_epoch', geometry=GeometryConfig(north_up=False),
    )


def test_euvi_peak_dn_is_on_the_14_bit_scale_of_the_header_saturation_value():
    data = np.full((32, 32), 10.0)
    data[5:8, 20] = 1000.0
    _, mask, provenance = _prepare_euvi(_euvi(data, dsatval=1000.0, datasat=3))
    assert mask.all()
    assert provenance['frame_peak_dn'] == pytest.approx(16383.0)
    assert provenance['frame_peak_pixels'] == 3

    unsaturated = np.full((8, 8), 10.0)
    _, _, provenance = _prepare_euvi(_euvi(unsaturated, dsatval=None))
    assert provenance['peak_dn_conversion'] == 'unavailable'
    assert provenance['peak_dn_layer'] is None
    with pytest.raises(ValueError, match='not in the units'):
        _prepare_euvi(_euvi(unsaturated * 1000, dsatval=1000.0))
    with pytest.raises(ValueError, match='DATASAT=4 but no pixel'):
        _prepare_euvi(_euvi(unsaturated, dsatval=1000.0, datasat=4))


def _write_eui_with_peak(tmp_path, peak_rate):
    data = np.full((16, 16), 10.0, dtype=np.float32)
    data[4, 4] = peak_rate  # in the 2 s exposure
    source = tmp_path / 'eui.fits'
    _euv_map(data).save(source, overwrite=True)
    results = prepare_euv_files(
        [source], tmp_path / 'prepared', adapter='eui', sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False), workers=1,
    )
    return results[0].output_path


def test_eui_products_carry_the_peak_dn_layer_and_header(tmp_path):
    path = _write_eui_with_peak(tmp_path, 500.0)
    with fits.open(path) as hdul:
        header = hdul[0].header
        assert (header['PEAKDN'], header['PEAKNPX']) == (1000.0, 1)
        assert header['PEAKEXT'] == 'PEAK_DN'
        assert header['PEAKCONV'] == 'EUI level-2 rate * XPOSURE'
        layer = hdul['PEAK_DN'].data
        assert layer.shape == hdul[0].data.shape
        assert layer[4, 4] == 1000.0 and layer[15, 15] == 20.0


def test_the_loader_masks_pixels_whose_peak_reaches_the_saturation_level(tmp_path):
    path = _write_eui_with_peak(tmp_path, 500.0)
    _, mask = load_prepared_observation(path, required_mask=True)
    assert mask.all()
    _, mask = load_prepared_observation(path, required_mask=True, saturation_dn=900.0)
    expected = np.zeros(mask.shape, dtype=bool)
    expected[max(4 - ROWS, 0):5 + ROWS, max(4 - COLS, 0):5 + COLS] = True
    assert np.array_equal(~mask, expected)
    _, mask = load_prepared_observation(path, required_mask=True, saturation_dn=1500.0)
    assert mask.all()

    # A product without PEAK_DN has no peak information: it loads unmasked, with a warning.
    with fits.open(path) as hdul:
        stripped = fits.HDUList([hdul[0].copy(), hdul['VALID_MASK'].copy()])
        stripped.writeto(tmp_path / 'old.fits')
    with pytest.warns(UserWarning, match='without a PEAK_DN extension'):
        _, mask = load_prepared_observation(tmp_path / 'old.fits', required_mask=True, saturation_dn=900.0)
    assert mask.all()


def test_preparation_takes_no_saturation_level():
    common = ['--input', 'x.fits', '--output-dir', 'out', '--sensitivity-convention',
              'reference_epoch']
    build_parser().parse_args(['eui', *common])
    with pytest.raises(SystemExit):
        build_parser().parse_args(['eui', *common, '--saturation-dn', '13000'])


def test_the_peak_dn_layer_follows_reprojection_to_the_prepared_grid():
    data = np.full((32, 32), 10.0)
    data[14:18, 15] = 500.0  # 1000 DN in the 2 s exposure
    prepared, mask, provenance = prepare_eui_map(
        _euv_map(data), sensitivity_convention='reference_epoch',
        geometry=GeometryConfig(north_up=False, shape=(16, 16),
                                hpc_bounds_arcsec=(-30.0, -30.0, 30.0, 30.0)),
    )
    layer = provenance['peak_dn_layer']
    assert layer.shape == prepared.data.shape == (16, 16)
    assert np.array_equal(np.isfinite(layer), mask)
    assert np.nanmax(layer) == 1000.0
    # The saturated trail and its reach cover a compact patch of the new grid.
    assert 0 < np.count_nonzero(layer >= 900.0) < 0.25 * layer.size

