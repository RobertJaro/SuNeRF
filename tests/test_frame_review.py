import csv
import glob
import json
import re
import shutil
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta

import numpy as np
import pytest
from astropy.io import fits

from sunerf.data import frame_review


START = datetime(2022, 3, 28, 10, 0, 0)


def _image(index, channel, rng, shape=(128, 128)):
    y, x = np.mgrid[:shape[0], :shape[1]]
    radius = np.hypot(x - shape[1] / 2, y - shape[0] / 2)
    disk = np.where(radius < 50, 100.0, 100.0 * np.exp(-(radius - 50) / 8))
    # A bright region that drifts slowly across the disk.
    cx, cy = 40 + index, 64
    disk += 400.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30.0)
    scale = 1.0 if channel == 171 else 1.6
    return (scale * disk * rng.normal(1.0, 0.03, shape)).astype(np.float32)


def _write(path, data, *, channel, index, bunit='DN/s', exposure=2.0, valid_mask=None, **header):
    hdu = fits.PrimaryHDU(data)
    hdu.header['DATE-OBS'] = (START + timedelta(minutes=12 * index)).isoformat(timespec='seconds')
    hdu.header['WAVELNTH'] = channel
    hdu.header['EXPTIME'] = exposure
    hdu.header['BUNIT'] = bunit
    hdu.header['INSTRUME'] = 'AIA_3'
    hdu.header['TELESCOP'] = 'SDO/AIA'
    hdu.header['OBSRVTRY'] = 'SDO'
    for key, value in header.items():
        hdu.header[key] = value
    hdus = [hdu]
    if valid_mask is not None:
        hdus.append(fits.ImageHDU(valid_mask.astype(np.uint8), name='VALID_MASK'))
    fits.HDUList(hdus).writeto(path)


def _sequence(tmp_path):
    rng = np.random.default_rng(0)
    data_dir = tmp_path / 'prepared'
    source_dir = tmp_path / 'raw'
    data_dir.mkdir()
    source_dir.mkdir()
    defects = {}
    for channel in (171, 193):
        for index in range(20):
            name = f'aia_{channel}_{index:02d}.prepared.fits'
            data = _image(index, channel, rng)
            mask = np.ones(data.shape, dtype=bool)
            mask[:, :4] = False
            header = {}
            exposure = 2.0
            if channel == 193 and index == 5:
                data = data * 5.0
                defects[name] = 'level jump'
            if channel == 193 and index == 10:
                data[:, :64] = np.nan
                defects[name] = 'valid fraction'
            if channel == 171 and index == 7:
                header['QUALITY'] = 1024
                defects[name] = 'QUALITY'
            if channel == 171 and index == 12:
                header['NSATPIX'] = 3000
                defects[name] = 'saturated'
            if channel == 193 and index == 14:
                header['SRCFILE'] = 'raw_193_14.fits'
                fits.PrimaryHDU(np.zeros((64, 64), dtype=np.int16), header=fits.Header(
                    [('NSATPIX', 50), ('QUALITY', 0)])).writeto(source_dir / 'raw_193_14.fits')
                defects[name] = 'saturated'
            if channel == 171 and index == 15:
                exposure = 0.2
            _write(data_dir / name, data, channel=channel, index=index, exposure=exposure,
                   valid_mask=mask, **header)
    (data_dir / 'truncated.prepared.fits').write_bytes(b'SIMPLE  =                    T' + b' ' * 20)
    defects['truncated.prepared.fits'] = 'unreadable'
    return data_dir, source_dir, defects


def test_scan_flags_each_defect_and_keeps_clean_frames(tmp_path):
    data_dir, source_dir, defects = _sequence(tmp_path)
    output = tmp_path / 'review'
    records = frame_review.scan(
        [str(data_dir / '*.prepared.fits')], output, source_dir=str(source_dir), workers=1
    )
    status = {r['name']: r for r in records}
    for name, reason in defects.items():
        # Saturated pixels are masked at load time; the frame is only flagged.
        expected = 'warn' if reason == 'saturated' else 'reject'
        assert status[name]['status'] == expected, (name, status[name]['reasons'])
        assert any(reason in text for text in status[name]['reasons']), status[name]['reasons']
    clean = [r for r in records if r['name'] not in defects]
    assert all(r['status'] != 'reject' for r in clean), [
        (r['name'], r['reasons']) for r in clean if r['status'] == 'reject'
    ]
    short = status['aia_171_15.prepared.fits']
    assert short['status'] == 'warn'
    assert any('short exposure' in text for text in short['reasons'])

    listed = frame_review.read_decisions(output / 'frames.csv')
    assert sorted(p.split('/')[-1] for p in listed) == sorted(
        name for name, reason in defects.items() if reason != 'saturated'
    )
    meta = json.loads((output / 'scan.json').read_text())
    assert meta['frames'] == 41
    # Instrument colormaps and one display range per channel over the sequence.
    assert meta['channels']['SDO AIA 171']['cmap'] == 'sdoaia171'
    assert meta['channels']['SDO AIA 193']['cmap'] == 'sdoaia193'
    for group in ('SDO AIA 171', 'SDO AIA 193'):
        accepted = [r for r in records if r['group_name'] == group and r['status'] != 'reject']
        bounds = np.array([np.nanpercentile(r['thumb'].astype(np.float32), [1.0, 99.9])
                           for r in accepted])
        expected_min = max(bounds[:, 0].min(), bounds[:, 1].max() - frame_review.MAX_DISPLAY_DECADES)
        assert meta['channels'][group]['log10_min'] == pytest.approx(expected_min, abs=1e-3)
        assert meta['channels'][group]['log10_max'] == pytest.approx(bounds[:, 1].max(), abs=1e-3)
    # The rejected frame five times too bright does not stretch the range.
    jump = status['aia_193_05.prepared.fits']['thumb'].astype(np.float32)
    assert np.nanpercentile(jump, 99.9) > meta['channels']['SDO AIA 193']['log10_max'] + 0.5
    from PIL import Image
    rgb = np.asarray(Image.open(output / status['aia_171_03.prepared.fits']['thumb_url']))
    assert np.abs(rgb[..., 0].astype(int) - rgb[..., 2].astype(int)).max() > 50
    assert not list(output.rglob('*.fits'))
    page = (output / 'index.html').read_text()
    payload = _page_data(page)
    assert len(payload['frames']) == 41
    assert payload['groups'] == ['SDO AIA 171', 'SDO AIA 193', 'unreadable']
    thumbs = [f['thumb'] for f in payload['frames'] if f['thumb']]
    assert len(thumbs) == 40 and all((output / t).is_file() for t in thumbs)
    # Frames of both channels at one time share a review row.
    rows = {}
    for frame in payload['frames']:
        rows.setdefault(frame['row'], set()).add(frame['g'])
    assert sum(groups == {0, 1} for groups in rows.values()) == 20


def test_detector_count_saturation_is_computed_from_the_data(tmp_path):
    data = np.full((64, 64), 1000.0, dtype=np.float32)
    data[10:20, 10:20] = 16380.0
    path = tmp_path / 'raw.fits'
    _write(path, data, channel=193, index=0, bunit='DN')
    record = frame_review.scan_file(str(path))
    assert record['nsatpix'] == 100
    assert record['thumb_saturated'].any()
    # Exposure-normalized values are not compared with the detector limit.
    path_rate = tmp_path / 'rate.fits'
    _write(path_rate, data, channel=193, index=0, bunit='DN/s')
    assert frame_review.scan_file(str(path_rate))['nsatpix'] is None


def _page_data(page):
    return json.loads(re.search(r'const DATA = (.*?);\nconst F', page, re.S).group(1))


def _decisions_csv(frames, rejected_names):
    lines = [','.join(frame_review.DECISION_COLUMNS)]
    for frame in frames:
        decision = 'reject' if frame['name'] in rejected_names else 'keep'
        lines.append(','.join([frame['path'], decision, frame['status'], frame['group_name'],
                               frame['time'], '"' + '; '.join(frame['reasons']) + '"']))
    return '\n'.join(lines) + '\n'


def test_review_directory_works_on_a_copy_and_saves_decisions(tmp_path):
    data_dir, _, _ = _sequence(tmp_path)
    output = tmp_path / 'server' / 'review'
    frame_review.scan([str(data_dir / '*.prepared.fits')], output, workers=1)

    # The review directory is copied to another machine without the data.
    local = tmp_path / 'laptop' / 'review'
    shutil.copytree(output, local)
    frames = frame_review.load_frames(local)
    assert len(frames) == 41 and frames[0]['path'].startswith(str(data_dir))
    assert isinstance(frames[0]['level'], float) and isinstance(frames[0]['row'], int)

    server, url = frame_review.make_review_server(local, port=18765)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page = urllib.request.urlopen(url).read().decode()
        assert _page_data(page)['decisions'] == {}
        text = _decisions_csv(frames, {'aia_171_03.prepared.fits'})
        request = urllib.request.Request(url.replace('index.html', 'decisions.csv'),
                                         data=text.encode(), method='POST')
        assert json.loads(urllib.request.urlopen(request).read()) == {'frames': 41, 'rejected': 1}
        bad = urllib.request.Request(url.replace('index.html', 'decisions.csv'),
                                     data=b'path,decision\nx,maybe\n', method='POST')
        try:
            urllib.request.urlopen(bad)
            raise AssertionError('invalid decisions were accepted')
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()

    assert (local / 'decisions.csv').read_text() == text
    rejected = frame_review.read_decisions(local / 'decisions.csv')
    assert [p.split('/')[-1] for p in rejected] == ['aia_171_03.prepared.fits']
    # Rebuilding the page starts from the saved decisions.
    frame_review.build_page(local)
    decisions = _page_data((local / 'index.html').read_text())['decisions']
    assert sum(value == 'reject' for value in decisions.values()) == 1


def test_apply_moves_rejected_files_out_of_the_glob_and_restore_returns_them(tmp_path):
    data_dir, source_dir, defects = _sequence(tmp_path)
    pattern = str(data_dir / '*.prepared.fits')
    output = tmp_path / 'review'
    frame_review.scan([pattern], output, workers=1)
    before = set(glob.glob(pattern))

    # frames.csv applies the automatic flags.
    assert frame_review.apply_rejections(output / 'frames.csv', dry_run=True) > 0
    assert set(glob.glob(pattern)) == before

    # A reviewed decisions.csv keeps one automatic rejection and adds a frame.
    frames = frame_review.load_frames(output)
    rejected = (set(defects) - {'aia_171_07.prepared.fits'}) | {'aia_193_02.prepared.fits'}
    (output / 'decisions.csv').write_text(_decisions_csv(frames, rejected))
    moved = frame_review.apply_rejections(output / 'decisions.csv')
    after = set(glob.glob(pattern, recursive=True))
    assert len(before) - len(after) == moved
    names_after = {p.split('/')[-1] for p in after}
    assert 'aia_171_07.prepared.fits' in names_after
    assert 'aia_193_02.prepared.fits' not in names_after
    assert (data_dir / 'rejected' / 'rejected.log').is_file()

    assert frame_review.restore(data_dir) == moved
    assert set(glob.glob(pattern)) == before


def test_decisions_from_an_edited_csv_or_a_path_list(tmp_path):
    edited = tmp_path / 'edited.csv'
    with open(edited, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'decision', 'note'])
        writer.writerow(['/data/a.fits', 'reject', 'bloom'])
        writer.writerow(['/data/b.fits', 'keep', ''])
        writer.writerow(['/data/c.fits', ' Reject ', ''])
    assert frame_review.read_decisions(edited) == ['/data/a.fits', '/data/c.fits']
    listing = tmp_path / 'rejected.txt'
    listing.write_text('# comment\n/data/a.fits\n\n')
    assert frame_review.read_decisions(listing) == ['/data/a.fits']


def test_rows_anchor_on_the_densest_channel_and_keep_gaps():
    def frame(group, minute, second):
        return {'group': group, 'unix': 60.0 * minute + second}

    # 171 every minute at :03, 211 at :01 of the next minute with two gaps.
    reference = [frame(0, m, 3) for m in range(10)]
    other = [frame(1, m, 1) for m in range(10) if m not in (4, 6)]
    records = reference + other
    assert frame_review.assign_rows(records) == 10
    for record in other:
        assert record['row'] == int(record['unix'] // 60)
    rows = {}
    for record in records:
        rows.setdefault(record['row'], set()).add(record['group'])
    assert [row for row, groups in sorted(rows.items()) if groups == {0}] == [4, 6]


def test_apply_walks_a_folder_and_prefers_decisions_over_automatic_flags(tmp_path, capsys):
    root = tmp_path / 'review'
    datasets = {}
    for name in ('reviewed', 'automatic'):
        (tmp_path / name).mkdir()
        data_dir, _, defects = _sequence(tmp_path / name)
        frame_review.scan([str(data_dir / '*.prepared.fits')], root / name, workers=1)
        datasets[name] = (data_dir, defects)

    # Only one dataset was reviewed: it keeps the QUALITY frame.
    reviewed = root / 'reviewed'
    frames = frame_review.load_frames(reviewed)
    kept = 'aia_171_07.prepared.fits'
    (reviewed / 'decisions.csv').write_text(
        _decisions_csv(frames, set(datasets['reviewed'][1]) - {kept}))

    assert frame_review.find_review_dirs(root) == [root / 'automatic', root / 'reviewed']
    moved = frame_review.apply_review(root)
    report = capsys.readouterr().out
    assert 'automatic: frames.csv (automatic flags, not reviewed)' in report
    assert 'reviewed: decisions.csv (reviewed decisions)' in report

    total = 0
    for name, (data_dir, _) in datasets.items():
        rejected = {p.name for p in (data_dir / 'rejected').glob('*.fits')}
        if name == 'reviewed':  # the decisions, including a manual rejection
            expected = set(datasets[name][1]) - {kept}
        else:  # the automatic flags
            expected = {f['name'] for f in frame_review.load_frames(root / name)
                        if f['status'] == 'reject'}
        assert rejected == expected
        total += len(expected)
    assert moved == total

    with pytest.raises(SystemExit, match='No review directory'):
        frame_review.apply_review(tmp_path / 'automatic')


def _peak_file(path, data, **header):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype=np.float32))
    hdu.header['DATE-OBS'] = '2022-03-28T11:00:00'
    hdu.header['WAVELNTH'] = 193
    for key, value in header.items():
        hdu.header[key] = value
    hdu.writeto(path)
    return frame_review.scan_file(str(path))


def test_peak_detector_dn_follows_the_prepare_saturation_convention(tmp_path):
    data = np.full((32, 32), 100.0)
    data[4:8, 4] = 16375.0  # a clipped plateau of four pixels
    aia = _peak_file(tmp_path / 'aia.fits', data, INSTRUME='AIA_3', BUNIT='DN', EXPTIME=2.0)
    assert (aia['peak_dn'], aia['plateau_pixels']) == (16375.0, 4)
    # EUI level 2: rate times exposure.
    eui = _peak_file(tmp_path / 'eui.fits', data / 4.0, INSTRUME='EUI', BUNIT='DN/s', XPOSURE=4.0)
    assert eui['peak_dn'] == pytest.approx(16375.0)
    # SECCHI_PREP: on the 14-bit scale of DSATVAL.
    euvi = _peak_file(tmp_path / 'euvi.fits', data / 16383.0 * 1864.0, INSTRUME='SECCHI',
                      DETECTOR='EUVI', BUNIT='DN/s', EXPTIME=8.0, DSATVAL=1864.0)
    assert euvi['peak_dn'] == pytest.approx(16375.0, rel=1e-5)
    # Prepared products are degradation corrected and already masked.
    prepared = _peak_file(tmp_path / 'prep.fits', data, INSTRUME='AIA_3', BUNIT='DN/s',
                          EXPTIME=2.0, PREPSCHM='sunerf-prepared-euv-v2')
    assert np.isnan(prepared['peak_dn'])


def test_saturation_is_found_at_a_shared_plateau_and_recommended_below_it():
    unsaturated = [{'peak_dn': value, 'plateau_pixels': 1} for value in (9000., 11000., 12500.)]
    saturated = [{'peak_dn': value, 'plateau_pixels': n}
                 for value, n in ((16375., 40), (16370., 12), (16375., 300))]
    estimate = frame_review.saturation_estimate(unsaturated + saturated)
    assert estimate['saturation_dn'] == 16375.0
    assert estimate['saturated_frames'] == 3
    assert estimate['recommended_dn'] == 13100

    # A bright frame without a plateau, or a single clipped frame, is no ceiling.
    assert frame_review.saturation_estimate(unsaturated)['recommended_dn'] is None
    one = frame_review.saturation_estimate(unsaturated + saturated[:1])
    assert (one['saturation_dn'], one['ceiling_dn']) == (None, 16375.0)
    assert frame_review.saturation_estimate([{'peak_dn': None}])['ceiling_dn'] is None


def test_review_saves_the_selected_saturation_level(tmp_path):
    data_dir, _, _ = _sequence(tmp_path)
    review = tmp_path / 'review'
    frame_review.scan([str(data_dir / '*.prepared.fits')], review, workers=1)
    meta = json.loads((review / 'scan.json').read_text())
    assert set(meta['saturation']['channels']) == {'SDO AIA 171', 'SDO AIA 193', 'unreadable'}

    server, url = frame_review.make_review_server(review, port=18775)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = url.replace('index.html', 'saturation.csv')
    header = ','.join(frame_review.SATURATION_COLUMNS)
    try:
        good = f'{header}\nSDO AIA 171,,,0,,13100\nSDO AIA 193,,,0,,13100\n'
        request = urllib.request.Request(endpoint, data=good.encode(), method='POST')
        assert json.loads(urllib.request.urlopen(request).read())['selected_dn'] == '13100'
        for bad in (f'{header}\nSDO AIA 171,,,0,,13100\nSDO AIA 193,,,0,,12000\n',
                    f'{header}\nSDO AIA 171,,,0,,-5\n'):
            try:
                urllib.request.urlopen(urllib.request.Request(endpoint, data=bad.encode(), method='POST'))
                raise AssertionError('invalid saturation level was accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 400
    finally:
        server.shutdown()
        server.server_close()
    assert (review / 'saturation.csv').read_text() == good
    frame_review.build_page(review)
    assert _page_data((review / 'index.html').read_text())['saturationSelected'] == '13100'



def test_cme_review_masks_round_trip_on_native_grid(tmp_path):
    import io
    from PIL import Image
    from sunerf.data.review_masks import mask_path
    from sunerf.data.loader.base_loader import MapDataLoader
    from sunerf.data.loader.thomson_instrument import _load_fits_stack

    server = tmp_path / 'server'
    channel = server / 'cor2' / 'tB'
    channel.mkdir(parents=True)
    data = np.linspace(-1e-11, 1e-9, 35 * 79, dtype=np.float32).reshape(35, 79)
    data[2, 7] = np.nan
    paths = [channel / name for name in ('0.fts', '1.fits.gz', '2.fits')]
    for i, path in enumerate(paths):
        _write(path, data, channel=1, index=i, QUALITY=int(i == 2), INSTRUME='COR2', bunit='MSB',
               CTYPE1='HPLN-TAN', CTYPE2='HPLT-TAN', CUNIT1='arcsec', CUNIT2='arcsec',
               CDELT1=30., CDELT2=30., CRPIX1=40., CRPIX2=18., HGLN_OBS=0., HGLT_OBS=0.,
               DSUN_OBS=1.496e11, RSUN_REF=6.957e8)
    review = server / 'review'
    (channel / 'notes.txt').touch()
    frame_review.scan([str(channel)], review, cme=True, workers=1)
    local = tmp_path / 'local'
    shutil.copytree(review, local)
    frames = frame_review.load_frames(local)
    assert [f['path'] for f in frames] == [str(path) for path in paths]
    assert {f['group_name'] for f in frames} == {'cor2 tB'}
    assert frames[0]['valid_fraction'] == pytest.approx(1 - 1 / data.size, abs=1e-6)
    assert frames[2]['status'] == 'reject'
    with Image.open(local / frames[0]['preview']) as preview:
        assert preview.size == data.shape[::-1]

    # Paint near the top left in display coordinates, away from the native NaN.
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[3:6, 9:11] = 255
    image_bytes = io.BytesIO()
    Image.fromarray(rgba).save(image_bytes, format='PNG')
    # The local server only has scan artifacts. Brush edits must not change decisions.
    server_http, url = frame_review.make_review_server(local, port=0)
    thread = threading.Thread(target=server_http.serve_forever, daemon=True)
    thread.start()
    mask_file = local / 'masks' / frames[0]['mask_file']
    assert mask_file.name == paths[0].name + '.npy'
    endpoint = url.replace('index.html', 'pixel-mask/tB/0.fts.npy')
    try:
        urllib.request.urlopen(urllib.request.Request(endpoint, data=image_bytes.getvalue(), method='POST')).close()
        assert not (local / 'decisions.csv').exists()
        np.testing.assert_array_equal(np.load(mask_file), rgba[::-1, :, 3] != 0)
        assert np.load(mask_file).dtype == np.bool_
        page = _page_data(urllib.request.urlopen(url).read().decode())
        assert page['masks'] == {str(paths[0]): 'mask-preview/tB/0.fts.npy'}
        with Image.open(urllib.request.urlopen(url.replace('index.html', 'mask-preview/tB/0.fts.npy'))) as overlay:
            np.testing.assert_array_equal(np.asarray(overlay)[..., 3], rgba[..., 3])
        shutil.copytree(local / 'masks', review / 'masks', dirs_exist_ok=True)
        urllib.request.urlopen(urllib.request.Request(endpoint, data=b'', method='POST')).close()
        assert not mask_file.exists()
    finally:
        server_http.shutdown()
        server_http.server_close()

    original_fits = paths[0].read_bytes()
    assert frame_review.apply_review(review, dry_run=True) == 1
    assert not mask_path(paths[0]).exists() and paths[2].exists()
    assert frame_review.apply_review(review) == 1
    assert frame_review.apply_review(review) == 0
    assert (channel / 'rejected' / paths[2].name).exists()
    assert mask_path(paths[0]).is_file() and not mask_path(paths[1]).exists()
    expected = data.copy()
    expected[-6:-3, 9:11] = np.nan
    tb = MapDataLoader(1, reference_frame='inertial').load(str(paths[0]))['image']
    pb = _load_fits_stack([str(paths[0])], workers=0, description='Loading pB')[0]
    np.testing.assert_array_equal(tb, expected)
    np.testing.assert_array_equal(pb, expected)
    assert paths[0].read_bytes() == original_fits
    np.testing.assert_array_equal(fits.getdata(paths[0]), data)
    # A fresh server scan preserves installed masks. Mirroring a local clear removes only the sidecar.
    fresh = server / 'fresh'
    frame_review.scan([str(channel)], fresh, cme=True, workers=1)
    assert len(frame_review.load_frames(fresh)) == 2
    np.testing.assert_array_equal(np.load(fresh / 'masks' / frames[0]['mask_file']), rgba[::-1, :, 3] != 0)
    (review / 'masks' / frames[0]['mask_file']).unlink()
    frame_review.apply_review(review)
    assert not mask_path(paths[0]).exists()
    assert paths[0].read_bytes() == original_fits


def test_cme_preview_subtracts_daily_mean_minimum_only_from_tb(tmp_path):
    from PIL import Image
    from sunerf.data.review_masks import mask_path, write_mask

    # Different daily counts, a signed transient and a bad frame on an otherwise empty day.
    y, x = np.mgrid[:32, :48]
    baseline = (20 + x + y).astype(np.float32) * 1e-12
    transient = np.where(x < 24, 4e-12, -2e-12)
    dates = [1, 1, 1, 2, 3]
    amplitudes = [0, 0, 3, 2, -100]
    images = [(baseline + a * transient).astype(np.float32) for a in amplitudes]
    images[2][0, 0] = np.nan
    for image in images:
        image[-1, 0] = np.nan
    for channel in ('tB', 'pB'):
        folder = tmp_path / 'cor2' / channel
        folder.mkdir(parents=True)
        for i, (day, image) in enumerate(zip(dates, images)):
            _write(folder / f'{i}.fts', image, channel=1, index=i, exposure=-1, QUALITY=int(i == 4),
                   **{'DATE-OBS': f'2025-09-{day:02d}T00:0{i}:00'})

    excluded = np.zeros(baseline.shape, dtype=bool)
    excluded[4, 4] = True
    write_mask(mask_path(tmp_path / 'cor2' / 'tB' / '2.fts'), excluded)
    review = tmp_path / 'review'
    records = frame_review.scan([str(tmp_path / 'cor2' / c) for c in ('tB', 'pB')], review, workers=1, cme=True)
    first_day = np.stack(images[:3])
    first_day[2, excluded] = np.nan
    with pytest.warns(RuntimeWarning, match='Mean of empty slice'):
        background = np.fmin(np.nanmean(first_day, axis=0, dtype=np.float64), images[3])
    for record in records:
        i = int(record['name'][0])
        assert record['status'] == ('reject' if i == 4 else 'ok')
        assert np.isnan(record['exposure'])
        corrected = record['channel'] == 'tB'
        data = images[i] - background if corrected else images[i]
        expected = np.arcsinh(data.astype(np.float64) / 1e-12)
        np.testing.assert_allclose(record['thumb'], expected, rtol=1e-3, atol=1e-6)
        assert bool(record.get('tb_background_subtracted')) == corrected
        # These frames fit inside a thumbnail. Both JPEGs must use the same
        # colors across the series, including for the rejected outlier.
        with Image.open(review / record['preview']) as preview, Image.open(review / record['thumb_url']) as thumb:
            for col in (5, 40):
                np.testing.assert_allclose(preview.getpixel((col, 10)), thumb.getpixel((col, 10)), atol=6)
            if i == 4 and corrected:
                assert preview.getpixel((5, 10))[0] < 5
                assert preview.getpixel((40, 10))[0] > 250


def test_cme_without_background_frames_keeps_raw_preview(tmp_path):
    path = tmp_path / 'bad.fts'
    data = np.linspace(1e-12, 1e-10, 64, dtype=np.float32).reshape(8, 8)
    _write(path, data, channel=1, index=0, QUALITY=1)
    review = tmp_path / 'review'
    frame, = frame_review.scan([str(path)], review, workers=1, cme=True)
    assert frame['status'] == 'reject' and not frame['tb_background_subtracted']
    np.testing.assert_allclose(frame['thumb'], np.arcsinh(data / 1e-12), rtol=1e-3)
    assert (review / frame['preview']).is_file()


@pytest.mark.parametrize('instrument, channel, cmap_name', [
    ('cor2', 'pB', 'soholasco2'), ('cor2_clear', 'tB', 'soholasco2'),
    ('punch_pam', 'pB', 'punch'), ('ccor', 'tB', 'Blues_r'),
    ('lasco_c2', 'tB', 'soholasco2'), ('lasco_c3', 'tB', 'soholasco3'),
])
def test_cme_preview_colors(tmp_path, monkeypatch, instrument, channel, cmap_name):
    from PIL import Image

    folder = tmp_path / instrument
    if instrument not in ('cor2_clear', 'ccor'):
        folder /= channel
    folder.mkdir(parents=True)
    monkeypatch.setattr(frame_review, 'THUMBNAIL_SIZE', 32)
    data = np.tile(np.repeat([1., 3., 9., 27.], 16), (64, 1)) * 1e-12
    data[8, 8] = np.nan
    valid = np.ones_like(data, dtype=bool)
    valid[16:24, 8:16] = False
    # One rejected frame gives both products the raw asinh display range.
    _write(folder / 'frame.fts', data, channel=1, index=0, valid_mask=valid, QUALITY=1)
    review = tmp_path / 'review'
    record, = frame_review.scan([str(folder)], review, workers=1, cme=True)
    assert record['cmap'] == cmap_name
    levels = np.arcsinh([1., 3., 27.])
    color = frame_review._colormap(cmap_name)((levels[1] - levels[0]) / (levels[2] - levels[0]), bytes=True)[:3]
    for key, factor in [('preview', 1), ('thumb_url', 2)]:
        with Image.open(review / record[key]) as image:
            pixels = np.asarray(image)[::-1].astype(int)
        np.testing.assert_allclose(pixels[32 // factor, 24 // factor], color, atol=4)
        for row, col in [(8, 8), (20, 12)]:
            # JPEG can shift the RGB values of an isolated pixel.
            red, green, blue = pixels[row // factor, col // factor]
            assert green > 200 and green - max(red, blue) > 120


def test_cme_borderline_frame_is_warned_and_hard_quality_flag_rejects():
    members = [
        dict(error='', unix=i, valid_fraction=1., level_z=0., structure_z=0.)
        for i in range(4)
    ]
    members[1].update(valid_fraction=0.88, structure_z=6.5)
    members[2]['quality'] = 1
    frame_review._classify_cme(members)
    assert [r['status'] for r in members] == ['ok', 'warn', 'reject', 'ok']


def test_mask_partners_require_paired_native_grids_and_observation_times():
    def frame(index, path, row, seconds=30., width=32):
        return dict(index=index, path=path, row=row, unix=seconds, width=width, height=16,
                    preview='preview.jpg', mask_file=path + '.npy')

    frames = [
        frame(0, 'cor2/tB/t.fts', 0), frame(1, 'cor2/pB/p.fts', 0, seconds=30.5),
        frame(2, 'cor2/tB/t2.fts', 1), frame(3, 'cor2/pB/p2.fts', 1, seconds=40.),
        frame(4, 'punch/tB/image.fits', 2), frame(5, 'punch/pB/image.fits', 2, width=64),
        frame(6, 'lasco_c2/tB/image.fits', 3), frame(7, 'punch/pB/image.fits', 3),
    ]
    assert frame_review._mask_partners(frames) == {0: 1, 1: 0}


def test_bulk_mask_keeps_existing_pixels_and_can_retry_partial_write(tmp_path):
    shape = (2, 3)
    editable = {name: (tmp_path / (name + '.npy'), shape) for name in ('source', 'first', 'second')}
    painted = np.array([[True, False, False], [False, False, False]])
    existing = np.array([[False, False, False], [False, True, False]])
    np.save(editable['source'][0], painted)
    np.save(editable['first'][0], existing)
    editable['second'][0].write_bytes(b'invalid npy')
    result = frame_review._merge_review_masks(editable, ['source'], ['first', 'second'])
    assert result['applied'] == ['first'] and 'second' in result['error']
    np.testing.assert_array_equal(np.load(editable['first'][0]), existing | painted)
    editable['second'][0].unlink()
    result = frame_review._merge_review_masks(editable, ['source'], ['first', 'second'])
    assert result == {'applied': ['first', 'second']}
    np.testing.assert_array_equal(np.load(editable['first'][0]), existing | painted)
    np.testing.assert_array_equal(np.load(editable['second'][0]), painted)
