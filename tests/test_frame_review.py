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

