import pytest

from sunerf.data.euv.clear_sources import clear_sources, prepared_name


def test_prepared_names_of_aia_and_secchi_sources():
    assert prepared_name('raw/aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.fits') == (
        'aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.prepared.fits')
    assert prepared_name('calibrated/20120801_041630_14euB.fts') == '20120801_041630_14euB.prepared.fits'
    assert prepared_name('raw/20120801_041630_n4euB.fts', secchi_level_0=True) == (
        '20120801_041630_14euB.prepared.fits')
    assert prepared_name('raw/20110208_095430_s4euB.fts', secchi_level_0=True) == (
        '20110208_095430_14euB.prepared.fits')
    with pytest.raises(ValueError, match='level-0.5'):
        prepared_name('raw/20120801_041630_14euB.fts', secchi_level_0=True)


def test_unrecognized_names_are_kept_not_fatal(tmp_path, capsys):
    raw, prepared = tmp_path / 'raw', tmp_path / 'prepared'
    raw.mkdir()
    prepared.mkdir()
    (raw / '20110208_095430_s4euB.fts').write_bytes(b'x')
    (raw / 'odd.fts').write_bytes(b'x')
    (prepared / '20110208_095430_14euB.prepared.fits').write_bytes(b'')

    assert clear_sources(str(raw / '*.fts'), prepared, secchi_level_0=True) == (1, 1)
    assert [p.name for p in raw.iterdir()] == ['odd.fts']
    assert 'no recognized SECCHI level-0.5 name' in capsys.readouterr().out


def test_only_sources_with_a_prepared_product_are_removed(tmp_path):
    raw, prepared = tmp_path / 'raw', tmp_path / 'prepared'
    (prepared / 'rejected').mkdir(parents=True)
    raw.mkdir()
    for name in ('20120801_000000_n4euA.fts', '20120801_020000_n4euA.fts', '20120801_040000_n4euA.fts'):
        (raw / name).write_bytes(b'x' * 10)
    (prepared / '20120801_000000_14euA.prepared.fits').write_bytes(b'')
    (prepared / 'rejected' / '20120801_020000_14euA.prepared.fits').write_bytes(b'')

    assert clear_sources(str(raw / '*.fts'), prepared, secchi_level_0=True, dry_run=True) == (2, 1)
    assert len(list(raw.iterdir())) == 3
    assert clear_sources(str(raw / '*.fts'), prepared, secchi_level_0=True) == (2, 1)
    assert [p.name for p in raw.iterdir()] == ['20120801_040000_n4euA.fts']
