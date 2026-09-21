from datetime import datetime, timedelta, timezone

import pytest
from astropy.io import fits

from sunerf.data.loader.thomson_instrument import GenericThomsonDataset


def _utc_naive(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc).replace(tzinfo=None)


def test_observational_timestamp_uses_date_obs(tmp_path):
    path = tmp_path / "observation.fits"
    hdu = fits.PrimaryHDU()
    hdu.header["DATE-OBS"] = "2025-09-01T00:00:00"
    hdu.header["DATE-AVG"] = "2025-09-01T00:01:00"
    hdu.writeto(path)

    assert GenericThomsonDataset._read_obs_date(path) == _utc_naive(2025, 9, 1)


def test_date_obs_in_extension_precedes_primary_date_avg(tmp_path):
    path = tmp_path / "observation-extension.fits"
    primary = fits.PrimaryHDU()
    primary.header["DATE-AVG"] = "2025-09-01T00:01:00"
    image = fits.ImageHDU()
    image.header["DATE-OBS"] = "2025-09-01T00:00:00"
    fits.HDUList([primary, image]).writeto(path)

    assert GenericThomsonDataset._read_obs_date(path) == _utc_naive(2025, 9, 1)


def test_tb_pb_pairing_uses_header_times_not_filename_order(monkeypatch):
    base = _utc_naive(2025, 9, 1)
    times = {
        'tb_z': base,
        'tb_a': base + timedelta(minutes=1),
        'pb_a': base,
        'pb_z': base + timedelta(minutes=1),
    }
    monkeypatch.setattr(
        GenericThomsonDataset,
        '_read_obs_date',
        classmethod(lambda cls, path: times[path]),
    )

    tb, pb = GenericThomsonDataset._pair_files_by_observation_time(
        ['tb_a', 'tb_z'], ['pb_a', 'pb_z']
    )

    assert tb == ['tb_z', 'tb_a']
    assert pb == ['pb_a', 'pb_z']


def test_tb_pb_pairing_rejects_missing_or_time_shifted_partner(monkeypatch):
    with pytest.raises(ValueError, match='one polarization partner'):
        GenericThomsonDataset._pair_files_by_observation_time(['tb'], [])

    base = _utc_naive(2025, 9, 1)
    times = {'tb': base, 'pb': base + timedelta(seconds=2)}
    monkeypatch.setattr(
        GenericThomsonDataset,
        '_read_obs_date',
        classmethod(lambda cls, path: times[path]),
    )
    with pytest.raises(ValueError, match='Could not pair'):
        GenericThomsonDataset._pair_files_by_observation_time(
            ['tb'], ['pb'], tolerance_seconds=1.0
        )


def test_tb_pb_pairing_rejects_duplicate_timestamps(monkeypatch):
    base = _utc_naive(2025, 9, 1)
    times = {'tb_1': base, 'tb_2': base, 'pb_1': base, 'pb_2': base + timedelta(seconds=1)}
    monkeypatch.setattr(
        GenericThomsonDataset,
        '_read_obs_date',
        classmethod(lambda cls, path: times[path]),
    )

    with pytest.raises(ValueError, match='Duplicate tB'):
        GenericThomsonDataset._pair_files_by_observation_time(
            ['tb_1', 'tb_2'], ['pb_1', 'pb_2']
        )


def test_tb_only_sequence_is_ordered_by_header_time(monkeypatch):
    base = _utc_naive(2025, 9, 1)
    times = {
        'tb_filename_first': base + timedelta(minutes=1),
        'tb_filename_last': base,
    }
    monkeypatch.setattr(
        GenericThomsonDataset,
        '_read_obs_date',
        classmethod(lambda cls, path: times[path]),
    )

    tb, pb = GenericThomsonDataset._pair_files_by_observation_time(
        ['tb_filename_first', 'tb_filename_last'], None
    )

    assert tb == ['tb_filename_last', 'tb_filename_first']
    assert pb is None
