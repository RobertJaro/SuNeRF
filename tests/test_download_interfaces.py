import datetime as dt
import importlib
from pathlib import Path

import pytest

from sunerf.data.download import core
from sunerf.data.download.core import (
    DownloadRequest,
    add_common_arguments,
    cadence_string,
    parse_cadence,
    request_from_args,
)
from sunerf.data.download.download_aia import aia_dataset


DOWNLOADER_MODULES = (
    "download_aia",
    "download_eui",
    "download_euvi",
    "download_suvi",
    "download_cor",
    "download_ccor",
    "download_lasco",
    "download_punch",
    "download_psp_insitu",
    "download_solo_insitu",
)


@pytest.mark.parametrize("module_name", DOWNLOADER_MODULES)
def test_every_downloader_exposes_the_common_interface(module_name, tmp_path):
    module = importlib.import_module(f"sunerf.data.download.{module_name}")
    parser = module.build_parser()
    destinations = {action.dest for action in parser._actions}
    assert {"start", "end", "output", "overwrite", "dry_run"} <= destinations
    assert callable(module.download)
    assert callable(module.main)

    arguments = [
        "--start", "2025-01-01T00:00:00Z",
        "--end", "2025-01-02T00:00:00Z",
        "--output", str(tmp_path),
        "--dry-run",
    ]
    if module_name == "download_aia":
        arguments.extend(("--email", "registered@example.org"))
    request = request_from_args(parser.parse_args(arguments))
    assert request.start == dt.datetime(2025, 1, 1)
    assert request.end == dt.datetime(2025, 1, 2)
    assert request.output == tmp_path
    assert request.dry_run


def test_common_request_rejects_reverse_time_range(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    args = parser.parse_args([
        "--start", "2025-01-02", "--end", "2025-01-01",
        "--output", str(tmp_path),
    ])
    with pytest.raises(SystemExit, match="start must be earlier than end"):
        request_from_args(args)


@pytest.mark.parametrize(
    "text, seconds",
    (("30s", 30), ("1.5h", 5400), ("2d", 172800)),
)
def test_common_cadence_parser(text, seconds):
    cadence = parse_cadence(text)
    assert cadence.total_seconds() == seconds
    assert cadence_string(cadence) == f"{seconds}s"


class _Records:
    colnames = ("file",)

    def __init__(self, count):
        self._count = count

    def __len__(self):
        return self._count

    def __str__(self):
        return f"records={self._count}"


def test_fetch_fido_dry_run_uses_common_result(capsys, tmp_path):
    request = DownloadRequest(
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 2),
        output=tmp_path / "unused",
        dry_run=True,
    )

    result = core.fetch_fido(
        _Records(2), _Records(3), request=request, description="test download"
    )

    assert result == core.DownloadResult(selected=5)
    assert "records=2" in capsys.readouterr().out
    assert not request.output.exists()


def test_combine_results_preserves_uniform_counts_and_files():
    result = core.combine_results([
        core.DownloadResult(selected=2, downloaded=1, skipped=1, files=("a.fits",)),
        core.DownloadResult(selected=3, downloaded=3, files=("b.fits", "c.fits")),
    ])

    assert result == core.DownloadResult(
        selected=5,
        downloaded=4,
        skipped=1,
        files=("a.fits", "b.fits", "c.fits"),
    )


def test_aia_query_uses_common_time_and_cadence_contract(tmp_path):
    request = DownloadRequest(
        start=dt.datetime(2012, 8, 1),
        end=dt.datetime(2012, 8, 2),
        output=Path(tmp_path),
        dry_run=True,
    )
    dataset = aia_dataset(request, [171, 193, 211], parse_cadence("6h"))
    assert dataset == (
        "aia.lev1_euv_12s[2012-08-01_00:00:00 / 86400s@21600s]"
        "[171,193,211]{image}"
    )


class _Export:
    """Minimal stand-in for a finished drms export request."""

    def __init__(self, filenames):
        import pandas as pd

        self.urls = pd.DataFrame({"filename": list(filenames)})
        self.downloaded = None

    def wait(self):
        return None

    def download(self, directory, index=None):
        import pandas as pd

        names = self.urls.filename if index is None else self.urls.filename.iloc[index]
        paths = [Path(directory) / str(name) for name in names]
        for path in paths:
            path.write_text("fits")
        self.downloaded = [str(path) for path in paths]
        return pd.DataFrame({"download": self.downloaded})


def test_aia_download_skips_files_that_are_already_present(tmp_path, monkeypatch):
    from sunerf.data.download import download_aia

    export = _Export(["a.fits", "b.fits", "c.fits"])
    (tmp_path / "a.fits").write_text("existing")

    class _Client:
        def __init__(self, email):
            self.email = email

        def export(self, dataset, protocol):
            return export

    monkeypatch.setattr(download_aia.drms, "Client", _Client)
    request = DownloadRequest(
        start=dt.datetime(2012, 8, 1), end=dt.datetime(2012, 8, 2), output=tmp_path
    )

    result = download_aia.download(request, email="registered@example.org")

    assert result == core.DownloadResult(
        selected=3,
        downloaded=2,
        skipped=1,
        files=(str(tmp_path / "b.fits"), str(tmp_path / "c.fits")),
    )
    assert (tmp_path / "a.fits").read_text() == "existing"
    assert sorted(path.name for path in tmp_path.glob("*")) == ["a.fits", "b.fits", "c.fits"]


def test_aia_download_replaces_existing_files_when_overwriting(tmp_path, monkeypatch):
    from sunerf.data.download import download_aia

    export = _Export(["a.fits"])
    (tmp_path / "a.fits").write_text("existing")

    monkeypatch.setattr(
        download_aia.drms, "Client", lambda email: type("C", (), {"export": lambda *a, **k: export})()
    )
    request = DownloadRequest(
        start=dt.datetime(2012, 8, 1),
        end=dt.datetime(2012, 8, 2),
        output=tmp_path,
        overwrite=True,
    )

    result = download_aia.download(request, email="registered@example.org")

    assert result == core.DownloadResult(
        selected=1, downloaded=1, skipped=0, files=(str(tmp_path / "a.fits"),)
    )
    assert (tmp_path / "a.fits").read_text() == "fits"


def test_aia_export_waits_while_another_export_is_pending(tmp_path, monkeypatch, capsys):
    import drms

    from sunerf.data.download import download_aia

    export = _Export(["a.fits"])
    pending = drms.DrmsExportError(
        "User x has 1 pending export requests (JSOC_1); please wait [status=7]"
    )
    refusals = [pending, pending]

    class _Client:
        def __init__(self, email):
            pass

        def export(self, dataset, protocol):
            if refusals:
                raise refusals.pop()
            return export

    sleeps = []
    monkeypatch.setattr(download_aia.drms, "Client", _Client)
    monkeypatch.setattr(download_aia.time, "sleep", sleeps.append)
    request = DownloadRequest(
        start=dt.datetime(2012, 8, 1), end=dt.datetime(2012, 8, 2), output=tmp_path
    )

    result = download_aia.download(request, email="registered@example.org")

    assert result.downloaded == 1
    assert sleeps == [download_aia.PENDING_EXPORT_WAIT_SECONDS] * 2
    assert "submitting again" in capsys.readouterr().out


def test_aia_export_failure_other_than_pending_is_raised(tmp_path, monkeypatch):
    import drms

    from sunerf.data.download import download_aia

    class _Client:
        def __init__(self, email):
            pass

        def export(self, dataset, protocol):
            raise drms.DrmsExportError(" [status=4]")

    monkeypatch.setattr(download_aia.drms, "Client", _Client)
    request = DownloadRequest(
        start=dt.datetime(2012, 8, 1), end=dt.datetime(2012, 8, 2), output=tmp_path
    )
    with pytest.raises(drms.DrmsExportError, match="status=4"):
        download_aia.download(request, email="registered@example.org")


def test_search_with_retry_recovers_from_transient_failures():
    from sunerf.data.download.core import search_with_retry

    calls = []
    delays = []

    def search(*query):
        calls.append(query)
        if len(calls) < 3:
            raise RuntimeError("VSO returned no matching records")
        return "records"

    result = search_with_retry(search, "q", attempts=4, initial_delay=1.0, sleep=delays.append)
    assert result == "records"
    assert len(calls) == 3
    assert delays == [1.0, 2.0]


def test_search_with_retry_raises_after_last_attempt():
    from sunerf.data.download.core import search_with_retry

    def search(*query):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="Search failed after 2 attempts"):
        search_with_retry(search, attempts=2, initial_delay=0.0, sleep=lambda _: None)


def test_split_request_covers_the_range_without_overlap(tmp_path):
    import datetime as dt

    from sunerf.data.download.core import DownloadRequest, split_request

    request = DownloadRequest(
        start=dt.datetime(2025, 9, 27), end=dt.datetime(2025, 10, 13), output=tmp_path
    )
    parts = list(split_request(request, dt.timedelta(days=7)))
    assert [(p.start, p.end) for p in parts] == [
        (dt.datetime(2025, 9, 27), dt.datetime(2025, 10, 4)),
        (dt.datetime(2025, 10, 4), dt.datetime(2025, 10, 11)),
        (dt.datetime(2025, 10, 11), dt.datetime(2025, 10, 13)),
    ]
    assert list(split_request(request, None)) == [request]


def _euvi_records(times):
    from astropy.table import Table
    from astropy.time import Time

    fileids = [
        f"secchi/L0/a/img/euvi/{time:%Y%m%d}/{time:%Y%m%d_%H%M%S}_n4euA.fts" for time in times
    ]
    return Table({"Start Time": Time(times), "fileid": fileids})


def _fake_euvi_archive(records_by_channel):
    import astropy.units as u

    def search(_search, *query):
        from sunpy.net import attrs as a

        channel = next(round(q.min.to_value(u.AA)) for q in query if isinstance(q, a.Wavelength))
        if channel not in records_by_channel:
            raise RuntimeError("VSO returned no matching records")
        return _euvi_records(records_by_channel[channel])

    return search


def test_euvi_sets_drop_far_channel_records(monkeypatch):
    from sunerf.data.download import download_euvi

    t0 = dt.datetime(2012, 6, 1)
    archive = {
        284: [t0, t0 + dt.timedelta(hours=2)],
        171: [t0 + dt.timedelta(minutes=1), t0 + dt.timedelta(hours=1)],
    }
    monkeypatch.setattr(download_euvi, "search_with_retry", _fake_euvi_archive(archive))
    request = DownloadRequest(start=t0, end=t0 + dt.timedelta(days=1), output=Path("unused"))
    selected = download_euvi.select_channel_sets(
        request, source="STEREO_A", channels=(171, 195, 284), cadence=None
    )
    # 195 has no records and is skipped; the 171 record 1 h from the second
    # anchor is too far from it, so only its first match remains.
    assert [len(records) for records in selected] == [1, 2]


def test_euvi_download_skips_chunks_without_anchor_records(monkeypatch, capsys):
    from sunerf.data.download import download_euvi
    from sunerf.data.download.core import DownloadResult

    t0 = dt.datetime(2012, 6, 1)
    archive = {284: [t0 + dt.timedelta(days=8)], 171: [t0 + dt.timedelta(days=8)]}

    def search(_search, *query):
        from sunpy.net import attrs as a

        start = next(q.start for q in query if isinstance(q, a.Time))
        if start.datetime < t0 + dt.timedelta(days=7):
            raise RuntimeError("VSO returned no matching records")
        return _fake_euvi_archive(archive)(_search, *query)

    fetched = []
    monkeypatch.setattr(download_euvi, "search_with_retry", search)
    monkeypatch.setattr(
        download_euvi, "fetch_fido",
        lambda *responses, request, description: fetched.append(responses) or DownloadResult(selected=len(responses)),
    )
    request = DownloadRequest(start=t0, end=t0 + dt.timedelta(days=14), output=Path("unused"))
    result = download_euvi.download(request, channels=(171, 284), sources=("STEREO_A",))
    assert len(fetched) == 1 and result.selected == 2
    assert "1 chunks without EUVI records" in capsys.readouterr().out


def _aia_client(export, records, exports):
    import pandas as pd

    class _Client:
        def __init__(self, email):
            pass

        def query(self, dataset, key):
            return pd.DataFrame(records, columns=["T_REC", "WAVELNTH"])

        def export(self, dataset, protocol):
            exports.append(dataset)
            return export

    return _Client


def test_aia_export_names_match_jsoc():
    from sunerf.data.download.download_aia import export_filename

    assert export_filename("2012-08-01T00:00:01Z", 211) == (
        "aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.fits"
    )


def test_aia_chunk_with_every_record_present_is_not_exported(tmp_path, monkeypatch):
    from sunerf.data.download import download_aia

    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    (prepared / "rejected").mkdir(parents=True)
    raw.mkdir()
    (raw / "aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.fits").write_text("raw")
    (prepared / "aia.lev1_euv_12s.2012-08-01T020001Z.211.image_lev1.prepared.fits").write_text("")
    (prepared / "rejected" / "aia.lev1_euv_12s.2012-08-01T040001Z.211.image_lev1.prepared.fits").write_text("")
    records = [("2012-08-01T00:00:01Z", 211), ("2012-08-01T02:00:01Z", 211), ("2012-08-01T04:00:01Z", 211)]
    exports = []
    monkeypatch.setattr(download_aia.drms, "Client", _aia_client(None, records, exports))
    request = DownloadRequest(start=dt.datetime(2012, 8, 1), end=dt.datetime(2012, 8, 2), output=raw)

    result = download_aia.download(request, email="registered@example.org", prepared_dir=prepared)

    assert exports == [] and result.downloaded == 0


def test_aia_records_with_a_prepared_product_are_not_downloaded(tmp_path, monkeypatch):
    from sunerf.data.download import download_aia

    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    prepared.mkdir()
    raw.mkdir()
    names = [
        "aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.fits",
        "aia.lev1_euv_12s.2012-08-01T020001Z.211.image_lev1.fits",
    ]
    (prepared / "aia.lev1_euv_12s.2012-08-01T000001Z.211.image_lev1.prepared.fits").write_text("")
    records = [("2012-08-01T00:00:01Z", 211), ("2012-08-01T02:00:01Z", 211)]
    exports = []
    monkeypatch.setattr(download_aia.drms, "Client", _aia_client(_Export(names), records, exports))
    request = DownloadRequest(start=dt.datetime(2012, 8, 1), end=dt.datetime(2012, 8, 2), output=raw)

    result = download_aia.download(request, email="registered@example.org", prepared_dir=prepared)

    assert len(exports) == 1
    assert result == core.DownloadResult(
        selected=2, downloaded=1, skipped=1, files=(str(raw / names[1]),)
    )
    assert [path.name for path in raw.iterdir()] == [names[1]]


def test_euvi_records_with_a_raw_file_or_prepared_product_are_dropped(tmp_path):
    from sunerf.data.download import download_euvi

    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    (prepared / "rejected").mkdir(parents=True)
    raw.mkdir()
    t0 = dt.datetime(2012, 6, 1)
    times = [t0 + dt.timedelta(hours=2 * index) for index in range(4)]
    (raw / "20120601_000000_n4euA.fts").write_text("raw")
    (prepared / "20120601_020000_14euA.prepared.fits").write_text("")
    (prepared / "rejected" / "20120601_040000_14euA.prepared.fits").write_text("")
    request = DownloadRequest(start=t0, end=t0 + dt.timedelta(days=1), output=raw)

    remaining, skipped = download_euvi.missing_records(
        (_euvi_records(times[:2]), _euvi_records(times[2:])), request, prepared
    )
    assert skipped == 3
    assert [list(records["fileid"]) for records in remaining] == [
        ["secchi/L0/a/img/euvi/20120601/20120601_060000_n4euA.fts"]
    ]

    # Without the prepared directory only the raw file counts.
    remaining, skipped = download_euvi.missing_records((_euvi_records(times),), request)
    assert skipped == 1 and len(remaining[0]) == 3
    # Overwriting fetches everything.
    overwrite = DownloadRequest(start=request.start, end=request.end, output=raw, overwrite=True)
    assert download_euvi.missing_records((_euvi_records(times),), overwrite, prepared)[1] == 0


def test_euvi_download_fetches_nothing_when_every_record_is_present(tmp_path, monkeypatch):
    from sunerf.data.download import download_euvi

    t0 = dt.datetime(2012, 6, 1)
    archive = {284: [t0], 171: [t0]}
    (tmp_path / "20120601_000000_n4euA.fts").write_text("raw")
    fetched = []
    monkeypatch.setattr(download_euvi, "search_with_retry", _fake_euvi_archive(archive))
    monkeypatch.setattr(
        download_euvi, "fetch_fido",
        lambda *responses, request, description: fetched.append(responses),
    )
    request = DownloadRequest(start=t0, end=t0 + dt.timedelta(days=1), output=tmp_path)
    result = download_euvi.download(request, channels=(171, 284), sources=("STEREO_A",))
    # Both channels map to the same fake fileid, so both records are present.
    assert fetched == [] and result.skipped == 2
