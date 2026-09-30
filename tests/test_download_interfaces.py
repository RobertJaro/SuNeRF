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
