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
    dataset = aia_dataset(request, 171, parse_cadence("6h"))
    assert dataset == (
        "aia.lev1_euv_12s[2012-08-01_00:00:00 / 86400s@21600s]"
        "[171]{image}"
    )
