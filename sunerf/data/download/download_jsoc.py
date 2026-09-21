"""Shared JSOC export helper used by instrument downloaders."""

from pathlib import Path

from sunerf.data.download.core import DownloadResult


def download_jsoc_export(dataset, output, client, *, process=None, dry_run=False):
    output = Path(output)
    if dry_run:
        print(dataset)
        return DownloadResult(selected=1)
    output.mkdir(parents=True, exist_ok=True)
    export = client.export(dataset, protocol="fits", process=process)
    export.wait()
    result = export.download(str(output))
    paths = tuple(map(str, result.download))
    return DownloadResult(selected=len(paths), downloaded=len(paths), files=paths)
