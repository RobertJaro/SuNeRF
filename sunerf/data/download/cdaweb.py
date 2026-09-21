"""Shared CDAWeb downloader implementation."""

from sunpy.net import Fido, attrs as a

from sunerf.data.download.core import DownloadRequest, combine_results, fetch_fido


def download_cdaweb(request: DownloadRequest, datasets, *, mission: str):
    results = []
    time = a.Time(request.start, request.end)
    for dataset in datasets:
        response = Fido.search(time, a.cdaweb.Dataset(dataset))
        if sum(len(block) for block in response) == 0:
            print(f"No {mission} records found for CDAWeb dataset {dataset}")
            continue
        results.append(fetch_fido(
            response,
            request=request,
            description=f"{mission} dataset {dataset}",
        ))
    return combine_results(results)
