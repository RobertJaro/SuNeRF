"""Uniform data-download interfaces for supported SuNeRF instruments."""

from sunerf.data.download.core import DownloadRequest, DownloadResult, combine_results

__all__ = ["DownloadRequest", "DownloadResult", "combine_results"]
