"""Download Parker Solar Probe in-situ CDF products from CDAWeb."""

import argparse

from sunerf.data.download.cdaweb import download_cdaweb
from sunerf.data.download.core import add_common_arguments, request_from_args


DEFAULT_DATASETS = (
    "PSP_SWP_SPC_L3I",
    "PSP_FLD_L3_SQTN_RFS_V1V2",
    "PSP_COHO1HR_MERGED_MAG_PLASMA",
)


def download(request, *, datasets=DEFAULT_DATASETS):
    return download_cdaweb(request, datasets, mission="PSP")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    download(request_from_args(args), datasets=args.datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
