"""Export tB and pB images of a SuNeRF reconstruction rendered for the PSI reference views.

Every view is rendered for one frame of its image series, with the geometry,
field of view and time of the PSI image ``<data_path>/<view>/pb/*<frame>.fts``;
the learned detector terms are not applied.  The images are written in the units
of the PSI images (MSB) with the names of the reference files::

    <out_path>/images/<view>/tb/<view>_tb0150.npy
    <out_path>/images/<view>/pb/<view>_pb0150.npy

Example::

    python -m sunerf.evaluation.cme.export_images \
        --sunerf_path /path/save_state.snf --data_path /path/clean \
        --frames L4=153 L5_wide=153 P1=153 --out_path /path/export
"""
import argparse
import os
from pathlib import Path

import numpy as np
from sunpy.map import Map

from sunerf.data.psi.psi_test_paths import PRODUCTS, channel_file
from sunerf.evaluation.cme.export_layout import image_file, view_frame
from sunerf.evaluation.loader import ThomsonSuNeRFLoader

OUTPUT_MAP_KEYS = {'tb': 'tB_map', 'pb': 'pB_map'}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--data_path', type=Path, required=True, help='Directory with <view>/{tb,pb}/*.fts')
    parser.add_argument('--out_path', required=True, help='Output directory')
    parser.add_argument('--frames', type=view_frame, nargs='+', required=True, metavar='VIEW=FRAME',
                        help='Frame of the image series to render for each view, e.g. L4=153 P1=153')
    parser.add_argument('--device', default=None, help='torch device (default: cuda if available)')
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--trusted', action='store_true', help='Load a pickled (non weights-only) save state')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    loader = ThomsonSuNeRFLoader(args.sunerf_path, device=args.device, trusted=args.trusted)
    for view, frame in args.frames:
        # tB and pB share the geometry of the view; the pB file is the reference
        reference = Map(channel_file(args.data_path, view, 'pb', f'{frame:04d}'))
        rendered = loader.load_map(reference, instrument_key=view, batch_size=args.batch_size)
        for product in PRODUCTS:
            file = image_file(args.out_path, view, product, frame)
            os.makedirs(os.path.dirname(file), exist_ok=True)
            np.save(file, np.asarray(rendered[OUTPUT_MAP_KEYS[product]].data, dtype=np.float32))


if __name__ == '__main__':
    main()
