import argparse
import glob
import os.path
from multiprocessing import Pool

import numpy as np
from skimage.util import view_as_blocks
from tqdm import tqdm

from sunerf.data.loader.base_loader import MapDataLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert AIA FITS maps to NPZ files for conditioned SuNeRF."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input glob pattern for FITS files (e.g. '/path/to/aia/*.193.*.fits')",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Number of parallel processes (default: max available cores)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        default=None,
        help="If specified, split maps into patches of this size (height width). If not specified, save full maps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    files = sorted(glob.glob(args.input))
    if len(files) == 0:
        raise RuntimeError(f"No files found for input pattern: {args.input}")

    loader = MapDataLoader(Rs_per_ds=1, reference_frame="carrington", add_hpc=True)

    patch_size = tuple(args.patch_size) if args.patch_size is not None else None
    nproc = args.nproc if args.nproc is not None else os.cpu_count()
    with Pool(processes=nproc) as pool:
        for data in tqdm(pool.imap(loader.load, files), total=len(files)):
            if patch_size is None:  # save full map
                out_file = os.path.join(
                    args.output,
                    f'{data["time"].strftime("%Y%m%d_%H%M%S")}.npz',
                )
                np.savez(out_file, image=data["image"], rays=data["rays"], hpc=data["hpc"],
                         latitude=data["observer"]['latitude'], longitude=data["observer"]['longitude'])
                continue

            # separate into patches
            image_patches = view_as_blocks(data["image"], block_shape=patch_size)
            ray_patches = view_as_blocks(data["rays"], block_shape=patch_size + (2, 3,))
            hpc_patches = view_as_blocks(data["hpc"], block_shape=patch_size + (2,))

            for i in range(image_patches.shape[0]):
                for j in range(image_patches.shape[1]):
                    # check if more than 50% are NaNs
                    if np.isnan(image_patches[i, j]).sum() > 0.5 * image_patches[i, j].size:
                        continue
                    out_file = os.path.join(
                        args.output,
                        f'{data["time"].strftime("%Y%m%d_%H%M%S")}_patch_{i:02d}_{j:02d}.npz',
                    )
                    np.savez(
                        out_file,
                        image=image_patches[i, j],
                        rays=ray_patches[i, j, 0, 0],
                        hpc=hpc_patches[i, j, 0],
                    )


if __name__ == "__main__":
    main()
