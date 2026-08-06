#!/usr/bin/env python3
import argparse, glob
import multiprocessing
import os.path
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from astropy.visualization import AsinhStretch, ImageNormalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sunpy.map
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("out_path")
    p.add_argument("--tb", default=None)
    p.add_argument("--pb", default=None)
    p.add_argument("--tb_correction", default=None)
    p.add_argument("--pb_correction", default=None)
    p.add_argument("--a", type=float, default=1e-4)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--size", type=float, default=4.2)
    p.add_argument("--num_workers", type=int, default=16)
    return p.parse_args()


def stem(p):
    return Path(p).stem.replace(" ", "_")


def load_correction(path):
    return np.load(path) if path else None


def first_frame_max(file_path, correction=None):
    data = np.asarray(sunpy.map.Map(file_path).data, float)
    if correction is not None:
        data = data - correction
    finite_data = data[np.isfinite(data)]
    if finite_data.size == 0:
        raise ValueError(f"No finite pixels found in first frame: {file_path}")
    return finite_data.max()


class FrameRenderer:
    def __init__(self, out_dir, norm_by_kind, correction_by_kind, dpi, size):
        self.out_dir = Path(out_dir)
        self.norm_by_kind = norm_by_kind
        self.correction_by_kind = correction_by_kind
        self.cmap = plt.get_cmap("inferno").copy()
        self.cmap.set_bad("green")
        self.dpi = dpi
        self.size = size

    def __call__(self, row):
        tb_fp, pb_fp = row
        maps = []
        labels = []
        stems = []
        kinds = []

        if tb_fp:
            m = sunpy.map.Map(tb_fp)
            maps.append(m)
            labels.append(f"tB: {m.date.isot if m.date else stem(tb_fp)}")
            stems.append(stem(tb_fp))
            kinds.append("tb")

        if pb_fp:
            m = sunpy.map.Map(pb_fp)
            maps.append(m)
            labels.append(f"pB: {m.date.isot if m.date else stem(pb_fp)}")
            stems.append(stem(pb_fp))
            kinds.append("pb")

        n = len(maps)
        fig = plt.figure(figsize=(self.size * n, self.size), dpi=self.dpi)

        for i, (m, title, kind) in enumerate(zip(maps, labels, kinds), start=1):
            ax = fig.add_subplot(1, n, i, projection=m.wcs)
            data = np.asarray(m.data, float)
            correction = self.correction_by_kind.get(kind)
            if correction is not None:
                data = data - correction
            data = np.ma.masked_invalid(data)
            im = ax.imshow(data, norm=self.norm_by_kind[kind], cmap=self.cmap, origin="lower")
            m.draw_limb(axes=ax, color="red", linewidth=1.2)
            ax.set_title(title, fontsize=9, pad=6)

            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
            fig.colorbar(im, cax=cax)

            ax.set_xlabel("Tx [arcsec]")
            ax.set_ylabel("Ty [arcsec]")

        out_name = f"{stems[0]}__{stems[1]}.jpg" if len(stems) == 2 else f"{stems[0]}.jpg"
        out_path = self.out_dir / out_name
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return str(out_path)


def main():
    a = parse_args()
    if not a.tb and not a.pb:
        raise SystemExit("Provide --tb and/or --pb")

    tb_files = sorted(glob.glob(a.tb)) if a.tb else []
    pb_files = sorted(glob.glob(a.pb)) if a.pb else []
    correction_by_kind = {
        "tb": load_correction(a.tb_correction),
        "pb": load_correction(a.pb_correction),
    }

    if tb_files and pb_files and len(tb_files) != len(pb_files):
        raise ValueError(f"tB/pB list lengths differ: {len(tb_files)} vs {len(pb_files)}")

    rows = list(zip(tb_files, pb_files)) if (tb_files and pb_files) else \
           [(f, None) for f in tb_files] if tb_files else \
           [(None, f) for f in pb_files]

    out_dir = Path(a.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_by_kind = {
        kind: ImageNormalize(
            vmin=0,
            vmax=first_frame_max(files[0], correction_by_kind[kind]),
            stretch=AsinhStretch(a.a),
            clip=False,
        )
        for kind, files in (("tb", tb_files), ("pb", pb_files))
        if files
    }

    renderer = FrameRenderer(
        out_dir=str(out_dir), norm_by_kind=norm_by_kind, correction_by_kind=correction_by_kind,
        dpi=a.dpi, size=a.size,
    )

    if a.num_workers is None or a.num_workers <= 1:
        for row in tqdm(rows, total=len(rows), desc="Rendering frames"):
            renderer(row)
    else:
        with multiprocessing.Pool(a.num_workers) as pool:
            for _ in tqdm(pool.imap(renderer, rows), total=len(rows), desc="Rendering frames"):
                pass

    # create zip of all images
    shutil.make_archive(a.out_path, 'zip', a.out_path)


if __name__ == "__main__":
    main()
