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
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--a", type=float, default=1e-4)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--size", type=float, default=4.2)
    p.add_argument("--num_workers", type=int, default=16)
    return p.parse_args()


def stem(p):
    return Path(p).stem.replace(" ", "_")


def first_frame_range(file_path, vmin=None, vmax=None):
    if vmin is not None and vmax is not None:
        return vmin, vmax

    data = np.asarray(sunpy.map.Map(file_path).data, float)
    finite = np.isfinite(data)
    if not np.any(finite):
        raise ValueError(f"No finite pixels found to determine display range: {file_path}")

    data_min = float(np.nanmin(data[finite]))
    data_max = float(np.nanmax(data[finite]))
    return data_min if vmin is None else vmin, data_max if vmax is None else vmax


class InvalidPreservingImageNormalize(ImageNormalize):
    def __call__(self, values, clip=None, invalid=None):
        invalid_mask = np.ma.getmaskarray(np.ma.masked_invalid(values))
        normalized = super().__call__(values, clip=clip, invalid=invalid)
        return np.ma.array(normalized, mask=np.ma.getmaskarray(normalized) | invalid_mask)


class FrameRenderer:
    def __init__(self, out_dir, norm_by_kind, dpi, size):
        self.out_dir = Path(out_dir)
        self.norm_by_kind = norm_by_kind
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
            data = np.ma.masked_invalid(np.asarray(m.data, float))
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

    if tb_files and pb_files and len(tb_files) != len(pb_files):
        raise ValueError(f"tB/pB list lengths differ: {len(tb_files)} vs {len(pb_files)}")

    rows = list(zip(tb_files, pb_files)) if (tb_files and pb_files) else \
           [(f, None) for f in tb_files] if tb_files else \
           [(None, f) for f in pb_files]

    out_dir = Path(a.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_by_kind = {}
    if tb_files:
        tb_vmin, tb_vmax = first_frame_range(tb_files[0], vmin=a.vmin, vmax=a.vmax)
        norm_by_kind["tb"] = InvalidPreservingImageNormalize(
            vmin=tb_vmin, vmax=tb_vmax, stretch=AsinhStretch(float(a.a)), clip=True
        )
    if pb_files:
        pb_vmin, pb_vmax = first_frame_range(pb_files[0], vmin=a.vmin, vmax=a.vmax)
        norm_by_kind["pb"] = InvalidPreservingImageNormalize(
            vmin=pb_vmin, vmax=pb_vmax, stretch=AsinhStretch(float(a.a)), clip=True
        )

    renderer = FrameRenderer(out_dir=str(out_dir), norm_by_kind=norm_by_kind, dpi=a.dpi, size=a.size)

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
