#!/usr/bin/env python3
import argparse, glob
import os.path
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sunpy.map
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("out_path")
    p.add_argument("--tb", default=None)
    p.add_argument("--pb", default=None)
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--size", type=float, default=4.2)
    return p.parse_args()


def first_log_range(maps):
    for m in maps:
        d = np.asarray(m.data, float)
        good = np.isfinite(d) & (d > 0)
        if np.any(good):
            vmin, vmax = float(d[good].min()), float(d[good].max())
            if vmin > 0 and vmax > vmin:
                return vmin, vmax
    raise ValueError("No finite positive pixels found to determine LogNorm range.")


def stem(p):
    return Path(p).stem.replace(" ", "_")


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

    # One shared LogNorm for all images
    preview_maps = []
    if tb_files:
        preview_maps.append(sunpy.map.Map(tb_files[0]))
    if pb_files:
        preview_maps.append(sunpy.map.Map(pb_files[0]))

    vmin0, vmax0 = first_log_range(preview_maps)
    vmin = vmin0 if a.vmin is None else float(a.vmin)
    vmax = vmax0 if a.vmax is None else float(a.vmax)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    out_dir = Path(a.out_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for tb_fp, pb_fp in tqdm(rows):
        maps = []
        labels = []
        stems = []

        if tb_fp:
            m = sunpy.map.Map(tb_fp)
            maps.append(m)
            labels.append(f"tB: {m.date.isot if m.date else stem(tb_fp)}")
            stems.append(stem(tb_fp))

        if pb_fp:
            m = sunpy.map.Map(pb_fp)
            maps.append(m)
            labels.append(f"pB: {m.date.isot if m.date else stem(pb_fp)}")
            stems.append(stem(pb_fp))

        n = len(maps)
        fig = plt.figure(figsize=(a.size * n, a.size), dpi=a.dpi)

        axes = []
        ims = []

        for i, (m, title) in enumerate(zip(maps, labels), start=1):
            ax = fig.add_subplot(1, n, i, projection=m.wcs)
            im = ax.imshow(m.data, norm=norm, cmap="inferno", origin="lower")
            m.draw_limb(axes=ax, color="red", linewidth=1.2)
            ax.set_title(title, fontsize=9, pad=6)

            # plot colorbar
            cax = make_axes_locatable(ax, ).append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
            cbar = fig.colorbar(im, cax=cax)

            # Static helioprojective labels
            ax.set_xlabel("Tx [arcsec]")
            ax.set_ylabel("Ty [arcsec]")


        out_name = f"{stems[0]}__{stems[1]}.jpg" if len(stems) == 2 else f"{stems[0]}.jpg"
        fig.tight_layout()
        fig.savefig(out_dir / out_name, bbox_inches="tight")
        plt.close(fig)

    # create zip of all images
    shutil.make_archive(a.out_path, 'zip', a.out_path)


if __name__ == "__main__":
    main()
