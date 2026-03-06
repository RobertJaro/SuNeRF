#!/usr/bin/env python3
"""Download PUNCH FITS files and save a log-scale JPG image sequence."""

import argparse
import os
import re
import subprocess
from urllib.parse import urljoin
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm
from sunpy.visualization.colormaps import cm
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="Source directory URL.")
    parser.add_argument("--out-dir", help="Output folder for JPG sequence.")
    parser.add_argument("--suffix", default="v0h.fits", help="File suffix to match.")
    return parser.parse_args()


def list_matching_files(url: str, suffix: str):
    if not url.endswith("/"):
        url = f"{url}/"
    with urlopen(url) as r:
        html = r.read().decode("utf-8", errors="ignore")
    hrefs = re.findall(r'href="([^"]+)"', html)
    files = sorted(
        {
            h
            for h in hrefs
            if not h.endswith("/")
               and not h.startswith("?")
               and not h.startswith("#")
               and os.path.basename(h).endswith(suffix)
        }
    )
    return [urljoin(url, f) for f in files]


def download_file(file_url: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["wget", "-q", "-nc", "-P", out_dir, file_url]
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, os.path.basename(file_url))


def load_fits_2d(path: str):
    data = fits.getdata(path)
    data = np.asarray(data, dtype=float)
    data = np.squeeze(data)
    if data.ndim > 2:
        data = data.reshape((-1, data.shape[-2], data.shape[-1]))[0]
    if data.ndim != 2:
        raise ValueError(f"{path} does not contain a 2D image after squeeze.")
    return data


def render_jpgs(fits_files, jpg_dir: str, norm: LogNorm):
    os.makedirs(jpg_dir, exist_ok=True)
    for i, fp in enumerate(fits_files):
        img = load_fits_2d(fp)
        img = np.where(img > 0, img, np.nan)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        im = ax.imshow(img, origin="lower", cmap=cm.soholasco2, norm=norm)
        ax.set_title(os.path.basename(fp), fontsize=9)
        ax.set_xlabel("X [pix]")
        ax.set_ylabel("Y [pix]")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity (log scale)")
        frame_path = os.path.join(jpg_dir, f"frame_{i:05d}.jpg")
        fig.tight_layout()
        fig.savefig(frame_path)
        plt.close(fig)


def main():
    args = parse_args()
    out_dir = args.out_dir

    print(f"Listing files at: {args.url}")
    file_urls = list_matching_files(args.url, args.suffix)
    if not file_urls:
        raise SystemExit(f"No files ending with '{args.suffix}' found at {args.url}")
    file_urls = file_urls[:20]
    print(f"Using {len(file_urls)} files (max 20).")

    fits_files = [download_file(u, out_dir) for u in tqdm(file_urls, desc="Downloading FITS")]
    fits_files = sorted(fits_files)
    print(f"Downloaded/available: {len(fits_files)} FITS files in {out_dir}")

    norm = LogNorm()
    render_jpgs(fits_files, out_dir, norm=norm)
    print(f"Saved JPG sequence to: {out_dir}")


if __name__ == "__main__":
    main()
