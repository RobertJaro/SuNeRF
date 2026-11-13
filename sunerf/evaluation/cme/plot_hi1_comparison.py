"""Qualitative comparison of STEREO-A HI-1 browse images with the SuNeRF reconstruction.

For every HI-1 Level-0 FITS file the model is rendered on the field of view of its header
(pointing, pixel grid and observer position), and the clean tB render is shown next to the
browse image of the STEREO Science Center with the same time stamp.  The browse images are the
standard quicklook processing of the SSC; nothing is calibrated or background-subtracted here,
so the comparison is of morphology and timing only.  The log of the render is normalized to the
mean and standard deviation of the gray browse image (robust statistics of the central
percentiles, so that stars and the saturated planet do not count), and both are shown on the
same gray scale.

Output: ``<out_path>/hi1_<time>.jpg`` per frame.  ``--binning`` renders on a coarser pixel grid.

Example::

    python -m sunerf.evaluation.cme.plot_hi1_comparison --sunerf_path /path/save_state.snf \
        --fits "/path/L0/*.fts" --browse_path /path/browse --out_path /path/hi1
"""
import argparse
import glob
import os
from datetime import datetime

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from PIL import Image
from sunpy.map import Map

from sunerf.evaluation.loader import ThomsonSuNeRFLoader

# percentile range of the robust mean and standard deviation (excludes stars and the planet)
STATISTICS_PERCENTILES = (5.0, 95.0)
# displayed range in standard deviations around the mean
DISPLAY_SIGMA = 3.0


def binned_header(header, binning):
    """Header of the same field of view with ``binning`` times larger pixels."""
    header = dict(header)
    for axis in (1, 2):
        header[f'NAXIS{axis}'] = int(header[f'NAXIS{axis}'] // binning)
        header[f'CRPIX{axis}'] = (header[f'CRPIX{axis}'] - 0.5) / binning + 0.5
        header[f'CDELT{axis}'] = header[f'CDELT{axis}'] * binning
    return header


def robust_statistics(values):
    """Mean and standard deviation of the values between the statistics percentiles."""
    low, high = np.nanpercentile(values, STATISTICS_PERCENTILES)
    inside = values[(values >= low) & (values <= high)]
    return inside.mean(), inside.std()


def normalized(values, reference):
    """``values`` rescaled to the robust mean and standard deviation of ``reference``."""
    mean, std = robust_statistics(values)
    reference_mean, reference_std = robust_statistics(reference)
    return (values - mean) / std * reference_std + reference_mean


def render(loader, fits_file, binning):
    """Clean tB [MSB] of the reconstruction on the (binned) field of view of the HI-1 file."""
    header = binned_header(fits.getheader(fits_file), binning)
    ref_map = Map(np.zeros((header['NAXIS2'], header['NAXIS1']), dtype=np.float32), header)
    return loader.load_map(ref_map, progress=False)['tB_map'].data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', required=True, help='SuNeRF save state')
    parser.add_argument('--fits', required=True, help='Glob of HI-1 Level-0 FITS files (geometry)')
    parser.add_argument('--browse_path', required=True, help='Directory of the SSC browse JPEGs')
    parser.add_argument('--out_path', required=True, help='Output directory')
    parser.add_argument('--time_range', nargs=2, metavar=('START', 'END'), default=None,
                        type=lambda v: datetime.fromisoformat(v), help='Only frames inside this period')
    parser.add_argument('--binning', type=int, default=1, help='Pixel binning of the render')
    args = parser.parse_args(argv)
    os.makedirs(args.out_path, exist_ok=True)

    loader = ThomsonSuNeRFLoader(args.sunerf_path)
    for fits_file in sorted(glob.glob(args.fits)):
        stamp = os.path.basename(fits_file)[:15]
        time = datetime.strptime(stamp, '%Y%m%d_%H%M%S')
        if args.time_range is not None and not args.time_range[0] <= time <= args.time_range[1]:
            continue
        browse = glob.glob(os.path.join(args.browse_path, f'{stamp}*.jpg'))
        if not browse:
            print(f'No browse image for {stamp}; skipped')
            continue
        tB = render(loader, fits_file, args.binning)
        # gray browse image (the browse JPEG has the first row at the top)
        observed = np.asarray(Image.open(browse[0]).convert('L'), dtype=np.float32)[::-1]
        with np.errstate(divide='ignore'):
            log_tB = np.log10(np.where(tB > 0, tB, np.nan))
        model = normalized(log_tB, observed)
        mean, std = robust_statistics(observed)
        limits = dict(vmin=mean - DISPLAY_SIGMA * std, vmax=mean + DISPLAY_SIGMA * std)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        axes[0].imshow(observed, origin='lower', cmap='gray', **limits)
        axes[0].set_title('STEREO-A HI-1 browse (SSC)')
        axes[1].imshow(model, origin='lower', cmap='gray', **limits)
        axes[1].set_title('SuNeRF log tB, normalized to the browse image')
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(fits.getheader(fits_file)['DATE-OBS'][:16].replace('T', ' '))
        out_file = os.path.join(args.out_path, f'hi1_{stamp}.jpg')
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f'Figure written to {out_file}', flush=True)


if __name__ == '__main__':
    main()
