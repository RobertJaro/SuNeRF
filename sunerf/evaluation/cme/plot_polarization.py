"""Degree of polarization pB/tB of the input images against the forward-rendered reconstruction.

``--n_frames`` times are sampled from ``--time_range``; every instrument contributes its frame
closest to each.  One figure per sampled time (``frames/polarization_<k>.jpg``) and the main
figure ``polarization.jpg`` for the sampled time closest to ``--time``, for PUNCH, CCOR,
STEREO-A COR2, LASCO C2 and LASCO C3::

    top     rows tB, pB, pB/tB; columns observation and SuNeRF render of every instrument
    bottom  pB/tB against the projected distance, mean and standard deviation of the pixels in
            bins of the distance, with the Thomson limit and the range expected for a smooth
            corona, from the pixels of all sampled frames in every figure

The render is the clean forward model without the learned detector terms, shown for the full
field between the inner and the outer radius of the observation (no gaps of the observed image);
the radial profiles use only the pixels of the observation.  CCOR, LASCO C2 and LASCO C3
use tB observations only, so their rendered pB and pB/tB are predictions.  The maps highlight
and count pB/tB values below zero or above the Thomson limit without clipping them.

The renders (GPU) are cached per instrument and sampled time in
``<out_path>/cache/<instrument>_<k>.npz``; the figures are redrawn from the cache unless
``--overwrite`` is given.
"""
import argparse
import glob
import os

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from sunpy.map import Map, all_coordinates_from_map
from sunpy.visualization.colormaps import cm

from sunerf.data.review_masks import load_reviewed_map
from sunerf.data.coronagraph.brightness_pairs import exclude_rejected_partners
from sunerf.data.ray_sampling import hpc_impact_parameter
from sunerf.evaluation.cme.plot_ref_series import normalize_datetime, read_observation_date

# display name, instrument key of the save state
INSTRUMENTS = {'punch': ('PUNCH', 'PUNCH'), 'ccor': ('CCOR', 'CCOR'), 'cor2': ('COR2', 'STEREO_A_COR2'),
               'lasco_c2': ('LASCO C2', 'LASCO_C2'), 'lasco_c3': ('LASCO C3', 'LASCO_C3')}
LIMB_DARKENING = 0.63
N_BINS = 40
# format of the cached frames; a cache of another version is rendered again
CACHE_VERSION = 3
# linear width of the asinh scale of the tB and pB maps, in the units of the training images
# (MSB / MSB_DIVISOR, the scaling of the coronagraph data sets)
ASINH_WIDTH = 1e-5
MSB_DIVISOR = 1e-9
# radial fall-off n_e ~ r^-alpha of a smooth corona, for the expected range of pB/tB
SMOOTH_CORONA_ALPHA = (2.0, 6.0)


def thomson_limit(radius_rsun, limb_darkening=LIMB_DARKENING):
    """pB/tB of an electron in the plane of the sky at ``radius_rsun`` (van de Hulst coefficients).

    This is the upper limit of the degree of polarization of a line of sight with this impact
    parameter; it approaches one far from the Sun.
    """
    radius_rsun = np.asarray(radius_rsun, dtype=np.float64)
    sin_omega = 1.0 / radius_rsun
    cos_omega = np.sqrt(1.0 - sin_omega ** 2)
    log_term = np.log((1.0 + sin_omega) / cos_omega)
    a = cos_omega * sin_omega ** 2
    b = -(1.0 - 3.0 * sin_omega ** 2 - cos_omega ** 2 / sin_omega * (1.0 + 3.0 * sin_omega ** 2) * log_term) / 8.0
    c = 4.0 / 3.0 - cos_omega - cos_omega ** 3 / 3.0
    d = (5.0 + sin_omega ** 2 - cos_omega ** 2 / sin_omega * (5.0 - sin_omega ** 2) * log_term) / 8.0
    polarized = (1.0 - limb_darkening) * a + limb_darkening * b
    tangential = (1.0 - limb_darkening) * c + limb_darkening * d
    return polarized / (2.0 * tangential - polarized)


def smooth_corona_polarization(alpha):
    """pB/tB of a spherically symmetric corona with ``n_e ~ r^-alpha`` far from the Sun.

    Thomson scattering of a point source integrated along an infinite line of sight; the value
    does not depend on the projected distance.
    """
    return (alpha + 1.0) / (alpha + 3.0)


def polarization(pB, tB):
    """pB/tB without clipping; NaN where either image is missing or tB is not positive."""
    ratio = np.full(tB.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(pB) & np.isfinite(tB) & (tB > 0)
    ratio[valid] = pB[valid] / tB[valid]
    return ratio


def list_frames(tB_pattern, pB_pattern, tB_correction, pB_correction):
    """Observation times and files of one instrument; tB and pB files are paired in sorted order."""
    tB_paths = sorted(p for p in glob.glob(tB_pattern) if os.path.isfile(p))
    if not tB_paths:
        raise ValueError(f'No files found for {tB_pattern}')
    pB_paths = sorted(p for p in glob.glob(pB_pattern) if os.path.isfile(p)) if pB_pattern is not None else [None] * len(tB_paths)
    if pB_pattern is not None:
        tB_paths, pB_paths = exclude_rejected_partners(tB_paths, pB_paths)
    if len(pB_paths) != len(tB_paths):
        raise ValueError(f'{len(tB_paths)} tB files but {len(pB_paths)} pB files for {tB_pattern}')
    times = np.array([read_observation_date(path) for path in tB_paths])
    return {'times': times, 'tB': tB_paths, 'pB': pB_paths,
            'tB_correction': None if tB_correction is None else np.load(tB_correction),
            'pB_correction': None if pB_correction is None else np.load(pB_correction)}


def load_observation(path, correction):
    """Prepared image as it enters the training: static correction image subtracted, solar north up."""
    s_map = load_reviewed_map(path)
    if correction is not None:
        s_map = Map(s_map.data - correction, s_map.meta)
    return s_map.rotate(order=3)


def load_frame(loader, frames, index, instrument_key):
    """Observed and rendered tB and pB of one frame with the projected distance of every pixel."""
    tB_map = load_observation(frames['tB'][index], frames['tB_correction'])
    observed_tB = np.array(tB_map.data, dtype=np.float32)
    observed_pB = None
    if frames['pB'][index] is not None:
        observed_pB = np.array(load_observation(frames['pB'][index], frames['pB_correction']).data, dtype=np.float32)

    coords = all_coordinates_from_map(tB_map)
    radius = hpc_impact_parameter(coords.Tx, coords.Ty, tB_map.dsun).to_value(u.R_sun).astype(np.float32)
    # image plane in projected solar radii: tangent-plane pixel offsets times the observer distance
    rsun_per_pixel = [float(scale.to_value(u.rad / u.pix)) * float(tB_map.dsun.to_value(u.R_sun))
                      for scale in tB_map.scale]
    center = [float(tB_map.reference_pixel.x.to_value(u.pix)), float(tB_map.reference_pixel.y.to_value(u.pix))]
    ny, nx = tB_map.data.shape
    extent = np.array([(-0.5 - center[0]) * rsun_per_pixel[0], (nx - 0.5 - center[0]) * rsun_per_pixel[0],
                       (-0.5 - center[1]) * rsun_per_pixel[1], (ny - 0.5 - center[1]) * rsun_per_pixel[1]])
    # the render covers the full field between the inner and the outer radius of the observation,
    # without the gaps of the observed image
    observed_radius = radius[np.isfinite(observed_tB)]
    field = (radius >= observed_radius.min()) & (radius <= observed_radius.max())
    field_map = Map(np.where(field, 0.0, np.nan), tB_map.meta)
    model_out = loader.load_map(field_map, progress=False, filter_occ=True, instrument_key=instrument_key)
    model_tB = np.where(field, model_out['tB_map'].data, np.nan).astype(np.float32)
    model_pB = np.where(field, model_out['pB_map'].data, np.nan).astype(np.float32)
    return {'radius': radius, 'extent': extent, 'observed_tB': observed_tB, 'observed_pB': observed_pB,
            'model_tB': model_tB, 'model_pB': model_pB}


def radial_statistics(ratio, radius, bin_edges):
    """Pixel count, sum and sum of squares of pB/tB in bins of the projected distance, and the pixel
    counts (outside the physical range, total).  The sums of several frames add up."""
    valid = np.isfinite(ratio) & (radius > 1.0)
    index = np.clip(np.digitize(radius[valid], bin_edges) - 1, 0, len(bin_edges) - 2)
    values = ratio[valid]
    statistics = np.stack([np.bincount(index, minlength=len(bin_edges) - 1),
                           np.bincount(index, weights=values, minlength=len(bin_edges) - 1),
                           np.bincount(index, weights=values ** 2, minlength=len(bin_edges) - 1)], axis=-1)
    unphysical = (values < 0) | (values > thomson_limit(radius[valid]))
    return statistics, np.array([unphysical.sum(), values.size])


def mean_and_std(statistics):
    """Mean and standard deviation per bin from the count, sum and sum of squares; NaN below 10 pixels."""
    count, total, squares = statistics.T
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count >= 10, total / count, np.nan)
        std = np.sqrt(np.clip(squares / count - mean ** 2, 0, None))
    return mean, std


def sample_times(time_range, n_frames):
    """Equally spaced times of the sampled frames."""
    start, end = time_range
    return [start + (end - start) * i / max(n_frames - 1, 1) for i in range(n_frames)]


def evaluate_frame(loader, frames, instrument_key, time):
    """Maps and radial statistics of the frame of one instrument closest to ``time``."""
    index = int(np.argmin([abs((t - time).total_seconds()) for t in frames['times']]))
    print(f'{instrument_key}: frame {frames["times"][index]} for {time}', flush=True)
    frame = load_frame(loader, frames, index, instrument_key)
    observed_pixels = np.isfinite(frame['observed_tB'])
    bin_edges = np.linspace(frame['radius'][observed_pixels].min(), frame['radius'][observed_pixels].max(),
                            N_BINS + 1)
    result = {'cache_version': CACHE_VERSION, 'time': frames['times'][index].isoformat(), 'bin_edges': bin_edges,
              **{key: value for key, value in frame.items() if value is not None}}
    # the profiles compare the same pixels: the render is restricted to those of the observation
    ratios = {'model': np.where(observed_pixels, polarization(frame['model_pB'], frame['model_tB']), np.nan)}
    if frame['observed_pB'] is not None:
        ratios['observed'] = polarization(frame['observed_pB'], frame['observed_tB'])
    for key, ratio in ratios.items():
        result[f'{key}_profile'], result[f'{key}_counts'] = radial_statistics(ratio, frame['radius'], bin_edges)
    return result


def average_profiles(frames):
    """Statistics of the pixels of all sampled frames together."""
    averaged = {'bin_edges': frames[0]['bin_edges']}
    for key in ('observed', 'model'):
        if f'{key}_profile' in frames[0]:
            averaged[f'{key}_profile'] = np.sum([frame[f'{key}_profile'] for frame in frames], axis=0)
            averaged[f'{key}_counts'] = np.sum([frame[f'{key}_counts'] for frame in frames], axis=0)
    return averaged


def brightness_norm(data):
    """Asinh scale from zero shared by tB and pB of the observation and the render of one instrument."""
    values = np.concatenate([data[key][np.isfinite(data[key])]
                             for key in ('observed_tB', 'observed_pB', 'model_tB', 'model_pB') if key in data])
    return AsinhNorm(linear_width=ASINH_WIDTH * MSB_DIVISOR, vmin=0.0, vmax=np.percentile(values, 99.9))


def plot(results, profiles, out_file):
    """Figure of the maps of ``results[name]`` and the radial profiles of ``profiles[name]``."""
    names = [name for name in INSTRUMENTS if name in results]
    n = len(names)
    # manual layout: square map panels with minimal gaps; narrow spacer columns hold the y tick
    # labels of every instrument, a spacer row the x tick labels of the bottom map row
    width_ratios = sum(([1.0, 1.0, 0.22] for _ in names), [])[:-1]
    fig = plt.figure(figsize=(4.4 * n, 10.0))
    gs = fig.add_gridspec(5, 3 * n - 1, width_ratios=width_ratios, height_ratios=(1, 1, 1, 0.2, 1.3),
                          left=0.06, right=0.94, bottom=0.05, top=0.86, wspace=0.04, hspace=0.04)
    brightness_cmap = cm.soholasco2.copy()
    brightness_cmap.set_bad('black')
    # pixels outside the physical range stand out from the scale
    ratio_cmap = plt.get_cmap('viridis').copy()
    ratio_cmap.set_under('magenta')
    ratio_cmap.set_over('red')
    ratio_cmap.set_bad('black')

    pair_axes, profile_axes = [], []
    for column, name in enumerate(names):
        data, label = results[name], INSTRUMENTS[name][0]
        profile = profiles[name]
        norm = brightness_norm(data)
        first = 3 * column
        # the axes share x in the columns and y in the rows of the instrument
        axes = np.empty((3, 2), dtype=object)
        for row in range(3):
            for side in range(2):
                axes[row, side] = fig.add_subplot(gs[row, first + side],
                                                  sharex=axes[0, side] if row else None,
                                                  sharey=axes[row, 0] if side else None)
        pair_axes.append(axes)
        extent = data['extent']
        half_size = np.abs(extent).max()
        images = {'observed': (data.get('observed_tB'), data.get('observed_pB')),
                  'model': (data['model_tB'], data['model_pB'])}
        for side, (key, title) in enumerate((('observed', label), ('model', f'SuNeRF-{label}'))):
            tB, pB = images[key]
            ratio = None if pB is None else polarization(pB, tB)
            limit = thomson_limit(np.clip(data['radius'], 1.0 + 1e-6, None))
            for row, image in enumerate((tB, pB, ratio)):
                ax = axes[row, side]
                if image is None:
                    ax.set_facecolor('0.85')
                    ax.text(0.5, 0.5, 'not observed', transform=ax.transAxes, ha='center', va='center', fontsize=9)
                else:
                    # the square window shows black outside a non-square image
                    ax.set_facecolor('black')
                    ax.add_patch(Circle((0.0, 0.0), 1.0, color='yellow', linewidth=0, zorder=5))
                    if row < 2:
                        im_brightness = ax.imshow(image, origin='lower', cmap=brightness_cmap, norm=norm,
                                                  extent=extent)
                    else:
                        # the Thomson limit depends on the distance: values above it are set above the scale
                        shown = np.where(image > limit, 2.0, image)
                        im_ratio = ax.imshow(shown, origin='lower', cmap=ratio_cmap, vmin=0, vmax=1, extent=extent)
                ax.set_xlim(-half_size, half_size)
                ax.set_ylim(-half_size, half_size)
                ax.set_aspect('equal')
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.tick_params(labelsize=8, labelbottom=row == 2, labelleft=side == 0)
            # fixed position: the automatic one moves the title above the colorbar inset
            axes[0, side].set_title(title, y=1.0, pad=6)
        axes[2, 0].set_xlabel(r'HPC-x [R$_\odot$]', fontsize=9, x=1.0 + 0.5 * 0.04)
        # brightness colorbar above the instrument, spanning both columns
        cax = axes[0, 0].inset_axes([0.0, 1.3, 2.0 + 0.04, 0.05])
        colorbar = fig.colorbar(im_brightness, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        colorbar.set_label(f'{label} tB, pB [MSB]', fontsize=9)
        cax.tick_params(labelsize=8)
        # decades from the linear width to the maximum
        decades = 10.0 ** np.arange(np.ceil(np.log10(ASINH_WIDTH * MSB_DIVISOR)), np.log10(norm.vmax) + 1e-6)
        colorbar.set_ticks(np.concatenate([[0.0], decades]))

        ax = fig.add_subplot(gs[4, first:first + 2], sharey=profile_axes[0] if profile_axes else None)
        profile_axes.append(ax)
        centers = 0.5 * (profile['bin_edges'][1:] + profile['bin_edges'][:-1])
        # band of the smooth corona, colored by the fall-off alpha that gives this pB/tB
        band = np.linspace(*(smooth_corona_polarization(alpha) for alpha in SMOOTH_CORONA_ALPHA), 256)
        band_alpha = (3.0 * band - 1.0) / (1.0 - band)
        im_alpha = ax.imshow(band_alpha[:, None], origin='lower', aspect='auto', cmap='Greens_r', alpha=0.6,
                             extent=(profile['bin_edges'][0], profile['bin_edges'][-1], band[0], band[-1]),
                             vmin=SMOOTH_CORONA_ALPHA[0], vmax=SMOOTH_CORONA_ALPHA[1] + 1.0, zorder=0)
        ax.plot(centers, thomson_limit(centers), color='black', ls='--', lw=1, label='Thomson limit')
        note = []
        for key, color, curve_label in (('observed', 'black', label), ('model', 'tab:orange', f'SuNeRF-{label}')):
            if f'{key}_profile' not in profile:
                continue
            mean, std = mean_and_std(profile[f'{key}_profile'])
            ax.fill_between(centers, mean - std, mean + std, color=color, alpha=0.2, lw=0)
            ax.plot(centers, mean, color=color, lw=1.4, label=curve_label)
            unphysical, total = profile[f'{key}_counts']
            note.append(f'{curve_label}: {100 * unphysical / total:.1f} %')
        ax.axhline(0, color='0.5', lw=0.6)
        ax.text(0.03, 0.04, 'outside physical range\n' + '\n'.join(note), transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8)
        ax.set_xlim(profile['bin_edges'][0], profile['bin_edges'][-1])
        ax.set_xlabel(r'Projected distance [R$_\odot$]')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.25)
        if column > 0:
            ax.tick_params(labelleft=False)
    for row, product in enumerate(('tB', 'pB', 'pB/tB')):
        pair_axes[0][row, 0].set_ylabel(f'{product}\nHPC-y [R$_\\odot$]')
    profile_axes[0].set_ylabel('pB/tB')
    profile_axes[0].set_ylim(-0.2, 1.2)
    fig.colorbar(im_ratio, cax=pair_axes[-1][2, 1].inset_axes([1.06, 0.0, 0.05, 1.0]), extend='both', label='pB/tB')
    fig.colorbar(im_alpha, cax=profile_axes[-1].inset_axes([1.03, 0.0, 0.025, 1.0]),
                 boundaries=np.linspace(*SMOOTH_CORONA_ALPHA, 256), ticks=np.arange(2, 7),
                 label=r'Smooth corona, $n_e \propto r^{-\alpha}$: $\alpha$')
    fig.suptitle(normalize_datetime(results[names[0]]['time']).isoformat(' ', timespec='minutes'), fontsize=14,
                 y=0.985)

    # the map cells are square: scale the figure height so that a map panel fills its cell
    fig.canvas.draw()
    bbox = pair_axes[0][0, 0].get_position()
    cell = gs[0, 0].get_position(fig)
    fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() * (bbox.width * fig.get_figwidth())
                        / (cell.height * fig.get_figheight()))
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f'Figure written to {out_file}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, required=True, help='Output directory of the cache and the figure')
    parser.add_argument('--punch_tB', type=str, default=None, help='Glob of the prepared PUNCH tB files')
    parser.add_argument('--punch_pB', type=str, default=None, help='Glob of the prepared PUNCH pB files')
    parser.add_argument('--ccor_tB', type=str, default=None, help='Glob of the prepared CCOR tB files')
    parser.add_argument('--cor2_tB', type=str, default=None, help='Glob of the prepared COR2 tB files')
    parser.add_argument('--cor2_pB', type=str, default=None, help='Glob of the prepared COR2 pB files')
    parser.add_argument('--cor2_tB_correction', type=str, default=None,
                        help='Static tB correction image subtracted from COR2 in the training (npy)')
    parser.add_argument('--cor2_pB_correction', type=str, default=None,
                        help='Static pB correction image subtracted from COR2 in the training (npy)')
    parser.add_argument('--lasco_c2_tB', help='Glob of the prepared LASCO C2 tB files')
    parser.add_argument('--lasco_c2_correction', help='Static tB correction image for LASCO C2 (npy)')
    parser.add_argument('--lasco_c3_tB', help='Glob of the prepared LASCO C3 tB files')
    parser.add_argument('--lasco_c3_correction', help='Static tB correction image for LASCO C3 (npy)')
    parser.add_argument('--time', type=normalize_datetime, required=True,
                        help='The main figure shows the sampled time closest to this')
    parser.add_argument('--time_range', type=normalize_datetime, nargs=2, required=True, metavar=('START', 'END'),
                        help='Period of the frames of the radial profiles')
    parser.add_argument('--n_frames', type=int, default=20, help='Number of sampled times')
    parser.add_argument('--overwrite', action='store_true', help='Render again although a cache exists')
    args = parser.parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    inputs = {'punch': (args.punch_tB, args.punch_pB, None, None), 'ccor': (args.ccor_tB, None, None, None),
              'cor2': (args.cor2_tB, args.cor2_pB, args.cor2_tB_correction, args.cor2_pB_correction),
              'lasco_c2': (args.lasco_c2_tB, None, args.lasco_c2_correction, None),
              'lasco_c3': (args.lasco_c3_tB, None, args.lasco_c3_correction, None)}
    times = sample_times(args.time_range, args.n_frames)
    os.makedirs(os.path.join(args.out_path, 'cache'), exist_ok=True)
    os.makedirs(os.path.join(args.out_path, 'frames'), exist_ok=True)
    loader, results = None, {}
    for name, (tB_pattern, pB_pattern, tB_correction, pB_correction) in inputs.items():
        if tB_pattern is None:
            continue
        frames = None
        results[name] = []
        for k, time in enumerate(times):
            cache = os.path.join(args.out_path, 'cache', f'{name}_{k:03d}.npz')
            outdated = False
            if not args.overwrite and os.path.exists(cache):
                with np.load(cache) as data:
                    outdated = 'cache_version' not in data or int(data['cache_version']) != CACHE_VERSION
                if outdated:
                    print(f'Cache {cache} is from another version of the script; rendering again', flush=True)
            if args.overwrite or not os.path.exists(cache) or outdated:
                if loader is None:
                    from sunerf.evaluation.loader import ThomsonSuNeRFLoader
                    loader = ThomsonSuNeRFLoader(args.sunerf_path)
                if frames is None:
                    frames = list_frames(tB_pattern, pB_pattern, tB_correction, pB_correction)
                np.savez(cache, **evaluate_frame(loader, frames, INSTRUMENTS[name][1], time))
            with np.load(cache) as data:
                results[name].append({key: data[key] if data[key].ndim else data[key].item() for key in data.files})
    if not results:
        raise ValueError('Provide the tB files of at least one instrument')

    # every figure shows the profiles of all sampled frames; the maps change with the frame
    profiles = {name: average_profiles(frames) for name, frames in results.items()}
    for k, time in enumerate(times):
        plot({name: frames[k] for name, frames in results.items()}, profiles,
             os.path.join(args.out_path, 'frames', f'polarization_{k:03d}.jpg'))
    main_index = int(np.argmin([abs((time - args.time).total_seconds()) for time in times]))
    plot({name: frames[main_index] for name, frames in results.items()}, profiles,
         os.path.join(args.out_path, 'polarization.jpg'))


if __name__ == '__main__':
    main()
