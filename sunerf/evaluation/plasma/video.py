"""Generic video sweep of an EUV plasma reconstruction.

Counterpart of ``sunerf.evaluation.cme.video`` for the plasma model: the camera
moves along waypoints in position and time (or follows a real spacecraft) and
every frame shows a configurable row of panels: rendered EUV channels of the
training instruments (``aia193``, ``euvi284``, ``eui174``), the column electron
density, the emission-weighted temperature, the line-of-sight hydrogen column of
the cool absorber (models with ``model.cool_absorber``), the radially integrated
emission measure of the visible disk in temperature bins, the emission-weighted
distance from the observer and a 3D sketch of the observer geometry.

The emission measure panels are a separate query of the model: every disk pixel
is traced to its footpoint on the solar surface, a radial column above the
footpoint is sampled, and n_e^2 dr is integrated per temperature bin; off-limb
pixels are NaN (transparent).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization import AsinhStretch, ImageNormalize
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
from sunpy.coordinates import frames
from sunerf.data.horizons import get_horizons_coord
from sunpy.map import all_coordinates_from_map
# Registers the instrument colormaps (sdoaia193, euvi284, ...) with Matplotlib.
from sunpy.visualization import colormaps as _sunpy_colormaps  # noqa: F401
from tqdm import tqdm

from sunerf.configuration import canonical_channel_id
from sunerf.evaluation.loader import PlasmaSuNeRFLoader, _get_scale_from_occ_max
from sunerf.evaluation.plasma.radial import integrate_columns, radial_columns, rendered_shell

# --observer keywords: JPL Horizons targets
OBSERVERS = {'earth': '399', 'stereo_a': 'STEREO-A', 'stereo_b': 'STEREO-B', 'psp': 'Parker Solar Probe',
             'solo': 'Solar Orbiter'}
# training instruments: the spacecraft that carries them (SDO is in geosynchronous orbit, Earth within 0.0003 au)
INSTRUMENT_OBSERVERS = {'AIA': 'earth', 'EUVI-A': 'stereo_a', 'EUVI-B': 'stereo_b', 'EUI': 'solo'}
OBSERVER_LABELS = {'earth': 'Earth', 'stereo_a': 'STEREO-A', 'stereo_b': 'STEREO-B', 'psp': 'PSP',
                   'solo': 'Solar Orbiter'}
OBSERVER_COLORS = {'earth': 'dodgerblue', 'stereo_a': 'darkorange', 'stereo_b': 'seagreen', 'psp': 'purple',
                   'solo': 'crimson'}
# plasma panels: title, colormap, scale
PLASMA_PANELS = {'density': (r'Column $N_e$ [cm$^{-2}$]', 'viridis', 'log'),
                 'temperature': (r'Mean log$_{10}$(T / K)', 'plasma', 'linear'),
                 'cool_column': (r'Cool absorber column $N_H$ [cm$^{-2}$]', 'bone', 'log'),
                 'distance': (r'Distance from observer $- d_\odot$ [R$_\odot$]', 'berlin', 'symmetric'),
                 'observer': ('Observer', None, None)}
# emission-measure panels: colormap; the title carries the temperature bin
EM_CMAP = 'magma'
# default half-width of the field of view [projected R_sun] without --hpc_frame
DEFAULT_FOV = 1.5
# percentiles of the first frame that fix the color scales of the video; EUV channels are shown with an
# asinh stretch from zero to the upper percentile
CHANNEL_PERCENTILES = (1.0, 99.9)
CHANNEL_STRETCH = AsinhStretch(0.001)
DENSITY_PERCENTILES = (1.0, 99.9)
EM_PERCENTILES = (1.0, 99.9)
TEMPERATURE_PERCENTILES = (1.0, 99.0)
# temperature and distance are emission weighted and only meaningful where plasma emits: pixels below this
# fraction of the 99th percentile of the emission measure are transparent
EMISSION_MEASURE_FLOOR = 1.0e-3
# renderer outputs needed for the panels
MODEL_OUTPUTS = ['image', 'mean_T', 'column_electron_density_cm2', 'emission_measure_cm5', 'weights', 'z_vals']
# line-of-sight hydrogen column of the cool absorber; only rendered for the cool_column panel (models with
# model.cool_absorber)
COOL_COLUMN_OUTPUT = 'cool_hydrogen_column_cm2'


def parse_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def build_parser():
    parser = argparse.ArgumentParser(
        description='Create a generic EUV plasma video sweep.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to the plasma save state')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output directory of the frames; default: <run directory>/video')
    parser.add_argument('--lon', type=float, action='append', default=None,
                        help='Longitude target in degrees; the first entry is the start value, later entries are '
                             'interpolation targets')
    parser.add_argument('--lat', type=float, action='append', default=None,
                        help='Latitude target in degrees; the first entry is the start value, later entries are '
                             'interpolation targets')
    parser.add_argument('--lon_frame', type=str, choices=['carrington', 'hci'], default='carrington',
                        help='Longitude frame used by --lon and the observer panel')
    parser.add_argument('--time', type=parse_time, action='append', default=None,
                        help='Time target in ISO format; the first entry is the start value, later entries are '
                             'interpolation targets; default: the time range of the training data')
    parser.add_argument('--steps', type=int, action='append', default=None,
                        help='Frames per waypoint; first entry should normally be 1, later entries interpolate '
                             'from the previous waypoint to the new target')
    parser.add_argument('--distance_au', type=float, default=1.0, help='Observer distance in AU')
    parser.add_argument('--observer', type=str, action='append', default=None,
                        help='Real observer per waypoint (' + ', '.join(sorted(OBSERVERS)) + ', or a training '
                             'instrument key such as AIA or EUI): its ephemeris position (JPL Horizons) at the '
                             'time of every frame replaces --lat, --lon and --distance_au of the waypoint; the '
                             'frames between two different observers move from one to the other. none keeps the '
                             'manual position of the waypoint (HCI).')
    parser.add_argument('--fov', type=float, default=None,
                        help=f'Half-width of the rendered field of view in projected solar radii (default '
                             f'{DEFAULT_FOV}); excludes --hpc_frame')
    parser.add_argument('--resolution', type=int, default=256,
                        help='Square output resolution in pixels; with --subframe or --hpc_frame, the pixels along '
                             'its longer side')
    frame_group = parser.add_mutually_exclusive_group()
    frame_group.add_argument('--subframe', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                             default=None,
                             help='Render only this helioprojective sub-region, in projected solar radii from Sun '
                                  'center (e.g. -0.3 0.3 0 0.6); replaces --fov for the image extent (--fov still '
                                  'sets the distance color range)')
    frame_group.add_argument('--hpc_frame', type=float, nargs=4, metavar=('TXMIN', 'TXMAX', 'TYMIN', 'TYMAX'),
                             default=None,
                             help='Render this fixed helioprojective region in arcsec (e.g. -1200 1200 -1200 1200), '
                                  'as an instrument with a fixed field of view: the solar disk grows and shrinks '
                                  'with the observer distance and the axes are in arcsec; excludes --fov (the '
                                  'distance color range is the half-width of the first frame in projected solar '
                                  'radii)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Rays per rendering batch')
    parser.add_argument('--dpi', type=int, default=200, help='Saved frame DPI')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames even if output files exist')
    parser.add_argument('--panels', nargs='+', default=['aia193', 'density', 'em', 'observer'],
                        help='Panels of every frame, in this order. EUV channels are <instrument><wavelength> '
                             '(aia193, euvi284, eui174) or <instrument key>:<channel id> (AIA:A193). em expands '
                             'to --em_bins panels over --em_log_T_range; em:5.8-6.3 is one explicit bin in '
                             'log10 K (radially integrated above the visible disk). Plasma panels: '
                             + ', '.join(sorted(PLASMA_PANELS)))
    parser.add_argument('--em_bins', type=int, default=3, help='Number of temperature bins of the em panel')
    parser.add_argument('--em_radius_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Radial integration range of the em panels in solar radii; default: the rendered '
                             'shell (sampling.min_distance to max_distance)')
    parser.add_argument('--em_n_radius', type=int, default=256, help='Radial samples of the em integration')
    parser.add_argument('--em_log_T_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='log10 K range split into --em_bins; default: the diagnostic temperature grid')
    parser.add_argument('--channel_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of every EUV channel color scale (asinh stretch); default: 0 to the 99.9th percentile of the first frame')
    parser.add_argument('--density_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the column density color scale [cm^-2]; default: percentiles of '
                             'the first frame')
    parser.add_argument('--em_range', type=float, nargs=2, metavar=('MIN', 'MAX'), default=None,
                        help='Fixed range of the emission measure color scales [cm^-5]; default: percentiles of '
                             'the first frame, per bin')
    parser.add_argument('--fps', type=float, default=10.0, help='Frame rate of the encoded video')
    parser.add_argument('--frames_only', action='store_true', help='Write the frames without encoding a video')
    return parser


################################################################################
# Panel specification
################################################################################

def _normalize_key(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def resolve_instrument(loader, token):
    """Instrument key of the artifact matching ``token`` (case and punctuation insensitive)."""
    normalized = _normalize_key(token)
    exact = [key for key in loader.instrument_keys if _normalize_key(key) == normalized]
    if exact:
        return exact[0]
    # 'euvi' selects EUVI-A if it is the only EUVI instrument
    prefix = [key for key in loader.instrument_keys if _normalize_key(key).startswith(normalized)]
    if len(prefix) == 1:
        return prefix[0]
    raise ValueError(f'Unknown instrument {token!r}; artifact instruments are: {loader.instrument_keys}')


def resolve_channel(loader, instrument_key, token):
    channels = loader.channel_metadata(instrument_key)
    exact = [c for c in channels if str(c['id']) == str(token)]
    matches = exact or [c for c in channels if canonical_channel_id(c['id']) == canonical_channel_id(token)]
    if len(matches) != 1:
        available = ', '.join(str(c['id']) for c in channels)
        raise ValueError(f'Channel {token!r} of {instrument_key} is {"ambiguous" if matches else "unknown"}; '
                         f'available: {available}')
    return matches[0]


def parse_panels(loader, panel_tokens, em_bins, em_log_T_range):
    """Panel dictionaries: {'type': 'channel', 'instrument', 'channel'}, {'type': 'em', 'log_T_range'} or
    {'type': <plasma panel>}."""
    if em_log_T_range is None:
        log_T = np.asarray(loader.log_T_range, dtype=float)
        em_log_T_range = (float(log_T.min()), float(log_T.max()))
    panels = []
    for token in panel_tokens:
        lowered = token.lower()
        if lowered in PLASMA_PANELS:
            panels.append({'type': lowered})
        elif lowered == 'em':
            edges = np.linspace(em_log_T_range[0], em_log_T_range[1], em_bins + 1)
            panels.extend({'type': 'em', 'log_T_range': (float(lo), float(hi))} for lo, hi in zip(edges[:-1], edges[1:]))
        elif lowered.startswith('em:'):
            match = re.fullmatch(r'em:([0-9.]+)-([0-9.]+)', lowered)
            if match is None:
                raise ValueError(f'Emission measure panel must be em or em:<min>-<max> in log10 K, got {token!r}')
            panels.append({'type': 'em', 'log_T_range': (float(match.group(1)), float(match.group(2)))})
        elif ':' in token:
            instrument, channel = token.split(':', 1)
            instrument_key = resolve_instrument(loader, instrument)
            panels.append({'type': 'channel', 'instrument': instrument_key,
                           'channel': resolve_channel(loader, instrument_key, channel)})
        else:
            match = re.fullmatch(r'([A-Za-z][A-Za-z_-]*?)[-_]?([0-9]+(?:\.[0-9]+)?)', token)
            if match is None:
                raise ValueError(f'Unknown panel {token!r}')
            instrument_key = resolve_instrument(loader, match.group(1))
            panels.append({'type': 'channel', 'instrument': instrument_key,
                           'channel': resolve_channel(loader, instrument_key, match.group(2))})
    return panels


def channel_title(panel):
    channel = panel['channel']
    wavelength = channel.get('wavelength_angstrom')
    label = f'{wavelength:g} Å' if wavelength is not None else str(channel['id'])
    unit = channel.get('measurement_unit')
    return f'{panel["instrument"]} {label}' + (f' [{unit}]' if unit else '')


def channel_cmap(channel):
    try:
        return plt.get_cmap(channel.get('cmap', 'gray')).copy()
    except ValueError:
        return plt.get_cmap('gray').copy()


################################################################################
# Camera path (as in sunerf.evaluation.cme.video)
################################################################################

def resolve_time_range(loader):
    times = np.concatenate([np.asarray(loader.times(k)) for k in loader.ds_keys])
    return pd.Timestamp(times.min()).to_pydatetime(), pd.Timestamp(times.max()).to_pydatetime()


def resolve_observer_keyword(loader, value):
    if value is None or value == 'none':
        return None
    if value in OBSERVERS:
        return value
    instrument_key = resolve_instrument(loader, value)
    if instrument_key not in INSTRUMENT_OBSERVERS:
        raise ValueError(f'No spacecraft known for instrument {instrument_key!r}; use one of {sorted(OBSERVERS)}')
    return INSTRUMENT_OBSERVERS[instrument_key]


def resolve_waypoints(args, loader, default_start_time, default_end_time):
    lons = args.lon if args.lon is not None else [0.0]
    lats = args.lat if args.lat is not None else [0.0]
    times = args.time if args.time is not None else [default_start_time, default_end_time]
    steps_list = args.steps if args.steps is not None else [1, 80]
    observers = [resolve_observer_keyword(loader, o) for o in args.observer] if args.observer is not None else [None]

    n_waypoints = max(len(lons), len(lats), len(times), len(steps_list), len(observers))

    def expand(values, name):
        if len(values) == 1:
            return values * n_waypoints
        if len(values) != n_waypoints:
            raise ValueError(f'{name} must be provided once or once per waypoint')
        return values

    lons = expand(lons, '--lon')
    lats = expand(lats, '--lat')
    times = expand(times, '--time')
    steps_list = expand(steps_list, '--steps')
    observers = expand(observers, '--observer')

    waypoints = []
    for lon, lat, time, steps, observer in zip(lons, lats, times, steps_list, observers):
        if steps < 1:
            raise ValueError('--steps values must be at least 1')
        waypoints.append({'lon': lon, 'lat': lat, 'time': time, 'steps': steps, 'observer': observer})
    waypoints[0]['steps'] = 1
    return waypoints


def build_points(args, waypoints):
    """Frames: (distance, lat, lon, time, (observer of the segment start, observer of its end, fraction))."""
    first = waypoints[0]
    points = [(args.distance_au * u.AU, first['lat'] * u.deg, first['lon'] * u.deg, first['time'],
               (first['observer'], first['observer'], 1.0))]
    for previous, current in zip(waypoints[:-1], waypoints[1:]):
        fractions = np.linspace(0.0, 1.0, current['steps'])
        segment = list(zip(
            np.full(current['steps'], args.distance_au) * u.AU,
            np.linspace(previous['lat'], current['lat'], current['steps']) * u.deg,
            np.linspace(previous['lon'], current['lon'], current['steps']) * u.deg,
            pd.date_range(start=previous['time'], end=current['time'], periods=current['steps']).to_pydatetime(),
            [(previous['observer'], current['observer'], f) for f in fractions],
        ))
        points.extend(segment[1:])
    return points


def ephemeris_positions(observer, times):
    """HCI latitude, longitude [deg] and distance [R_sun] of a real observer at ``times`` (one Horizons query)."""
    coords = get_horizons_coord(OBSERVERS[observer], times)
    hci = [coords[i].transform_to(frames.HeliocentricInertial(obstime=times[i])) for i in range(len(times))]
    return np.array([[c.lat.to_value(u.deg), c.lon.to_value(u.deg), c.distance.to_value(u.R_sun)] for c in hci])


def blend_positions(start, end, fraction):
    """Position between two observers: linear in latitude and distance, shortest arc in longitude."""
    delta_lon = (end[1] - start[1] + 180.0) % 360.0 - 180.0
    return np.array([start[0] + fraction * (end[0] - start[0]), start[1] + fraction * delta_lon,
                     start[2] + fraction * (end[2] - start[2])])


def get_input_observer_coord(lat, lon, distance, obstime, lon_frame):
    if lon_frame == 'hci':
        return SkyCoord(lat=lat, lon=lon, distance=distance, frame=frames.HeliocentricInertial, obstime=obstime)
    return SkyCoord(lat=lat, lon=lon, radius=distance, frame=frames.HeliographicCarrington, obstime=obstime,
                    observer='self')


def to_plot_frame(coord, obstime, lon_frame):
    """(lat, lon, distance) of a coordinate in the display frame."""
    if lon_frame == 'hci':
        c = coord.transform_to(frames.HeliocentricInertial(obstime=obstime))
        return c.lat.to(u.deg), c.lon.to(u.deg), c.distance.to(u.AU)
    c = coord.transform_to(frames.HeliographicCarrington(observer='self', obstime=obstime))
    return c.lat.to(u.deg), c.lon.to(u.deg), c.radius.to(u.AU)


################################################################################
# Observer panel
################################################################################

def spherical_to_cartesian(radius, lat, lon):
    lat_rad = lat.to_value(u.rad)
    lon_rad = lon.to_value(u.rad)
    return (radius * np.cos(lat_rad) * np.cos(lon_rad), radius * np.cos(lat_rad) * np.sin(lon_rad),
            radius * np.sin(lat_rad))


def display_radius(distance):
    return 1.08 + 0.08 * np.log10(max(distance.to_value(u.AU), 1e-3) * 10)


def add_front_text(ax, x, y, z, text, **kwargs):
    label = ax.text(x, y, z, text, zorder=1000, **kwargs)
    label.set_path_effects([patheffects.withStroke(linewidth=2.5, foreground='white', alpha=0.85)])
    return label


def plot_observer_geometry(ax, observer_position, time, training_positions):
    """Sun, virtual observer (red) and the training spacecraft in the display frame."""
    ax.computed_zorder = False
    sphere_u = np.linspace(0, 2 * np.pi, 120)
    sphere_v = np.linspace(0, np.pi, 60)
    x = np.outer(np.cos(sphere_u), np.sin(sphere_v))
    y = np.outer(np.sin(sphere_u), np.sin(sphere_v))
    z = np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
    ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color='0.75', linewidth=0.45, alpha=0.7)
    ax.scatter(0, 0, 0, color='#ffd54a', s=240, edgecolors='#ffb300', linewidths=0.9, depthshade=False)

    labels = []
    for observer, (lat, lon, distance) in training_positions.items():
        px, py, pz = spherical_to_cartesian(display_radius(distance), lat, lon)
        color = OBSERVER_COLORS[observer]
        ax.scatter(px, py, pz, color=color, s=220, alpha=0.25, depthshade=False)
        ax.scatter(px, py, pz, color=color, s=50, edgecolors='white', linewidths=0.6, depthshade=False)
        labels.append((px, py, pz, ' ' + OBSERVER_LABELS[observer], {'color': color, 'fontsize': 9}))

    lat, lon, distance = observer_position
    ox, oy, oz = spherical_to_cartesian(display_radius(distance), lat, lon)
    ax.plot([0, ox], [0, oy], [0, oz], color='red', linewidth=1.6, alpha=0.45, linestyle=':')
    ax.scatter(ox, oy, oz, color='#ffb3b3', s=220, alpha=0.28, depthshade=False)
    ax.scatter(ox, oy, oz, color='red', s=55, edgecolors='white', linewidths=0.6, depthshade=False)
    labels.append((ox, oy, oz - 0.08, 'SuNeRF', {'color': 'red', 'fontsize': 9, 'ha': 'center'}))
    for lx, ly, lz, text, kwargs in labels:
        add_front_text(ax, lx, ly, lz, text, **kwargs)

    ax.set_title(time.isoformat(' ', timespec='minutes'))
    ax.set_box_aspect((1, 1, 1), zoom=1.45)
    limit = 1.06
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()


################################################################################
# Image products
################################################################################

def frame_range(data, percentiles):
    """Percentiles of the finite positive values of an image."""
    values = data[np.isfinite(data) & (data > 0)]
    if values.size == 0:
        return 1.0, 10.0
    low, high = np.percentile(values, percentiles)
    return (low, high) if high > low else (values.min(), values.max() * 10)


def channel_norm(vmin, vmax):
    """Asinh-stretched color scale of an EUV channel."""
    return ImageNormalize(vmin=float(vmin), vmax=float(vmax), stretch=CHANNEL_STRETCH)


def decade_ticks(vmin, vmax):
    """Powers of ten inside [vmin, vmax], down to the linear part of the asinh stretch (vmax times the
    CHANNEL_STRETCH parameter); 2 and 5 multiples are added when fewer than three decades fit."""
    low = max(vmin, vmax * CHANNEL_STRETCH.a)
    decades = np.arange(np.ceil(np.log10(low)), np.floor(np.log10(vmax)) + 1)
    ticks = 10.0 ** decades
    if len(ticks) < 3:
        ticks = np.concatenate([10.0 ** (decades[0] - 1 + k) * m for k in range(len(decades) + 2) for m in (1, 2, 5)])
        ticks = ticks[(ticks >= low) & (ticks <= vmax)]
    return ticks


def set_channel_colorbar_ticks(colorbar, norm):
    """Logarithmically spaced 10^x ticks on an asinh-stretched channel colorbar."""
    ticks = decade_ticks(float(norm.vmin), float(norm.vmax))
    colorbar.ax.xaxis.set_major_locator(FixedLocator(ticks))
    colorbar.ax.xaxis.set_minor_locator(FixedLocator([]))
    colorbar.ax.xaxis.set_major_formatter(LogFormatterMathtext())


def emission_mask(emission_measure):
    """Pixels where plasma emits: emission measure above a fraction of its 99th percentile."""
    valid = np.isfinite(emission_measure) & (emission_measure > 0)
    if not valid.any():
        return valid
    floor = EMISSION_MEASURE_FLOOR * np.percentile(emission_measure[valid], 99.0)
    return valid & (emission_measure >= floor)


def disk_footpoints(ref_map):
    """Carrington latitude and longitude [rad] of the solar-surface point under every pixel of ``ref_map``;
    NaN off the limb."""
    coords = all_coordinates_from_map(ref_map)
    surface = coords.transform_to(
        frames.HeliographicCarrington(observer=ref_map.observer_coordinate, obstime=ref_map.date)
    )
    with np.errstate(invalid='ignore'):
        lat = surface.lat.to_value(u.rad)
        lon = surface.lon.to_value(u.rad)
    off_limb = ~(np.isfinite(lat) & np.isfinite(lon))
    lat = np.where(off_limb, np.nan, lat)
    lon = np.where(off_limb, np.nan, lon)
    return lat, lon


def radial_em_maps(loader, ref_map, time, bins, radius_range, n_radius, batch_size):
    """Radially integrated emission measure [cm^-5] per temperature bin on the visible disk (NaN off limb)."""
    lat, lon = disk_footpoints(ref_map)
    radii = np.linspace(radius_range[0], radius_range[1], n_radius)
    points = radial_columns(lat, lon, radii)
    maps, _ = integrate_columns(loader, points, radii, time, bins, batch_size)
    return maps


def image_geometry(args, distance):
    """Resolution (ny, nx), WCS scale, HPC image center (Tx, Ty) and the image center in plot units (projected
    solar radii, or arcsec with --hpc_frame). Without --subframe or --hpc_frame the image is the square --fov around
    Sun center; with one of them, the image covers the region with square pixels and its center as the WCS
    reference coordinate."""
    if args.subframe is None and args.hpc_frame is None:
        resolution = (args.resolution, args.resolution) * u.pix
        scale = _get_scale_from_occ_max(args.fov * u.R_sun, distance, resolution)
        return resolution, scale, 0 * u.arcsec, 0 * u.arcsec, (0.0, 0.0)
    region = args.subframe if args.hpc_frame is None else args.hpc_frame
    x_min, x_max, y_min, y_max = region
    width, height = x_max - x_min, y_max - y_min
    if width <= 0 or height <= 0:
        raise ValueError(f'The image region needs XMIN < XMAX and YMIN < YMAX, got {region}')
    pixel = max(width, height) / args.resolution
    nx, ny = max(1, round(width / pixel)), max(1, round(height / pixel))
    center = (0.5 * (x_min + x_max), 0.5 * (y_min + y_max))
    if args.hpc_frame is not None:
        # fixed angular extent, independent of the observer distance
        scale = u.Quantity([width / nx, height / ny]) * u.arcsec / u.pix
        return (ny, nx) * u.pix, scale, center[0] * u.arcsec, center[1] * u.arcsec, center
    d = distance.to_value(u.R_sun)
    # projected solar radii are tangent-plane offsets times the observer distance (as in the plot extent)
    scale = u.Quantity([(width / nx / d * u.rad).to(u.arcsec), (height / ny / d * u.rad).to(u.arcsec)]) / u.pix
    hpc_x = (np.arctan(center[0] / d) * u.rad).to(u.arcsec)
    hpc_y = (np.arctan(center[1] / d) * u.rad).to(u.arcsec)
    return (ny, nx) * u.pix, scale, hpc_x, hpc_y, center


def render(loader, hci_coord, time, distance, instruments, resolution, scale, hpc_x, hpc_y, batch_size,
           cool_column=False):
    """Rendered maps per instrument and the plasma diagnostics (from the first instrument); with
    ``cool_column``, the line-of-sight hydrogen column of the cool absorber."""
    renders = {}
    for i, instrument_key in enumerate(instruments):
        model_outputs = MODEL_OUTPUTS + [COOL_COLUMN_OUTPUT] if cool_column and i == 0 else MODEL_OUTPUTS
        renders[instrument_key] = loader.load_image(
            hci_coord.lat, hci_coord.lon, time, distance=distance, hpc_lon=hpc_x, hpc_lat=hpc_y,
            resolution=resolution, scale=scale, instrument_key=instrument_key, batch_size=batch_size,
            progress=False, model_outputs=model_outputs,
        )
    first = renders[instruments[0]]
    # emission-weighted LOS distance from the observer [R_sun], relative to the Sun center distance
    los_distance = (first['weights'] * first['z_vals']).sum(-1) * float(loader.Rs_per_ds)
    plasma = {
        'density': np.squeeze(first['column_electron_density_cm2']),
        'temperature': np.squeeze(first['mean_T']),
        'emission_measure': np.squeeze(first['emission_measure_cm5']),
        'distance': los_distance - distance.to_value(u.R_sun),
    }
    if cool_column:
        plasma['cool_column'] = np.squeeze(first[COOL_COLUMN_OUTPUT])
    return renders, plasma


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.hpc_frame is not None and args.fov is not None:
        parser.error('--fov and --hpc_frame are exclusive')
    if args.hpc_frame is None and args.fov is None:
        args.fov = DEFAULT_FOV
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'video')
    os.makedirs(args.out_path, exist_ok=True)

    loader = PlasmaSuNeRFLoader(args.sunerf_path)
    if args.em_radius_range is None:
        args.em_radius_range = rendered_shell(loader)
    panels = parse_panels(loader, args.panels, args.em_bins, args.em_log_T_range)
    instruments = list(dict.fromkeys(p['instrument'] for p in panels if p['type'] == 'channel'))
    if not instruments:
        # the plasma diagnostics are instrument independent; render the first instrument for them
        instruments = [loader.instrument_keys[0]]
    em_bins = [p['log_T_range'] for p in panels if p['type'] == 'em']
    training_observers = [INSTRUMENT_OBSERVERS[k] for k in loader.instrument_keys if k in INSTRUMENT_OBSERVERS]

    start_time, end_time = resolve_time_range(loader)
    waypoints = resolve_waypoints(args, loader, start_time, end_time)
    points = build_points(args, waypoints)
    frame_times = [point[3] for point in points]

    # ephemeris positions of the real observers at all frame times (one query per observer)
    camera_observers = {o for point in points for o in point[4][:2] if o is not None}
    if camera_observers:
        args.lon_frame = 'hci'
    ephemeris = {o: ephemeris_positions(o, frame_times) for o in camera_observers}
    show_observer = any(p['type'] == 'observer' for p in panels)
    cool_column = any(p['type'] == 'cool_column' for p in panels)
    training_coords = {o: get_horizons_coord(OBSERVERS[o], frame_times) for o in training_observers} \
        if show_observer else {}

    # one color scale per panel for the whole video: given ranges, or the percentiles of the first frame
    norms = {}
    fixed = {'channel': channel_norm(*args.channel_range) if args.channel_range else None,
             'density': LogNorm(*args.density_range) if args.density_range else None,
             'em': LogNorm(*args.em_range) if args.em_range else None}

    frame_paths = []
    for i, (distance, lat, lon_value, time, segment) in tqdm(enumerate(points), total=len(points)):
        frame_path = os.path.join(args.out_path, f'euv_frame{i:03d}.jpg')
        frame_paths.append(frame_path)
        if os.path.exists(frame_path) and not args.overwrite:
            print(f'Skipping existing frame: {frame_path}')
            continue

        start_observer, end_observer, fraction = segment
        if start_observer is None and end_observer is None:
            input_coord = get_input_observer_coord(lat, lon_value, distance, time, args.lon_frame)
        else:
            manual = np.array([lat.to_value(u.deg), lon_value.to_value(u.deg), distance.to_value(u.R_sun)])
            start = manual if start_observer is None else ephemeris[start_observer][i]
            end = manual if end_observer is None else ephemeris[end_observer][i]
            position = end if start_observer == end_observer else blend_positions(start, end, fraction)
            distance = position[2] * u.R_sun
            input_coord = get_input_observer_coord(position[0] * u.deg, position[1] * u.deg, distance, time, 'hci')
        hci_coord = input_coord if args.lon_frame == 'hci' else \
            input_coord.transform_to(frames.HeliocentricInertial(obstime=time))
        plot_position = to_plot_frame(input_coord, time, args.lon_frame)

        resolution, scale, hpc_x, hpc_y, image_center = image_geometry(args, distance)
        renders, plasma = render(loader, hci_coord, time, distance, instruments, resolution, scale, hpc_x, hpc_y,
                                 args.batch_size, cool_column=cool_column)
        emitting = emission_mask(plasma['emission_measure'])
        ref_map = next(iter(renders[instruments[0]]['maps'].values()))
        if em_bins:
            # separate query: radial columns above the footpoints of the disk pixels
            em_maps = dict(zip(em_bins, radial_em_maps(loader, ref_map, time, em_bins, args.em_radius_range,
                                                      args.em_n_radius, args.batch_size)))

        # image plane: tangent-plane pixel offsets, shifted by the image center (Sun center, or the region center);
        # in projected solar radii (offsets times the observer distance), or in arcsec with --hpc_frame
        if args.hpc_frame is None:
            units_per_rad, axis_unit = float(ref_map.dsun.to_value(u.R_sun)), r'R$_\odot$'
        else:
            units_per_rad, axis_unit = (1 * u.rad).to_value(u.arcsec), 'arcsec'
        units_per_pixel = [float(s.to_value(u.rad / u.pix)) * units_per_rad for s in ref_map.scale]
        center = [float(ref_map.reference_pixel.x.to_value(u.pix)), float(ref_map.reference_pixel.y.to_value(u.pix))]
        ny, nx = ref_map.data.shape
        extent = [image_center[0] + (-0.5 - center[0]) * units_per_pixel[0],
                  image_center[0] + (nx - 0.5 - center[0]) * units_per_pixel[0],
                  image_center[1] + (-0.5 - center[1]) * units_per_pixel[1],
                  image_center[1] + (ny - 0.5 - center[1]) * units_per_pixel[1]]

        print(f'[{time.isoformat(sep=" ", timespec="minutes")}] {args.lon_frame.upper()} | '
              f'lat={plot_position[0].to_value(u.deg):7.3f} deg, lon={plot_position[1].to_value(u.deg):8.3f} deg, '
              f'r={plot_position[2].to_value(u.AU):6.3f} AU')

        n_cols = len(panels)
        fig = plt.figure(figsize=(3.4 * n_cols, 4.6), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.0)

        first_image_ax = None
        for column, panel in enumerate(panels):
            if panel['type'] == 'observer':
                observer_ax = fig.add_subplot(1, n_cols, column + 1, projection='3d')
                training_positions = {o: to_plot_frame(coords[i], time, args.lon_frame)
                                      for o, coords in training_coords.items()}
                plot_observer_geometry(observer_ax, plot_position, time, training_positions)
                continue

            ax = fig.add_subplot(1, n_cols, column + 1, sharex=first_image_ax, sharey=first_image_ax)
            if panel['type'] == 'channel':
                key = ('channel', panel['instrument'], str(panel['channel']['id']))
                data = renders[panel['instrument']]['maps'][panel['channel']['id']].data
                title, cmap = channel_title(panel), channel_cmap(panel['channel'])
                if key not in norms:
                    norms[key] = fixed['channel'] or channel_norm(0.0, frame_range(data, CHANNEL_PERCENTILES)[1])
            elif panel['type'] == 'em':
                key = ('em', panel['log_T_range'])
                data = em_maps[panel['log_T_range']]
                lo, hi = panel['log_T_range']
                title = rf'Radial EM, log$_{{10}}$T in [{lo:.2f}, {hi:.2f}) [cm$^{{-5}}$]'
                cmap = plt.get_cmap(EM_CMAP).copy()
                if key not in norms:
                    norms[key] = fixed['em'] or LogNorm(*frame_range(data, EM_PERCENTILES))
            else:
                key = (panel['type'],)
                title, cmap, kind = PLASMA_PANELS[panel['type']]
                cmap = plt.get_cmap(cmap).copy()
                data = plasma[panel['type']].astype(float).copy()
                if panel['type'] == 'density':
                    if key not in norms:
                        norms[key] = fixed['density'] or LogNorm(*frame_range(data, DENSITY_PERCENTILES))
                elif panel['type'] == 'cool_column':
                    data[~(data > 0)] = np.nan
                    if key not in norms:
                        norms[key] = LogNorm(*frame_range(data, DENSITY_PERCENTILES))
                elif panel['type'] == 'temperature':
                    data[~emitting] = np.nan
                    if key not in norms:
                        values = data[np.isfinite(data)]
                        low, high = np.percentile(values, TEMPERATURE_PERCENTILES) if values.size else (5.5, 6.5)
                        norms[key] = Normalize(vmin=low, vmax=high)
                else:
                    data[~emitting] = np.nan
                    if key not in norms:
                        # --fov, or the half-width of the first --hpc_frame frame in projected solar radii
                        half_width = args.fov if args.fov is not None else distance.to_value(u.R_sun) * np.tan(
                            (np.abs(extent).max() * u.arcsec).to_value(u.rad))
                        norms[key] = Normalize(vmin=-half_width, vmax=half_width)
            # NaN (outside the field, non-emitting) is transparent on the white background
            cmap.set_bad((0.0, 0.0, 0.0, 0.0))
            im = ax.imshow(data, cmap=cmap, origin='lower', extent=extent, norm=norms[key])
            ax.set_facecolor('white')
            ax.set_aspect('equal')
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            colorbar = fig.colorbar(im, ax=ax, location='top', aspect=30, pad=0.02)
            colorbar.set_label(title)
            if panel['type'] == 'channel':
                set_channel_colorbar_ticks(colorbar, norms[key])
            ax.set_xlabel(f'HPC-x [{axis_unit}]')
            if first_image_ax is None:
                first_image_ax = ax
                ax.set_ylabel(f'HPC-y [{axis_unit}]')
            else:
                ax.tick_params(labelleft=False)

        fig.suptitle(
            f'Lat. ({args.lon_frame.upper()}): {plot_position[0].to_value(u.deg):.1f} deg, '
            f'Lon. ({args.lon_frame.upper()}): {plot_position[1].to_value(u.deg):.1f} deg, '
            f'Time: {time.isoformat(" ", timespec="minutes")}',
            fontsize=16,
        )
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close('all')

    if not args.frames_only:
        from sunerf.evaluation.image_sequence_to_video import write_video

        video_path = args.out_path.rstrip('/') + '.mp4'
        write_video(frame_paths, video_path, args.fps)
        print(f'Wrote {len(frame_paths)} frames to {args.out_path} and the video to {video_path}')


if __name__ == '__main__':
    main()
