import argparse
import os
from time import perf_counter

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr, spearmanr
from sunpy.coordinates import frames
from sunerf.data.horizons import get_horizons_coord
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


INERTIAL_COORDS_KEY = 'inertial_coords_r_lat_lon'
POSITION_SUPPORT_SECONDS = 60.0
# color of the solar panels of the probe icons and colormap of the trajectory over time
PROBE_COLORS = {'PSP': '#38bdf8', 'SolO': '#f59e0b'}
PROBE_CMAPS = {'PSP': 'viridis', 'SolO': 'cool'}
# upper limit of the pB color scale [MSB]
PB_MAX = 1e-10
# density_source of the prepared PSP products (sunerf.data.prep.psp_insitu): label and colour
DENSITY_SOURCES = {1.0: ('QTN', 'black'), 2.0: ('SPC', 'tab:blue')}


def parse_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def build_parser():
    parser = argparse.ArgumentParser(
        description='Render a solar-north PSP in-situ overview animation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--insitu_path', type=str, required=True, help='Path to PSP in-situ npz file')
    parser.add_argument('--solo_path', type=str, default=None,
                        help='Path to Solar Orbiter in-situ npz file; adds its trajectory and time series')
    parser.add_argument('--out_path', type=str, required=True, help='Path to output frame directory')
    parser.add_argument('--n_frames', '--N', dest='n_frames', type=int, default=80, help='Number of frames to render')
    parser.add_argument('--time_start', type=parse_time, default=None, help='First rendered time')
    parser.add_argument('--time_end', type=parse_time, default=None, help='Last rendered time')
    parser.add_argument('--distance_rsun', type=float, default=215.0,
                        help='Observer distance in solar radii; must exceed --occ_max')
    parser.add_argument('--resolution', type=int, default=512, help='Square render resolution in pixels')
    parser.add_argument('--occ_min', type=float, default=2.5, help='Inner projected radius in solar radii')
    parser.add_argument('--occ_max', type=float, default=100.0,
                        help='Outer projected radius of the Earth and the solar-north view in solar radii')
    parser.add_argument('--max_plot_radius_rsun', type=float, default=100.0,
                        help='Trajectory and time series of every probe cover all samples at or below this '
                             'radius; the default is the radius of the reconstructed domain')
    parser.add_argument('--max_trajectory_points', type=int, default=128,
                        help='Maximum PSP trajectory samples plotted per frame')
    parser.add_argument('--reconstruction_points', type=int, default=128,
                        help='Number of equal-time PSP positions used for reconstructed in-situ curves')
    parser.add_argument('--check_horizons', action='store_true',
                        help='Compare sampled PSP npz positions against JPL Horizons and print full coordinate deltas')
    parser.add_argument('--horizons_check_points', type=int, default=12,
                        help='Number of equal-time PSP samples to compare when --check_horizons is used')
    parser.add_argument('--dpi', type=int, default=300, help='Saved frame DPI')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames that already exist')
    return parser


def load_insitu(path, name='PSP'):
    print(f'Loading {name} in-situ data: {path}', flush=True)
    data = np.load(path, allow_pickle=True)
    required = ['time_unix', 'density_cm3']
    missing = [key for key in required if key not in data]
    if INERTIAL_COORDS_KEY not in data:
        missing.append(INERTIAL_COORDS_KEY)
    if missing:
        raise KeyError(f'In-situ file is missing required keys: {", ".join(missing)}')

    time_unix = data['time_unix'].astype(np.float64)
    order = np.argsort(time_unix)
    times = pd.to_datetime(time_unix[order], unit='s', utc=True).tz_convert(None).to_pydatetime()
    position = cartesian_from_inertial_coords(data[INERTIAL_COORDS_KEY].astype(np.float64))[order]
    if 'position_source' in data or 'position_coordinate_system' in data:
        position_source = str(data['position_source']) if 'position_source' in data else 'unknown'
        coordinate_system = str(data['position_coordinate_system']) if 'position_coordinate_system' in data else 'unknown'
        print(f'{name} position metadata: source={position_source}, coordinate_system={coordinate_system}', flush=True)
    radius_check = np.linalg.norm(position, axis=-1)
    if 'radius_rsun' in data:
        radius_delta = radius_check - data['radius_rsun'].astype(np.float64)[order]
        print(
            f'{name} position sanity: |xyz| radius range=({np.nanmin(radius_check):.2f}, {np.nanmax(radius_check):.2f}) R_sun, '
            f'|xyz|-radius max abs delta={np.nanmax(np.abs(radius_delta)):.3e} R_sun',
            flush=True,
        )
    else:
        print(
            f'{name} position sanity: |xyz| radius range=({np.nanmin(radius_check):.2f}, {np.nanmax(radius_check):.2f}) R_sun',
            flush=True,
        )

    return {
        'name': name,
        'time_unix': time_unix[order],
        'times': np.array(times),
        'position_hci_rsun': position,
        'density_cm3': data['density_cm3'].astype(np.float64)[order],
        'radius_rsun': data['radius_rsun'].astype(np.float64)[order] if 'radius_rsun' in data else None,
        'density_source': (data['density_source'].astype(np.float64)[order] if 'density_source' in data
                           else np.zeros(len(order))),
    }


def hourly_average(time_unix, density):
    """Arithmetic mean of the density in full UTC hours, the averaging of the hourly Solar Orbiter product.

    Hours without a valid sample are kept as NaN at the centre of the hour, so that gaps stay open.
    """
    valid = np.isfinite(density) & (density > 0)
    time_unix, density = time_unix[valid], density[valid]
    if time_unix.size == 0:
        return np.array([]), np.array([])
    hour = np.floor(time_unix / 3600.0).astype(np.int64)
    hours = np.arange(hour.min(), hour.max() + 1)
    count = np.bincount(hour - hours[0], minlength=len(hours))
    filled = count > 0
    mean_time = hours * 3600.0 + 1800.0
    mean_density = np.full(len(hours), np.nan)
    mean_time[filled] = np.bincount(hour - hours[0], weights=time_unix, minlength=len(hours))[filled] / count[filled]
    mean_density[filled] = np.bincount(hour - hours[0], weights=density, minlength=len(hours))[filled] / count[filled]
    return mean_time, mean_density


def observed_series(insitu, available_idx):
    """Hourly averages of the measured density: all samples and every density source on its own."""
    time_unix, density = insitu['time_unix'][available_idx], insitu['density_cm3'][available_idx]
    source = insitu['density_source'][available_idx]
    series = {'all': hourly_average(time_unix, density), 'sources': []}
    present = [value for value in np.unique(source[np.isfinite(density)]) if value in DENSITY_SOURCES]
    if len(present) > 1:
        for value in present:
            label, color = DENSITY_SOURCES[value]
            series['sources'].append((label, color) + hourly_average(time_unix[source == value],
                                                                     density[source == value]))
    return series


def cartesian_from_inertial_coords(inertial_coords_r_lat_lon):
    coords = np.atleast_2d(inertial_coords_r_lat_lon).astype(np.float64)
    r = coords[:, 0]
    lat = coords[:, 1]
    lon = coords[:, 2]
    return np.stack(
        [
            r * np.cos(lat) * np.cos(lon),
            r * np.cos(lat) * np.sin(lon),
            r * np.sin(lat),
        ],
        axis=-1,
    )


def resolve_time_range(loader, time_start, time_end):
    observer_times = sorted({o['time'] for o in loader.observers})
    start_time = observer_times[0] if time_start is None else time_start
    end_time = observer_times[-1] if time_end is None else time_end
    if end_time < start_time:
        raise ValueError('--time_end must be after --time_start')
    return start_time, end_time


def nearest_available_position(time_unix, available_idx, time):
    current_unix = pd.Timestamp(time).timestamp()
    return int(np.nanargmin(np.abs(time_unix[available_idx] - current_unix)))


def available_insitu_indices(insitu, start_time, end_time, max_radius_rsun):
    start_unix = pd.Timestamp(start_time).timestamp()
    end_unix = pd.Timestamp(end_time).timestamp()
    # datetimes hold microseconds; the tolerance keeps samples whose time was rounded on the way
    mask = (insitu['time_unix'] >= start_unix - 1e-3) & (insitu['time_unix'] <= end_unix + 1e-3)
    if insitu['radius_rsun'] is not None:
        mask &= insitu['radius_rsun'] <= max_radius_rsun
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            f'No {insitu["name"]} samples are available between {start_time} and {end_time} '
            f'with radius <= {max_radius_rsun} R_sun'
        )
    return indices


def equal_time_samples(start_time, end_time, n_samples):
    if n_samples < 1:
        raise ValueError('n_samples must be at least 1')
    start_unix = pd.Timestamp(start_time).timestamp()
    end_unix = pd.Timestamp(end_time).timestamp()
    time_unix = np.linspace(start_unix, end_unix, n_samples)
    times = pd.to_datetime(time_unix, unit='s', utc=True).tz_convert(None).to_pydatetime()
    return np.array(times), time_unix


def spherical_from_hci(position_hci_rsun):
    position_hci_rsun = np.atleast_2d(position_hci_rsun).astype(np.float64)
    radius = np.linalg.norm(position_hci_rsun, axis=-1)
    lon = np.rad2deg(np.arctan2(position_hci_rsun[:, 1], position_hci_rsun[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(position_hci_rsun[:, 2] / np.clip(radius, 1e-12, None), -1.0, 1.0)))
    return radius, lon, lat


def angle_delta_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def query_psp_horizons_hci(time):
    try:
        horizons = get_horizons_coord('Parker Solar Probe', time)
    except Exception:
        horizons = get_horizons_coord('Solar Probe Plus', time)
    horizons_hci = horizons.transform_to(frames.HeliocentricInertial(obstime=time))
    return np.asarray(horizons_hci.cartesian.xyz.to_value(u.R_sun), dtype=np.float64).reshape(3)


def interpolate_positions(insitu, time_unix, label):
    """Probe position in HCI [R_sun] at arbitrary times from the positions stored with the in-situ samples.

    The prepared in-situ file holds the JPL Horizons position of every sample.
    A cubic spline through these positions (one sample per minute at most)
    follows the orbit also across gaps of several hours in the measurements.
    """
    time_unix = np.asarray(time_unix, dtype=np.float64)
    sample_time = insitu['time_unix']
    if time_unix.min() < sample_time[0] or time_unix.max() > sample_time[-1]:
        raise ValueError(f'The {label} times are outside the time range of the in-situ positions')
    # the orbit is smooth: one support point per minute, with the first and the last sample
    _, support = np.unique(np.floor(sample_time / POSITION_SUPPORT_SECONDS), return_index=True)
    support = np.unique(np.append(support, sample_time.size - 1))
    support = support[np.concatenate([[True], np.diff(sample_time[support]) > 0])]
    positions = CubicSpline(sample_time[support], insitu['position_hci_rsun'][support], axis=0)(time_unix)
    radius = np.linalg.norm(positions, axis=-1)
    gap = np.diff(sample_time)[np.clip(np.searchsorted(sample_time, time_unix) - 1, 0, sample_time.size - 2)]
    print(
        f'{insitu["name"]} positions for {len(time_unix)} {label} time(s) from the in-situ file: '
        f'radius range=({radius.min():.2f}, {radius.max():.2f}) R_sun, '
        f'largest gap between the bracketing samples={gap.max() / 60:.1f} min',
        flush=True,
    )
    return positions


def check_horizons_positions(insitu, available_idx, n_points):
    check_idx = sample_time_indices(insitu['time_unix'], available_idx, n_points)
    print(f'Checking {len(check_idx)} PSP npz position sample(s) against JPL Horizons...', flush=True)
    print(
        '  Columns: time | npz xyz | Horizons xyz | delta xyz=npz-Horizons | |dxyz| | '
        'npz r/lon/lat | Horizons r/lon/lat | delta r/lon/lat',
        flush=True,
    )
    cartesian_deltas = []
    radius_deltas = []
    lon_deltas = []
    lat_deltas = []
    for idx in check_idx:
        time = insitu['times'][idx]
        stored_position = np.asarray(insitu['position_hci_rsun'][idx], dtype=np.float64)
        horizons_position = query_psp_horizons_hci(time)
        delta_position = stored_position - horizons_position
        cartesian_delta = np.linalg.norm(delta_position)

        stored_radius, stored_lon, stored_lat = spherical_from_hci(stored_position)
        horizons_radius, horizons_lon, horizons_lat = spherical_from_hci(horizons_position)
        radius_delta = stored_radius[0] - horizons_radius[0]
        lon_delta = angle_delta_deg(stored_lon[0], horizons_lon[0])
        lat_delta = stored_lat[0] - horizons_lat[0]

        cartesian_deltas.append(cartesian_delta)
        radius_deltas.append(radius_delta)
        lon_deltas.append(lon_delta)
        lat_deltas.append(lat_delta)
        print(
            f'  {time.isoformat(" ", timespec="seconds")} | '
            f'npz xyz=({stored_position[0]: .6f}, {stored_position[1]: .6f}, {stored_position[2]: .6f}) R_sun | '
            f'Horizons xyz=({horizons_position[0]: .6f}, {horizons_position[1]: .6f}, {horizons_position[2]: .6f}) R_sun | '
            f'dxyz=({delta_position[0]: .6e}, {delta_position[1]: .6e}, {delta_position[2]: .6e}) R_sun | '
            f'|dxyz|={cartesian_delta:.6e} R_sun | '
            f'npz r/lon/lat=({stored_radius[0]: .6f}, {stored_lon[0]: .6f}, {stored_lat[0]: .6f}) | '
            f'Horizons r/lon/lat=({horizons_radius[0]: .6f}, {horizons_lon[0]: .6f}, {horizons_lat[0]: .6f}) | '
            f'dr/dlon/dlat=({radius_delta: .6e}, {lon_delta: .6e}, {lat_delta: .6e})',
            flush=True,
        )

    cartesian_deltas = np.asarray(cartesian_deltas, dtype=np.float64)
    radius_deltas = np.asarray(radius_deltas, dtype=np.float64)
    lon_deltas = np.asarray(lon_deltas, dtype=np.float64)
    lat_deltas = np.asarray(lat_deltas, dtype=np.float64)
    print(
        'PSP npz vs Horizons summary: '
        f'max |dxyz|={np.nanmax(cartesian_deltas):.6e} R_sun, '
        f'rms |dxyz|={np.sqrt(np.nanmean(cartesian_deltas ** 2)):.6e} R_sun, '
        f'max |dr|={np.nanmax(np.abs(radius_deltas)):.6e} R_sun, '
        f'max |dlon|={np.nanmax(np.abs(lon_deltas)):.6e} deg, '
        f'max |dlat|={np.nanmax(np.abs(lat_deltas)):.6e} deg',
        flush=True,
    )


def get_earth_observer_coord(time):
    earth = get_body_heliographic_stonyhurst('earth', time)
    return earth.transform_to(frames.HeliocentricInertial(obstime=time))


def stereo_a_observers(loader):
    """Observation times [unix s] and HCI unit vectors of STEREO-A from the observers of the save state."""
    observers = [o for key, config in loader.config.items()
                 if key.startswith('STEREO_A') and config.get('reference_frame') == 'inertial'
                 for o in config.get('observers', [])]
    if not observers:
        print('No STEREO-A observers with an inertial frame in the save state; no STEREO-A arrow', flush=True)
        return None
    time_unix = np.array([pd.Timestamp(o['time']).timestamp() for o in observers])
    lat = np.array([u.Quantity(o['latitude']).to_value(u.rad) for o in observers])
    lon = np.array([u.Quantity(o['longitude']).to_value(u.rad) for o in observers])
    direction = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    order = np.argsort(time_unix)
    return time_unix[order], direction[order]


def direction_arrow(direction, occ_max, fractions=(0.72, 0.95)):
    """Tail and head [R_sun] of an arrow along a unit vector, at fractions of the field-of-view radius."""
    return direction[None] * np.array(fractions)[:, None] * occ_max


def build_hci_coord_at_time(position_hci_rsun, time):
    return SkyCoord(
        CartesianRepresentation(
            position_hci_rsun[:, 0] * u.R_sun,
            position_hci_rsun[:, 1] * u.R_sun,
            position_hci_rsun[:, 2] * u.R_sun,
        ),
        frame=frames.HeliocentricInertial,
        obstime=time,
    )


def project_hci_positions_to_map(position_hci_rsun, observer_coord, time, ref_map, label):
    print(f'  Projecting {label} into rendered frame...', flush=True)
    t0 = perf_counter()
    hpc_frame = frames.Helioprojective(observer=observer_coord, obstime=time)
    position_hci_rsun = np.atleast_2d(position_hci_rsun)
    radius = np.linalg.norm(position_hci_rsun, axis=-1)
    print(
        f'    HCI xyz range [R_sun]: '
        f'x=({np.nanmin(position_hci_rsun[:, 0]):.2f}, {np.nanmax(position_hci_rsun[:, 0]):.2f}), '
        f'y=({np.nanmin(position_hci_rsun[:, 1]):.2f}, {np.nanmax(position_hci_rsun[:, 1]):.2f}), '
        f'z=({np.nanmin(position_hci_rsun[:, 2]):.2f}, {np.nanmax(position_hci_rsun[:, 2]):.2f}), '
        f'r=({np.nanmin(radius):.2f}, {np.nanmax(radius):.2f})',
        flush=True,
    )
    insitu_coord = build_hci_coord_at_time(position_hci_rsun, time)
    n_coord = insitu_coord.size if hasattr(insitu_coord, 'size') else len(np.atleast_1d(insitu_coord))
    print(f'    transform_to Helioprojective start: {n_coord} point(s)', flush=True)
    t_transform = perf_counter()
    hpc_coord = insitu_coord.transform_to(hpc_frame)
    print(f'    transform_to done in {perf_counter() - t_transform:.2f}s', flush=True)
    tx = np.atleast_1d(np.asarray(hpc_coord.Tx.to_value(u.arcsec), dtype=np.float64))
    ty = np.atleast_1d(np.asarray(hpc_coord.Ty.to_value(u.arcsec), dtype=np.float64))
    print(
        f'    HPC Tx/Ty range [arcsec]: '
        f'Tx=({np.nanmin(tx):.1f}, {np.nanmax(tx):.1f}), '
        f'Ty=({np.nanmin(ty):.1f}, {np.nanmax(ty):.1f})',
        flush=True,
    )
    print('    world_to_pixel start', flush=True)
    t_pixel = perf_counter()
    x_pix, y_pix = ref_map.world_to_pixel(hpc_coord)
    print(f'    world_to_pixel done in {perf_counter() - t_pixel:.2f}s', flush=True)
    x = np.atleast_1d(np.asarray(x_pix.value, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y_pix.value, dtype=np.float64))
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(tx)
    print(
        f'    pixel range: x=({np.nanmin(x):.1f}, {np.nanmax(x):.1f}), '
        f'y=({np.nanmin(y):.1f}, {np.nanmax(y):.1f})',
        flush=True,
    )
    print(f'  Projected {label} in {perf_counter() - t0:.2f}s', flush=True)
    return x, y, finite


def sample_time_indices(time_unix, available_idx, n_samples):
    if n_samples is None or n_samples <= 0 or len(available_idx) <= n_samples:
        return available_idx
    available_times = time_unix[available_idx]
    target_times = np.linspace(available_times[0], available_times[-1], n_samples)
    positions = np.searchsorted(available_times, target_times)
    positions = np.clip(positions, 0, len(available_idx) - 1)
    previous_positions = np.clip(positions - 1, 0, len(available_idx) - 1)

    next_delta = np.abs(available_times[positions] - target_times)
    previous_delta = np.abs(available_times[previous_positions] - target_times)
    nearest_positions = np.where(previous_delta < next_delta, previous_positions, positions)
    return available_idx[np.unique(nearest_positions)]


def sample_reconstruction(loader, times, position_hci_rsun, batch_size=2048):
    print(f'Sampling reconstructed density at {len(times)} probe position(s)...', flush=True)
    time_norm = np.asarray(loader.normalize_datetime(list(times)), dtype=np.float32)
    query_points = np.concatenate([
        np.asarray(position_hci_rsun, dtype=np.float32) / float(loader.Rs_per_ds),
        time_norm[:, None],
    ], axis=-1)
    model_out = loader.load_coords(query_points, batch_size=batch_size, progress=True)
    return {
        'times': np.asarray(times),
        'position_hci_rsun': np.asarray(position_hci_rsun, dtype=np.float64),
        'density_cm3': np.asarray(model_out['rho'], dtype=np.float64).reshape(-1),
    }


def normalize_to_insitu(reconstruction, name, observed_time_unix, observed):
    """Adjust the mean and the standard deviation of the reconstructed log density to the measurement.

    The statistics are taken in log10 of the density (the quantity shown on the axis; the result
    stays positive) at the times of the valid hourly in-situ averages, to which the reconstruction
    is interpolated, so that data gaps affect both series in the same way.
    """
    model = reconstruction['density_cm3']
    model_valid = np.isfinite(model) & (model > 0)
    log_model = np.log10(model[model_valid])
    model_at_observed = np.interp(observed_time_unix, reconstruction['time_unix'][model_valid], log_model)
    observed_valid = np.isfinite(observed) & (observed > 0)
    if model_valid.sum() < 2 or observed_valid.sum() < 2:
        raise ValueError(f'Not enough valid {name} samples to normalize the reconstruction')
    log_observed = np.log10(observed[observed_valid])
    model_mean, model_std = model_at_observed[observed_valid].mean(), model_at_observed[observed_valid].std()
    observed_mean, observed_std = log_observed.mean(), log_observed.std()
    print(f'Normalizing {name} reconstruction (log10 n_e): mean {model_mean:.3f} -> {observed_mean:.3f}, '
          f'std {model_std:.3f} -> {observed_std:.3f}', flush=True)
    normalized = np.full_like(model, np.nan)
    normalized[model_valid] = 10 ** ((log_model - model_mean) / model_std * observed_std + observed_mean)
    statistics = {'name': name, 'n_hours': int(observed_valid.sum()),
                  'start': observed_time_unix[observed_valid][0], 'end': observed_time_unix[observed_valid][-1],
                  'model_mean': model_mean, 'model_std': model_std,
                  'observed_mean': observed_mean, 'observed_std': observed_std}
    return normalized, statistics


def correlation_statistics(reconstruction, insitu, available_idx, observed):
    """Agreement of the reconstructed with the measured density for all samples and every density source.

    The reconstruction (before any normalization) is interpolated to the hourly averages of the
    measurement.  Correlations are given for log10(n_e) and for the radially compensated
    log10(n_e r^2), which removes the fall-off with distance that otherwise dominates them.
    """
    model = reconstruction['density_cm3']
    model_valid = np.isfinite(model) & (model > 0)
    radius = np.linalg.norm(insitu['position_hci_rsun'][available_idx], axis=-1)
    rows = []
    for source_label, observed_time_unix, observed_density in (
            [('all',) + observed['all']] + [(label, t, d) for label, _, t, d in observed['sources']]):
        valid = np.isfinite(observed_density) & (observed_density > 0)
        if model_valid.sum() < 2 or valid.sum() < 3:
            continue
        time_unix = observed_time_unix[valid]
        log_observed = np.log10(observed_density[valid])
        log_model = np.interp(time_unix, reconstruction['time_unix'][model_valid], np.log10(model[model_valid]))
        log_r2 = 2 * np.log10(np.interp(time_unix, insitu['time_unix'][available_idx], radius))
        difference = log_model - log_observed
        rows.append({
            'probe': insitu['name'], 'source': source_label, 'n_hours': int(valid.sum()),
            'pearson': pearsonr(log_model, log_observed)[0],
            'spearman': spearmanr(log_model, log_observed)[0],
            'pearson_r2': pearsonr(log_model + log_r2, log_observed + log_r2)[0],
            'spearman_r2': spearmanr(log_model + log_r2, log_observed + log_r2)[0],
            'bias': difference.mean(), 'rms': np.sqrt(np.mean(difference ** 2)),
        })
    return rows


def write_correlation(path, probes):
    """Text table of the agreement between the reconstructed and the measured density of every probe."""
    lines = [
        '# Reconstructed against measured in-situ density (video_psp_north), hourly averages of the measurement,',
        '# reconstruction without normalization.',
        '# pearson, spearman: correlation of log10(n_e [cm^-3]); *_r2: of the radially compensated log10(n_e r^2).',
        '# bias, rms: mean and root mean square of log10(n_model) - log10(n_observed) [dex].',
        '# probe source n_hours pearson spearman pearson_r2 spearman_r2 bias rms',
    ]
    for probe in probes:
        for row in probe['correlation']:
            lines.append(f'{row["probe"]} {row["source"]} {row["n_hours"]} {row["pearson"]:.4f} '
                         f'{row["spearman"]:.4f} {row["pearson_r2"]:.4f} {row["spearman_r2"]:.4f} '
                         f'{row["bias"]:.4f} {row["rms"]:.4f}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Correlation written to {path}', flush=True)


def write_normalization(path, probes):
    """Text table of the normalization of every probe, for a later calibration of the reconstruction."""
    lines = [
        '# Normalization of the reconstructed density to the in-situ measurement (video_psp_north).',
        '# Statistics of log10(n_e [cm^-3]) at the hourly averages of the measurement.',
        '# Calibrated density: log10(n_cal) = scale * log10(n_model) + offset,',
        '# with scale = observed_std / model_std and offset = observed_mean - scale * model_mean.',
        '# probe start_utc end_utc n_hours model_mean model_std observed_mean observed_std scale offset',
    ]
    for probe in probes:
        stats = probe['normalization']
        scale = stats['observed_std'] / stats['model_std']
        offset = stats['observed_mean'] - scale * stats['model_mean']
        start, end = (pd.to_datetime(stats[key], unit='s', utc=True).strftime('%Y-%m-%dT%H:%M:%S')
                      for key in ('start', 'end'))
        lines.append(f'{stats["name"]} {start} {end} {stats["n_hours"]} {stats["model_mean"]:.6f} '
                     f'{stats["model_std"]:.6f} {stats["observed_mean"]:.6f} {stats["observed_std"]:.6f} '
                     f'{scale:.6f} {offset:.6f}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Normalization written to {path}', flush=True)


def add_satellite_icon(ax, x, y, size=16, zorder=10, panel_color='#38bdf8'):
    width = size
    height = size * 0.75
    cx = width / 2
    cy = height / 2
    panel_w = size * 0.22
    panel_h = size * 0.42
    body_w = size * 0.28
    body_h = size * 0.34

    icon = DrawingArea(width, height, 0, 0)
    icon.add_artist(Line2D([cx - body_w / 2, cx - panel_w * 1.7], [cy, cy], color='white', lw=2.4))
    icon.add_artist(Line2D([cx + body_w / 2, cx + panel_w * 1.7], [cy, cy], color='white', lw=2.4))
    icon.add_artist(Line2D([cx - body_w / 2, cx - panel_w * 1.7], [cy, cy], color='#111827', lw=0.9))
    icon.add_artist(Line2D([cx + body_w / 2, cx + panel_w * 1.7], [cy, cy], color='#111827', lw=0.9))
    for panel_x in (cx - body_w / 2 - panel_w * 1.75, cx + body_w / 2 + panel_w * 0.75):
        icon.add_artist(Rectangle(
            (panel_x, cy - panel_h / 2),
            panel_w,
            panel_h,
            facecolor=panel_color,
            edgecolor='white',
            linewidth=1.4,
        ))
        icon.add_artist(Rectangle(
            (panel_x, cy - panel_h / 2),
            panel_w,
            panel_h,
            facecolor='none',
            edgecolor='#075985',
            linewidth=0.6,
        ))
    icon.add_artist(Rectangle(
        (cx - body_w / 2, cy - body_h / 2),
        body_w,
        body_h,
        facecolor='#f8fafc',
        edgecolor='white',
        linewidth=1.6,
    ))
    icon.add_artist(Rectangle(
        (cx - body_w / 2, cy - body_h / 2),
        body_w,
        body_h,
        facecolor='none',
        edgecolor='#111827',
        linewidth=0.7,
    ))
    icon.add_artist(Line2D([cx, cx + size * 0.18], [cy + body_h / 2, cy + height * 0.42], color='white', lw=2.0))
    icon.add_artist(Line2D([cx, cx + size * 0.18], [cy + body_h / 2, cy + height * 0.42], color='#111827', lw=0.7))
    icon.add_artist(Circle((cx + size * 0.2, cy + height * 0.44), radius=size * 0.045,
                           facecolor='#f97316', edgecolor='white', linewidth=0.8))

    annotation = ax.add_artist(AnnotationBbox(
        icon,
        (x, y),
        xycoords='data',
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        zorder=zorder,
    ))
    # an icon that sticks out of the axes must not change the layout of the figure
    annotation.set_in_layout(False)


def add_map_panel(fig, ax, pB_map, probes, frame_index, observer_coord, time, title, norm, colorbar=True,
                  ylabel=True, arrows=()):
    """Rendered pB image with the trajectory (colored by time) and the current position of every probe.

    ``norm`` is the color scale shared by the panels; ``colorbar`` draws it next to this panel.
    ``ylabel`` is False for a panel that shares the y axis of its left neighbour: ticks without labels.
    ``arrows`` holds (label, HCI positions [R_sun] of tail and head) of the arrows towards the
    observers of the white-light images.
    """
    print('  Plotting rendered image and probe trajectories...', flush=True)
    # image plane in projected solar radii: tangent-plane pixel offsets times the observer distance
    rsun_per_pixel = [float(scale.to_value(u.rad / u.pix)) * float(pB_map.dsun.to_value(u.R_sun))
                      for scale in pB_map.scale]
    center = [float(pB_map.reference_pixel.x.to_value(u.pix)), float(pB_map.reference_pixel.y.to_value(u.pix))]
    ny, nx = pB_map.data.shape
    extent = [(-0.5 - center[0]) * rsun_per_pixel[0], (nx - 0.5 - center[0]) * rsun_per_pixel[0],
              (-0.5 - center[1]) * rsun_per_pixel[1], (ny - 0.5 - center[1]) * rsun_per_pixel[1]]

    def to_rsun(x_pix, y_pix):
        return (x_pix - center[0]) * rsun_per_pixel[0], (y_pix - center[1]) * rsun_per_pixel[1]

    im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=norm, origin='lower', extent=extent)
    ax.set_facecolor('black')
    # the Sun to scale
    ax.add_patch(Circle((0.0, 0.0), 1.0, color='yellow', linewidth=0, zorder=5))
    ax.set_xlabel(r'HPC-x [R$_\odot$]')
    if ylabel:
        ax.set_ylabel(r'HPC-y [R$_\odot$]')
    else:
        ax.tick_params(labelleft=False)
    divider = make_axes_locatable(ax)
    if colorbar:
        # an inset next to the image keeps the size of the panels equal; the layout leaves room for it
        fig.colorbar(im, cax=ax.inset_axes([1.03, 0.0, 0.04, 1.0]), label='pB [MSB]')

    for probe in probes:
        trajectory = probe['trajectory']
        norm = Normalize(vmin=trajectory['time_unix'][0], vmax=trajectory['time_unix'][-1])
        x, y, finite = project_hci_positions_to_map(
            trajectory['position_hci_rsun'], observer_coord, time, pB_map, f'equal-time {probe["name"]} trajectory')
        print(f'  {probe["name"]} trajectory projected: {int(np.sum(finite))} finite sampled points', flush=True)
        x, y = to_rsun(x, y)
        ax.scatter(x[finite], y[finite], c=trajectory['time_unix'][finite], cmap=probe['cmap'], norm=norm, s=5,
                   linewidths=0, alpha=0.85)

        current_position = probe['frame_positions'][frame_index]
        if not np.all(np.isfinite(current_position)):
            continue
        current_x, current_y, current_finite = project_hci_positions_to_map(
            current_position, observer_coord, time, pB_map, f'current {probe["name"]} point')
        current_x, current_y = to_rsun(current_x, current_y)
        if current_finite[0]:
            add_satellite_icon(ax, current_x[0], current_y[0], size=15, zorder=10, panel_color=probe['color'])
            ax.annotate(probe['name'], (current_x[0], current_y[0]), xytext=(0, 11), textcoords='offset points',
                        ha='center', va='bottom', color='white', fontsize=8, zorder=11).set_in_layout(False)

    for arrow_label, arrow_positions in arrows:
        arrow_x, arrow_y, _ = project_hci_positions_to_map(arrow_positions, observer_coord, time, pB_map,
                                                           f'{arrow_label} arrow')
        arrow_x, arrow_y = to_rsun(arrow_x, arrow_y)
        ax.annotate('', xy=(arrow_x[1], arrow_y[1]), xytext=(arrow_x[0], arrow_y[0]),
                    arrowprops=dict(arrowstyle='-|>', color='white', lw=1.5), zorder=9)
        # the label sits behind the tail, on the line of the arrow
        ax.text(arrow_x[0] - 0.25 * (arrow_x[1] - arrow_x[0]), arrow_y[0] - 0.25 * (arrow_y[1] - arrow_y[0]),
                arrow_label, color='white', fontsize=9, ha='center', va='center', zorder=9)

    # one colorbar per probe: every trajectory covers its own period
    for index, probe in enumerate(probes):
        time_unix = probe['trajectory']['time_unix']
        sm = ScalarMappable(norm=Normalize(vmin=time_unix[0], vmax=time_unix[-1]), cmap=probe['cmap'])
        sm.set_array([])
        cax_traj = divider.append_axes('bottom', size='4%', pad=0.6 if index == 0 else 0.3, axes_class=plt.Axes)
        cb = fig.colorbar(sm, cax=cax_traj, orientation='horizontal')
        ticks = np.linspace(time_unix[0], time_unix[-1], 4)
        cb.set_ticks(ticks)
        cb.set_ticklabels([pd.to_datetime(t, unit='s', utc=True).strftime('%m-%d') for t in ticks])
        cb.ax.tick_params(labelsize=8)
        # the outer labels stay inside the colorbar, clear of the neighbouring panel
        tick_labels = cb.ax.get_xticklabels()
        tick_labels[0].set_ha('left')
        tick_labels[-1].set_ha('right')
        if ylabel:
            cax_traj.set_ylabel(probe['name'], rotation=0, ha='right', va='center', fontsize=8)
    cb.set_label('Trajectory time')

    ax.set_title(title)


def add_overview_panel(ax, probe, current_time_unix, time_limits, xlabel=True):
    """Observed and reconstructed electron density along one probe.

    The panels share the time range ``time_limits``; only the bottom panel (``xlabel``) labels its ticks.
    """
    print(f'  Plotting {probe["name"]} in-situ density overview...', flush=True)
    reconstruction = probe['reconstruction']
    reconstruction_time_unix = reconstruction['time_unix']

    # hourly averages; the density sources of a probe with several instruments are drawn separately
    observed = probe['observed']
    curves = observed['sources'] or [('', 'black') + observed['all']]
    for source_label, color, observed_time_unix, observed_density in curves:
        observed_times = pd.to_datetime(observed_time_unix, unit='s', utc=True).tz_convert(None).to_pydatetime()
        ax.plot(observed_times, observed_density, color=color, lw=0.9,
                label=f'Observed {source_label}'.strip())
    # the unnormalized reconstruction stays visible behind the normalized curve
    ax.plot(reconstruction['times'], reconstruction['density_cm3_original'], color='tab:orange', lw=1.1,
            ls='--', alpha=0.7, label='SuNeRF')
    ax.plot(reconstruction['times'], reconstruction['density_cm3'], color='tab:orange', lw=1.1,
            label='SuNeRF (normalized)')
    # the icon marks the current time while the probe is inside the reconstructed range
    if reconstruction_time_unix[0] - 1.0 <= current_time_unix <= reconstruction_time_unix[-1] + 1.0:
        current_idx = int(np.nanargmin(np.abs(reconstruction_time_unix - current_time_unix)))
        if np.isfinite(reconstruction['density_cm3'][current_idx]):
            add_satellite_icon(ax, reconstruction['times'][current_idx], reconstruction['density_cm3'][current_idx],
                               size=12, zorder=7, panel_color=probe['color'])
    ax.set_yscale('log')
    ax.set_ylabel(r'$n_e$ [cm$^{-3}$]')
    ax.set_xlim(*time_limits)
    ax.grid(alpha=0.25)
    ax.set_title(f'{probe["name"]} in-situ density')
    ax.legend(loc='best', fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    if xlabel:
        ax.set_xlabel('Time')
    else:
        ax.tick_params(labelbottom=False)


def build_probe(loader, insitu, available_idx, color, frame_time_unix, args):
    """Trajectory, reconstruction and frame positions of one probe for its available samples."""
    probe_start, probe_end = insitu['times'][available_idx[0]], insitu['times'][available_idx[-1]]
    trajectory_count = (len(available_idx) if args.max_trajectory_points <= 0
                        else max(2, min(args.max_trajectory_points, len(available_idx))))
    trajectory_times, trajectory_time_unix = equal_time_samples(probe_start, probe_end, trajectory_count)
    _, reconstruction_time_unix = equal_time_samples(probe_start, probe_end,
                                                     max(args.reconstruction_points, args.n_frames))
    # the frames inside the range of the probe are part of the reconstructed curve; one second of
    # tolerance keeps the first and the last frame, whose times are rounded to the datetime resolution
    inside = ((frame_time_unix >= reconstruction_time_unix[0] - 1.0)
              & (frame_time_unix <= reconstruction_time_unix[-1] + 1.0))
    inside_time_unix = np.clip(frame_time_unix[inside], reconstruction_time_unix[0], reconstruction_time_unix[-1])
    reconstruction_time_unix = np.unique(np.concatenate([reconstruction_time_unix, inside_time_unix]))
    reconstruction_times = pd.to_datetime(reconstruction_time_unix, unit='s', utc=True).tz_convert(None).to_pydatetime()

    frame_positions = np.full((len(frame_time_unix), 3), np.nan)
    if inside.any():
        frame_positions[inside] = interpolate_positions(insitu, inside_time_unix, 'frame')
    reconstruction = sample_reconstruction(
        loader, reconstruction_times, interpolate_positions(insitu, reconstruction_time_unix, 'reconstruction'))
    reconstruction['time_unix'] = reconstruction_time_unix
    observed = observed_series(insitu, available_idx)
    correlation = correlation_statistics(reconstruction, insitu, available_idx, observed)
    reconstruction['density_cm3_original'] = reconstruction['density_cm3']
    reconstruction['density_cm3'], normalization = normalize_to_insitu(
        reconstruction, insitu['name'], *observed['all'])
    return {
        'name': insitu['name'],
        'color': color,
        'cmap': PROBE_CMAPS[insitu['name']],
        'insitu': insitu,
        'available_idx': available_idx,
        'observed': observed,
        'normalization': normalization,
        'correlation': correlation,
        'trajectory': {
            'times': trajectory_times,
            'time_unix': trajectory_time_unix,
            'position_hci_rsun': interpolate_positions(insitu, trajectory_time_unix, 'trajectory'),
        },
        'reconstruction': reconstruction,
        'frame_positions': frame_positions,
    }


def main():
    args = build_parser().parse_args()
    if args.n_frames < 1:
        raise ValueError('--n_frames must be at least 1')

    os.makedirs(args.out_path, exist_ok=True)

    print(f'Loading SuNeRF state: {args.sunerf_path}', flush=True)
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    start_time, end_time = resolve_time_range(sunerf_loader, args.time_start, args.time_end)
    # every probe is shown for all of its samples inside its radius limit
    psp = load_insitu(args.insitu_path, 'PSP')
    selected = [(psp, available_insitu_indices(psp, start_time, end_time, args.max_plot_radius_rsun))]
    if args.check_horizons:
        check_horizons_positions(*selected[0], args.horizons_check_points)
    if args.solo_path is not None:
        solo = load_insitu(args.solo_path, 'SolO')
        selected.append((solo, available_insitu_indices(solo, start_time, end_time, args.max_plot_radius_rsun)))

    # the frames cover the combined period of the time series of all probes
    plot_start_time = min(insitu['times'][idx[0]] for insitu, idx in selected)
    plot_end_time = max(insitu['times'][idx[-1]] for insitu, idx in selected)
    frame_times, frame_time_unix = equal_time_samples(plot_start_time, plot_end_time, args.n_frames)
    print(
        f'Rendering {len(frame_times)} equal-time frames from {plot_start_time} to {plot_end_time} '
        f'({", ".join(insitu["name"] for insitu, _ in selected)})',
        flush=True,
    )
    probes = [build_probe(sunerf_loader, insitu, idx, PROBE_COLORS[insitu['name']], frame_time_unix, args)
              for insitu, idx in selected]
    write_correlation(os.path.join(args.out_path, 'insitu_correlation.txt'), probes)
    write_normalization(os.path.join(args.out_path, 'insitu_normalization.txt'), probes)

    stereo_a = stereo_a_observers(sunerf_loader)
    distance = args.distance_rsun * u.R_sun
    resolution = (args.resolution, args.resolution) * u.pix
    occ_min = args.occ_min * u.R_sun
    occ_max = args.occ_max * u.R_sun

    for i, time in tqdm(list(enumerate(frame_times)), total=len(frame_times)):
        frame_path = os.path.join(args.out_path, f'psp_north_frame{i:03d}.jpg')
        if os.path.exists(frame_path) and not args.overwrite:
            print(f'Skipping existing frame: {frame_path}')
            continue

        print(f'Frame {i + 1}/{len(frame_times)}: {time.isoformat(" ", timespec="minutes")}', flush=True)
        earth_observer_coord = get_earth_observer_coord(time)
        observer_coord = SkyCoord(
            lat=89 * u.deg,
            lon=earth_observer_coord.lon,
            distance=distance,
            frame=frames.HeliocentricInertial,
            obstime=time,
        )
        print('  Rendering SuNeRF solar-north image...', flush=True)
        north_model_out = sunerf_loader.load_image(
            observer_coord.lat,
            observer_coord.lon,
            time,
            distance=distance,
            resolution=resolution,
            occ_min=occ_min,
            occ_max=occ_max,
            progress=False,
        )
        print('  Rendering SuNeRF Earth-perspective pB image...', flush=True)
        earth_model_out = sunerf_loader.load_image(
            earth_observer_coord.lat,
            earth_observer_coord.lon,
            time,
            distance=earth_observer_coord.spherical.distance,
            resolution=resolution,
            occ_min=occ_min,
            occ_max=occ_max,
            progress=False,
        )
        print('  SuNeRF images rendered', flush=True)
        north_pB_map = north_model_out['pB_map']
        earth_pB_map = earth_model_out['pB_map']

        # one density panel per probe on the right
        fig = plt.figure(figsize=(13.1, 4.8), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)
        # column widths in inches: the left map carries the y labels, the third column the shared pB colorbar
        gs = fig.add_gridspec(len(probes), 4, width_ratios=(3.8, 3.3, 0.65, 5.1))
        ax_earth_map = fig.add_subplot(gs[:, 0])
        ax_north_map = fig.add_subplot(gs[:, 1])

        # arrows towards Earth and STEREO-A near the edge of the solar-north field of view
        earth_direction = earth_observer_coord.cartesian.xyz.to_value(u.R_sun)
        arrows = [('Earth', direction_arrow(earth_direction / np.linalg.norm(earth_direction), args.occ_max))]
        if stereo_a is not None:
            # STEREO-A moves about 1 deg per day; the nearest observation is used
            nearest = int(np.argmin(np.abs(stereo_a[0] - frame_time_unix[i])))
            # STEREO-A lies towards a corner of the panel, so its arrow sits outside the rendered disk
            arrows.append(('STEREO-A', direction_arrow(stereo_a[1][nearest], args.occ_max, (0.97, 1.2))))
        # one color scale for both views, with the colorbar next to the right panel
        positive = np.concatenate([m.data[np.isfinite(m.data) & (m.data > 0)] for m in (north_pB_map, earth_pB_map)])
        norm = LogNorm(vmin=positive.min(), vmax=PB_MAX)
        add_map_panel(fig, ax_north_map, north_pB_map, probes, i, observer_coord, time,
                      'Solar north view', norm, ylabel=False, arrows=arrows)
        add_map_panel(fig, ax_earth_map, earth_pB_map, probes, i, earth_observer_coord, time,
                      'Earth view', norm, colorbar=False)
        # the density panels share the time axis of the combined period
        ax_overview = None
        for row, probe in enumerate(probes):
            ax_overview = fig.add_subplot(gs[row, 3], sharex=ax_overview)
            add_overview_panel(ax_overview, probe, frame_time_unix[i], (plot_start_time, plot_end_time),
                               xlabel=row == len(probes) - 1)

        fig.suptitle(time.isoformat(' ', timespec='minutes'), fontsize=14)
        print(f'  Saving frame: {frame_path}', flush=True)
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close(fig)
        print('  Frame done', flush=True)


if __name__ == '__main__':
    main()
