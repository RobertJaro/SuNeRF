import argparse
import os
from time import perf_counter

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.visualization.colormaps import cm
from tqdm import tqdm

from sunerf.evaluation.loader import ThomsonSuNeRFLoader


INERTIAL_COORDS_KEY = 'inertial_coords_r_lat_lon'


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
    parser.add_argument('--out_path', type=str, required=True, help='Path to output frame directory')
    parser.add_argument('--n_frames', '--N', dest='n_frames', type=int, default=80, help='Number of frames to render')
    parser.add_argument('--time_start', type=parse_time, default=None, help='First rendered time')
    parser.add_argument('--time_end', type=parse_time, default=None, help='Last rendered time')
    parser.add_argument('--distance_rsun', type=float, default=80.0, help='Observer distance in solar radii')
    parser.add_argument('--resolution', type=int, default=512, help='Square render resolution in pixels')
    parser.add_argument('--occ_min', type=float, default=2.5, help='Inner projected radius in solar radii')
    parser.add_argument('--occ_max', type=float, default=80.0, help='Outer projected radius in solar radii')
    parser.add_argument('--max_plot_radius_rsun', type=float, default=60.0,
                        help='Only use PSP samples at or below this radius for trajectory and frame sampling')
    parser.add_argument('--max_trajectory_points', type=int, default=128,
                        help='Maximum PSP trajectory samples plotted per frame')
    parser.add_argument('--reconstruction_points', type=int, default=128,
                        help='Number of equal-time PSP Horizons samples used for reconstructed in-situ curves')
    parser.add_argument('--check_horizons', action='store_true',
                        help='Compare sampled PSP npz positions against JPL Horizons and print full coordinate deltas')
    parser.add_argument('--horizons_check_points', type=int, default=12,
                        help='Number of equal-time PSP samples to compare when --check_horizons is used')
    parser.add_argument('--dpi', type=int, default=200, help='Saved frame DPI')
    parser.add_argument('--overwrite', action='store_true', help='Re-render frames that already exist')
    return parser


def load_insitu(path):
    print(f'Loading PSP in-situ data: {path}', flush=True)
    data = np.load(path, allow_pickle=True)
    required = ['time_unix', 'density_cm3', 'velocity_radial_kms']
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
        print(f'PSP position metadata: source={position_source}, coordinate_system={coordinate_system}', flush=True)
    radius_check = np.linalg.norm(position, axis=-1)
    if 'radius_rsun' in data:
        radius_delta = radius_check - data['radius_rsun'].astype(np.float64)[order]
        print(
            f'PSP position sanity: |xyz| radius range=({np.nanmin(radius_check):.2f}, {np.nanmax(radius_check):.2f}) R_sun, '
            f'|xyz|-radius max abs delta={np.nanmax(np.abs(radius_delta)):.3e} R_sun',
            flush=True,
        )
    else:
        print(
            f'PSP position sanity: |xyz| radius range=({np.nanmin(radius_check):.2f}, {np.nanmax(radius_check):.2f}) R_sun',
            flush=True,
        )

    return {
        'time_unix': time_unix[order],
        'times': np.array(times),
        'position_hci_rsun': position,
        'density_cm3': data['density_cm3'].astype(np.float64)[order],
        'velocity_radial_kms': data['velocity_radial_kms'].astype(np.float64)[order],
        'radius_rsun': data['radius_rsun'].astype(np.float64)[order] if 'radius_rsun' in data else None,
    }


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
    mask = (insitu['time_unix'] >= start_unix) & (insitu['time_unix'] <= end_unix)
    if insitu['radius_rsun'] is not None:
        mask &= insitu['radius_rsun'] <= max_radius_rsun
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(
            f'No PSP samples are available between {start_time} and {end_time} '
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


def query_psp_horizons_hci_positions(times, label):
    print(f'Querying PSP Horizons positions for {len(times)} {label} time(s)...', flush=True)
    positions = []
    t0 = perf_counter()
    for time in tqdm(times, desc=f'PSP Horizons ({label})'):
        positions.append(query_psp_horizons_hci(time))
    positions = np.asarray(positions, dtype=np.float64)
    radius = np.linalg.norm(positions, axis=-1)
    print(
        f'  PSP Horizons position radius range=({np.nanmin(radius):.2f}, {np.nanmax(radius):.2f}) R_sun; '
        f'loaded in {perf_counter() - t0:.2f}s',
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
    print(f'Sampling reconstructed density/velocity at {len(times)} PSP Horizons point(s)...', flush=True)
    time_norm = np.asarray(loader.normalize_datetime(list(times)), dtype=np.float32)
    query_points = np.concatenate([
        np.asarray(position_hci_rsun, dtype=np.float32) / float(loader.Rs_per_ds),
        time_norm[:, None],
    ], axis=-1)
    model_out = loader.load_coords(query_points, batch_size=batch_size, progress=True)
    density = np.asarray(model_out['rho'], dtype=np.float64).reshape(-1)
    velocity = np.asarray(model_out['v'], dtype=np.float64).reshape(-1, 3)
    radius = np.linalg.norm(position_hci_rsun, axis=-1)
    r_hat = position_hci_rsun / np.clip(radius[:, None], 1e-12, None)
    velocity_radial = np.sum(velocity * r_hat, axis=-1)
    return {
        'times': np.asarray(times),
        'position_hci_rsun': np.asarray(position_hci_rsun, dtype=np.float64),
        'density_cm3': density,
        'velocity_radial_kms': velocity_radial,
    }


def add_satellite_icon(ax, x, y, size=16, zorder=10):
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
            facecolor='#38bdf8',
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

    ax.add_artist(AnnotationBbox(
        icon,
        (x, y),
        xycoords='data',
        frameon=False,
        box_alignment=(0.5, 0.5),
        pad=0,
        zorder=zorder,
    ))


def add_map_panel(fig, ax, pB_map, trajectory, observer_coord, time, current_position_hci_rsun, title):
    print('  Plotting rendered image and PSP trajectory...', flush=True)
    im = ax.imshow(pB_map.data, cmap=cm.soholasco2, norm=LogNorm(), origin='lower')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax, label='pB [MSB]')

    pB_map.draw_grid(ax, color='white', alpha=0.35, linewidth=0.6)
    ax.coords[0].set_axislabel(' ')
    ax.coords[1].set_axislabel(' ')

    x, y, finite = project_hci_positions_to_map(
        trajectory['position_hci_rsun'],
        observer_coord,
        time,
        pB_map,
        'equal-time PSP Horizons trajectory',
    )
    print(
        f'  PSP trajectory projected: {int(np.sum(finite))} finite sampled points',
        flush=True,
    )
    available_time_unix = trajectory['time_unix']
    norm = Normalize(vmin=np.nanmin(available_time_unix), vmax=np.nanmax(available_time_unix))
    ax.scatter(
        x[finite],
        y[finite],
        c=trajectory['time_unix'][finite],
        cmap='viridis',
        norm=norm,
        s=5,
        linewidths=0,
        alpha=0.85,
    )

    current_x, current_y, current_finite = project_hci_positions_to_map(
        current_position_hci_rsun,
        observer_coord,
        time,
        pB_map,
        'current PSP point',
    )
    if current_finite[0]:
        add_satellite_icon(ax, current_x[0], current_y[0], size=15, zorder=10)

    sm = ScalarMappable(norm=norm, cmap='viridis')
    sm.set_array([])
    cax_traj = divider.append_axes('bottom', size='4%', pad=0.45, axes_class=plt.Axes)
    cb = fig.colorbar(sm, cax=cax_traj, orientation='horizontal')
    cb.set_label('PSP trajectory time')
    ticks = np.linspace(np.nanmin(available_time_unix), np.nanmax(available_time_unix), 4)
    cb.set_ticks(ticks)
    cb.set_ticklabels([
        pd.to_datetime(t, unit='s', utc=True).strftime('%m-%d') for t in ticks
    ])

    ax.set_title(title)


def add_overview_panel(ax_density, ax_velocity, insitu, reconstruction, current_obs_pos, current_reconstruction_idx,
                       available_idx):
    print('  Plotting in-situ density/velocity overview...', flush=True)
    times = insitu['times'][available_idx]
    density = insitu['density_cm3'][available_idx]
    velocity = insitu['velocity_radial_kms'][available_idx]

    ax_density.plot(times, density, color='black', lw=0.9, label='Observed')
    ax_density.plot(
        reconstruction['times'],
        reconstruction['density_cm3'],
        color='tab:orange',
        lw=1.1,
        label='SuNeRF',
    )
    reconstruction_density = reconstruction['density_cm3'][current_reconstruction_idx]
    if np.isfinite(reconstruction_density) and reconstruction_density > 0:
        add_satellite_icon(
            ax_density,
            reconstruction['times'][current_reconstruction_idx],
            reconstruction_density,
            size=12,
            zorder=7,
        )
    ax_density.set_yscale('log')
    ax_density.set_ylabel(r'$n_e$ [cm$^{-3}$]')
    ax_density.set_title('PSP in-situ overview')
    ax_density.legend(loc='best', fontsize=8)
    ax_density.grid(alpha=0.25)

    ax_velocity.plot(times, velocity, color='tab:blue', lw=0.9, label='Observed')
    ax_velocity.plot(
        reconstruction['times'],
        reconstruction['velocity_radial_kms'],
        color='tab:orange',
        lw=1.1,
        label='SuNeRF',
    )
    reconstruction_velocity = reconstruction['velocity_radial_kms'][current_reconstruction_idx]
    if np.isfinite(reconstruction_velocity):
        add_satellite_icon(
            ax_velocity,
            reconstruction['times'][current_reconstruction_idx],
            reconstruction_velocity,
            size=12,
            zorder=7,
        )
    ax_velocity.set_ylabel(r'$v_r$ [km s$^{-1}$]')
    ax_velocity.set_xlabel('Time')
    ax_velocity.grid(alpha=0.25)
    for label in ax_velocity.get_xticklabels():
        label.set_rotation(30)
        label.set_ha('right')


def main():
    args = build_parser().parse_args()
    if args.n_frames < 1:
        raise ValueError('--n_frames must be at least 1')

    os.makedirs(args.out_path, exist_ok=True)

    print(f'Loading SuNeRF state: {args.sunerf_path}', flush=True)
    sunerf_loader = ThomsonSuNeRFLoader(args.sunerf_path)
    insitu = load_insitu(args.insitu_path)
    start_time, end_time = resolve_time_range(sunerf_loader, args.time_start, args.time_end)
    available_idx = available_insitu_indices(insitu, start_time, end_time, args.max_plot_radius_rsun)
    if args.check_horizons:
        check_horizons_positions(insitu, available_idx, args.horizons_check_points)

    plot_start_time = insitu['times'][available_idx[0]]
    plot_end_time = insitu['times'][available_idx[-1]]
    if args.max_trajectory_points <= 0:
        trajectory_count = len(available_idx)
    else:
        trajectory_count = min(args.max_trajectory_points, len(available_idx))
    frame_times, frame_time_unix = equal_time_samples(plot_start_time, plot_end_time, args.n_frames)
    trajectory_times, trajectory_time_unix = equal_time_samples(
        plot_start_time,
        plot_end_time,
        trajectory_count,
    )
    reconstruction_times, reconstruction_time_unix = equal_time_samples(
        plot_start_time,
        plot_end_time,
        max(args.reconstruction_points, args.n_frames),
    )
    reconstruction_time_unix = np.unique(np.concatenate([reconstruction_time_unix, frame_time_unix]))
    reconstruction_times = pd.to_datetime(reconstruction_time_unix, unit='s', utc=True).tz_convert(None).to_pydatetime()

    print(
        f'Rendering {len(frame_times)} equal-time frames from {len(available_idx)} observed PSP samples '
        f'with radius <= {args.max_plot_radius_rsun:.1f} R_sun',
        flush=True,
    )
    frame_positions_hci = query_psp_horizons_hci_positions(frame_times, 'frame')
    trajectory_positions_hci = query_psp_horizons_hci_positions(trajectory_times, 'trajectory')
    reconstruction_positions_hci = query_psp_horizons_hci_positions(reconstruction_times, 'reconstruction')
    trajectory = {
        'times': trajectory_times,
        'time_unix': trajectory_time_unix,
        'position_hci_rsun': trajectory_positions_hci,
    }
    reconstruction = sample_reconstruction(sunerf_loader, reconstruction_times, reconstruction_positions_hci)

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
        current_position_hci = frame_positions_hci[i]
        current_available_pos = nearest_available_position(insitu['time_unix'], available_idx, time)
        current_reconstruction_idx = int(np.nanargmin(np.abs(reconstruction_time_unix - frame_time_unix[i])))
        print(
            f'  Current PSP Horizons radius: {np.linalg.norm(current_position_hci):.2f} R_sun; '
            f'nearest observed sample index: {available_idx[current_available_pos]}',
            flush=True,
        )

        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=(1.2, 1.2, 1.0), height_ratios=(1, 1))
        ax_north_map = fig.add_subplot(gs[:, 0], projection=north_pB_map)
        ax_earth_map = fig.add_subplot(gs[:, 1], projection=earth_pB_map)
        ax_density = fig.add_subplot(gs[0, 2])
        ax_velocity = fig.add_subplot(gs[1, 2], sharex=ax_density)

        add_map_panel(
            fig,
            ax_north_map,
            north_pB_map,
            trajectory,
            observer_coord,
            time,
            current_position_hci,
            'Solar north view with PSP trajectory',
        )
        add_map_panel(
            fig,
            ax_earth_map,
            earth_pB_map,
            trajectory,
            earth_observer_coord,
            time,
            current_position_hci,
            'Earth view pB with PSP trajectory',
        )
        add_overview_panel(
            ax_density,
            ax_velocity,
            insitu,
            reconstruction,
            current_available_pos,
            current_reconstruction_idx,
            available_idx,
        )

        fig.suptitle(
            f'Solar north, {args.distance_rsun:.0f} R_sun | '
            f'{time.isoformat(" ", timespec="minutes")}',
            fontsize=14,
        )
        print(f'  Saving frame: {frame_path}', flush=True)
        fig.savefig(frame_path, dpi=args.dpi)
        plt.close(fig)
        print('  Frame done', flush=True)


if __name__ == '__main__':
    main()
