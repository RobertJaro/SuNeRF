import argparse
import os

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header
from sunpy.visualization.colormaps import cm

from sunerf.data.ray_sampling import get_rays
from sunerf.evaluation.loader import ThomsonSuNeRFLoader, _get_mask, _get_scale_from_occ_max
from sunerf.evaluation.cme.video import get_input_observer_coord
from sunerf.physics.thomson import R_SUN_CM
from sunerf.train.coordinate_transformation import pose_spherical

R_SUN_M = R_SUN_CM * 1e-2


def parse_time(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.to_pydatetime()


def build_parser():
    parser = argparse.ArgumentParser(
        description='Export the CME electron-density profile along one helioprojective line of sight.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to SuNeRF save state')
    parser.add_argument('--out_path', type=str, default=None, help='Output directory')
    parser.add_argument('--time', type=parse_time, required=True, help='Observation time')
    parser.add_argument('--hpc', type=float, nargs=2, default=[3000.0, 5000.0],
                        metavar=('TX_ARCSEC', 'TY_ARCSEC'), help='Target HPC coordinate in arcsec')
    parser.add_argument('--point', nargs=3, action='append', default=None,
                        metavar=('LABEL', 'TX_ARCSEC', 'TY_ARCSEC'),
                        help='Named HPC point. Can be repeated, e.g. --point A 3000 5000 --point B 2000 5000')
    parser.add_argument('--observer', type=str, default='earth',
                        choices=['earth', 'stereo-a', 'custom'], help='Observer used for the HPC coordinate')
    parser.add_argument('--observer_lat', type=float, default=None, help='Custom observer latitude in degrees')
    parser.add_argument('--observer_lon', type=float, default=None, help='Custom observer longitude in degrees')
    parser.add_argument('--observer_frame', type=str, choices=['hci', 'carrington'], default='hci',
                        help='Coordinate frame for custom observer latitude/longitude')
    parser.add_argument('--distance_au', type=float, default=None,
                        help='Observer distance in AU; defaults to ephemeris distance for named observers or 1 AU')
    parser.add_argument('--resolution', type=int, default=256, help='Square forward-rendered image size in pixels')
    parser.add_argument('--occ_min', type=float, default=3.0, help='Inner occulting radius in solar radii')
    parser.add_argument('--occ_max', type=float, default=20.0, help='Outer rendered radius in solar radii')
    parser.add_argument('--profile_samples', type=int, default=512,
                        help='Number of equally spaced samples along the LOS profile')
    parser.add_argument('--profile_radius_min', type=float, default=None,
                        help='Minimum heliocentric radius to keep in the profile in solar radii; defaults to --occ_min')
    parser.add_argument('--profile_radius_max', type=float, default=None,
                        help='Maximum heliocentric radius for the LOS profile in solar radii; defaults to --occ_max')
    parser.add_argument('--batch_size', type=int, default=2 ** 11, help='Rendering batch size')
    parser.add_argument('--dpi', type=int, default=300, help='Saved figure DPI')
    return parser


def resolve_points(args):
    if args.point is None:
        return [('point', float(args.hpc[0]), float(args.hpc[1]))]

    points = []
    for label, tx, ty in args.point:
        points.append((label, float(tx), float(ty)))
    return points


def resolve_observer(args):
    obstime = args.time
    if args.observer == 'earth':
        coord = get_body_heliographic_stonyhurst('earth', obstime)
        coord = coord.transform_to(frames.HeliocentricInertial(obstime=obstime))
    elif args.observer == 'stereo-a':
        coord = get_horizons_coord('STEREO-A', obstime)
        coord = coord.transform_to(frames.HeliocentricInertial(obstime=obstime))
    else:
        if args.observer_lat is None or args.observer_lon is None:
            raise ValueError('--observer_lat and --observer_lon are required for --observer custom')
        distance = (args.distance_au if args.distance_au is not None else 1.0) * u.AU
        coord = get_input_observer_coord(
            args.observer_lat * u.deg,
            args.observer_lon * u.deg,
            distance,
            obstime,
            args.observer_frame,
        )
        coord = coord.transform_to(frames.HeliocentricInertial(obstime=obstime))

    if args.distance_au is not None and args.observer != 'custom':
        coord = SkyCoord(
            lat=coord.lat,
            lon=coord.lon,
            distance=args.distance_au * u.AU,
            frame=frames.HeliocentricInertial,
            obstime=obstime,
        )
    return coord


def build_sun_centered_reference_map(observer_coord, time, resolution, scale, occ_min, occ_max):
    sun_center = SkyCoord(
        0 * u.arcsec,
        0 * u.arcsec,
        obstime=time,
        observer=observer_coord,
        frame=frames.Helioprojective,
    )
    mock_data = np.zeros([int(r.to_value(u.pix)) for r in resolution], dtype=np.float32)
    header = make_fitswcs_header(mock_data, sun_center, scale=scale)
    ref_map = Map(mock_data, header)
    mask = _get_mask(ref_map, occ_min, occ_max)
    ref_map.data[mask] = np.nan
    return ref_map


def sample_equal_spaced_los_profile(loader, time, ray_origin_model, ray_direction, radius_min, radius_max,
                                    n_samples, batch_size):
    if n_samples < 2:
        raise ValueError('--profile_samples must be at least 2')

    outer_radius_model = radius_max.to_value(u.R_sun) / loader.Rs_per_ds
    inner_radius_model = None if radius_min is None else radius_min.to_value(u.R_sun) / loader.Rs_per_ds

    a = float(np.dot(ray_direction, ray_direction))
    b = float(2.0 * np.dot(ray_origin_model, ray_direction))
    c = float(np.dot(ray_origin_model, ray_origin_model) - outer_radius_model ** 2)
    discriminant = b ** 2 - 4.0 * a * c
    if discriminant < 0:
        raise ValueError('The requested LOS does not intersect the profile outer radius.')

    sqrt_discriminant = np.sqrt(discriminant)
    z_near = (-b - sqrt_discriminant) / (2.0 * a)
    z_far = (-b + sqrt_discriminant) / (2.0 * a)
    z_model = np.linspace(z_near, z_far, n_samples, dtype=np.float64)
    sample_xyz_model = ray_origin_model[None] + z_model[:, None] * ray_direction[None]
    sample_hci_xyz_rsun = sample_xyz_model * loader.Rs_per_ds
    sample_hci_xyz_m = sample_hci_xyz_rsun * R_SUN_M
    heliocentric_radius_rsun = np.linalg.norm(sample_hci_xyz_rsun, axis=-1)
    heliocentric_radius_m = heliocentric_radius_rsun * R_SUN_M
    sample_hci_lat_deg = np.rad2deg(
        np.arcsin(np.clip(sample_hci_xyz_rsun[:, 2] / heliocentric_radius_rsun, -1.0, 1.0))
    )
    sample_hci_lon_deg = np.rad2deg(np.arctan2(sample_hci_xyz_rsun[:, 1], sample_hci_xyz_rsun[:, 0]))
    keep_mask = np.ones(n_samples, dtype=bool)
    if inner_radius_model is not None:
        keep_mask &= np.linalg.norm(sample_xyz_model, axis=-1) >= inner_radius_model

    query_points = np.concatenate([
        sample_xyz_model,
        np.full((n_samples, 1), loader.normalize_datetime(time), dtype=np.float32),
    ], axis=-1).astype(np.float32)
    model_out = loader.load_coords(query_points, batch_size=batch_size, progress=False)
    rho_cm3 = loader.convert_rho(model_out['rho'])[..., 0].astype(np.float32)
    rho_cm3[~keep_mask] = np.nan

    dz_model = float(z_model[1] - z_model[0])
    ds_cm = np.full(n_samples, dz_model * float(loader.Rs_per_ds) * R_SUN_CM, dtype=np.float64)
    ds_cm[[0, -1]] *= 0.5
    column_density_cm2 = np.nansum(rho_cm3 * ds_cm)

    return {
        'rho_cm3': rho_cm3,
        'distance_from_observer_m': z_model * loader.Rs_per_ds * R_SUN_M,
        'ds_m': ds_cm * 1e-2,
        'sample_hci_xyz_m': sample_hci_xyz_m,
        'sample_hci_radius_m_lat_lon_deg': np.stack([
            heliocentric_radius_m,
            sample_hci_lat_deg,
            sample_hci_lon_deg,
        ], axis=-1).astype(np.float64),
        'ray_origin_hci_xyz_m': ray_origin_model * loader.Rs_per_ds * R_SUN_M,
        'ray_direction': ray_direction,
        'heliocentric_radius_m': heliocentric_radius_m,
        'column_density_cm2': column_density_cm2,
        'profile_radius_min_rsun': np.nan if radius_min is None else radius_min.to_value(u.R_sun),
        'profile_radius_max_rsun': radius_max.to_value(u.R_sun),
    }


def render_profile(loader, observer_coord, time, hpc_tx, hpc_ty, resolution, occ_min, occ_max,
                   profile_radius_min, profile_radius_max, profile_samples, batch_size):
    scale = _get_scale_from_occ_max(occ_max, observer_coord.distance, resolution)
    ref_map = build_sun_centered_reference_map(observer_coord, time, resolution, scale, occ_min, occ_max)
    target_coord = SkyCoord(hpc_tx, hpc_ty, obstime=time, observer=observer_coord, frame=frames.Helioprojective)
    img_coords = all_coordinates_from_map(ref_map)
    img_coords = np.stack([img_coords.Tx, img_coords.Ty], -1)
    img_coords[np.isnan(ref_map.data)] = np.nan

    target_pose = pose_spherical(
        observer_coord.lon.to_value(u.rad),
        observer_coord.lat.to_value(u.rad),
        observer_coord.distance.to_value(u.solRad) / loader.Rs_per_ds,
    )

    output = loader.load_pose(
        img_coords,
        target_pose,
        time,
        batch_size=batch_size,
        progress=False,
        model_outputs=None,
    )

    tB_map = Map(output['image'][..., 0], ref_map.meta)
    pB_map = Map(output['image'][..., 1], ref_map.meta)
    # The renderer already integrated rho with composite-trapezoidal node widths;
    # ThomsonSuNeRFLoader converts that model integral to electrons cm^-2.
    column_density_map = Map(output['density'], ref_map.meta)

    target_x, target_y = ref_map.world_to_pixel(target_coord)
    x_pix = target_x.to_value(u.pix)
    y_pix = target_y.to_value(u.pix)
    if not (0 <= x_pix < output['density'].shape[1] and 0 <= y_pix < output['density'].shape[0]):
        raise ValueError(
            f'HPC [{hpc_tx.to_value(u.arcsec)}, {hpc_ty.to_value(u.arcsec)}] arcsec is outside the rendered FOV. '
            'Increase --occ_max or --resolution/FOV settings.'
        )
    x_idx = int(np.clip(np.rint(x_pix), 0, output['density'].shape[1] - 1))
    y_idx = int(np.clip(np.rint(y_pix), 0, output['density'].shape[0] - 1))
    ray_o, ray_d = get_rays(
        img_coords[y_idx:y_idx + 1, x_idx:x_idx + 1, 0],
        img_coords[y_idx:y_idx + 1, x_idx:x_idx + 1, 1],
        target_pose,
    )
    ray_origin_model = ray_o[0, 0]
    ray_direction = ray_d[0, 0]
    profile = sample_equal_spaced_los_profile(
        loader,
        time,
        ray_origin_model,
        ray_direction,
        profile_radius_min,
        profile_radius_max,
        profile_samples,
        batch_size,
    )
    profile.update({
        'tB_msb': output['image'][y_idx, x_idx, 0],
        'pB_msb': output['image'][y_idx, x_idx, 1],
        'pixel_xy': np.array([x_idx, y_idx], dtype=np.int64),
    })

    return {
        'tB_map': tB_map,
        'pB_map': pB_map,
        'column_density_map': column_density_map,
        'profile': profile,
    }


def plot_map(ax, s_map, target_coords, labels, colors, title, cmap, norm):
    im = ax.imshow(s_map.data, cmap=cmap, norm=norm, origin='lower')
    for label, target_coord, color in zip(labels, target_coords, colors):
        ax.plot_coord(target_coord, marker='x', color=color, markersize=8, markeredgewidth=1.8)
        px, py = s_map.world_to_pixel(target_coord)
        ax.text(px.to_value(u.pix) + 3, py.to_value(u.pix) + 3, label, color=color, fontsize=8)
    ax.set_title(title)
    s_map.draw_grid(ax, color='blue')
    return im


def add_colorbar(fig, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05, axes_class=plt.Axes)
    fig.colorbar(im, cax=cax)


def save_render_figure(rendered, profiles, out_path, points, time, observer_coord, dpi):
    labels = [label for label, _, _ in points]
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [color_cycle[i % len(color_cycle)] for i in range(len(points))]
    target_coords = [
        SkyCoord(tx * u.arcsec, ty * u.arcsec, frame=frames.Helioprojective, obstime=time, observer=observer_coord)
        for _, tx, ty in points
    ]
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.8])
    ax_pb = fig.add_subplot(grid[0, 0], projection=rendered['pB_map'])
    ax_density = fig.add_subplot(grid[0, 1], projection=rendered['column_density_map'])
    ax_profile = fig.add_subplot(grid[1, :])

    im = plot_map(ax_pb, rendered['pB_map'], target_coords, labels, colors,
                  'Forward pB [MSB]', cm.soholasco2, LogNorm())
    add_colorbar(fig, ax_pb, im)
    im = plot_map(ax_density, rendered['column_density_map'], target_coords, labels, colors,
                  r'Column density [cm$^{-2}$]', 'RdPu', LogNorm())
    add_colorbar(fig, ax_density, im)

    for label, profile, color in zip(labels, profiles, colors):
        finite = np.isfinite(profile['rho_cm3']) & (profile['rho_cm3'] > 0)
        ax_profile.plot(
            profile['distance_from_observer_m'][finite],
            profile['rho_cm3'][finite],
            linewidth=1.5,
            label=label,
            color=color,
        )
    ax_profile.set_yscale('log')
    ax_profile.set_xlabel('Observer distance along LOS [m]')
    ax_profile.set_ylabel(r'$n_e$ [cm$^{-3}$]')
    ax_profile.set_title('LOS Density Profile')
    ax_profile.legend()
    ax_profile.grid(alpha=0.25)

    fig.suptitle(f'{time.isoformat(" ", timespec="minutes")}', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_path, 'render_with_profile.png'), dpi=dpi)
    plt.close(fig)


def save_npz(rendered, profiles, args, observer_coord, out_path, points):
    labels = [label for label, _, _ in points]
    hpc = [[tx, ty] for _, tx, ty in points]
    np.savez_compressed(
        os.path.join(out_path, 'density_los_profile.npz'),
        point_label=np.array(labels),
        sunerf_path=np.array(args.sunerf_path),
        time_iso=np.array(args.time.isoformat()),
        map_center_hpc_arcsec=np.array([0.0, 0.0], dtype=np.float32),
        hpc_arcsec=np.array(hpc, dtype=np.float32),
        observer=np.array(args.observer),
        observer_hci_lat_lon_distance_m=np.array([
            observer_coord.lat.to_value(u.deg),
            observer_coord.lon.to_value(u.deg),
            observer_coord.distance.to_value(u.m),
        ], dtype=np.float64),
        coordinate_frame=np.array('HeliocentricInertial'),
        profile_sampling=np.array('equally_spaced_los'),
        profile_radius_range_m=np.array([
            np.nan if np.isnan(profiles[0]['profile_radius_min_rsun']) else profiles[0]['profile_radius_min_rsun'] * R_SUN_M,
            profiles[0]['profile_radius_max_rsun'] * R_SUN_M,
        ], dtype=np.float64),
        distance_from_observer_m=np.stack([p['distance_from_observer_m'] for p in profiles]),
        ds_m=np.stack([p['ds_m'] for p in profiles]),
        heliocentric_radius_m=np.stack([p['heliocentric_radius_m'] for p in profiles]),
        electron_density_cm3=np.stack([p['rho_cm3'] for p in profiles]),
        column_density_cm2=np.array([p['column_density_cm2'] for p in profiles]),
        sample_hci_xyz_m=np.stack([p['sample_hci_xyz_m'] for p in profiles]),
        sample_hci_radius_m_lat_lon_deg=np.stack([p['sample_hci_radius_m_lat_lon_deg'] for p in profiles]),
        ray_origin_hci_xyz_m=np.stack([p['ray_origin_hci_xyz_m'] for p in profiles]),
        ray_direction=np.stack([p['ray_direction'] for p in profiles]),
        profile_pixel_xy=np.stack([p['pixel_xy'] for p in profiles]),
        pB_map_msb=rendered['pB_map'].data,
        tB_map_msb=rendered['tB_map'].data,
        column_density_map_cm2=rendered['column_density_map'].data,
    )


def save_content_description(out_path):
    description = """density_los_profile.npz contents

point_label: Tracked point labels. Point-dependent arrays use this leading point dimension.
sunerf_path: SuNeRF model checkpoint path.
time_iso: Evaluation time in ISO format.
map_center_hpc_arcsec: Rendered image center [Tx, Ty] in arcsec; fixed to [0, 0].
hpc_arcsec: Requested helioprojective coordinates [Tx, Ty] in arcsec for each point.
observer: Observer label used for the HPC coordinate.
observer_hci_lat_lon_distance_m: Observer position [lat_deg, lon_deg, distance_m] in HeliocentricInertial.
coordinate_frame: Coordinate frame for exported 3D positions, HeliocentricInertial.
profile_sampling: Sampling method for the 1D profile.
profile_radius_range_m: Heliocentric radial range [min_m, max_m] used for the profile.
distance_from_observer_m: Equally spaced LOS sample distances from the observer in meters.
ds_m: Trapezoidal LOS integration weights in meters.
heliocentric_radius_m: Heliocentric radius of each LOS sample in meters.
electron_density_cm3: Physical electron density at each LOS sample in cm^-3.
column_density_cm2: LOS-integrated electron column density in cm^-2 for each point.
sample_hci_xyz_m: LOS sample positions [x, y, z] in HeliocentricInertial meters.
sample_hci_radius_m_lat_lon_deg: LOS sample positions [radius_m, lat_deg, lon_deg] in HeliocentricInertial.
ray_origin_hci_xyz_m: Ray origin/observer position [x, y, z] in HeliocentricInertial meters.
ray_direction: Unitless LOS direction vector in HeliocentricInertial Cartesian coordinates.
profile_pixel_xy: Image pixel [x, y] corresponding to the requested HPC LOS.
pB_map_msb: Forward-rendered polarized brightness map in mean solar brightness.
tB_map_msb: Forward-rendered total brightness map in mean solar brightness.
column_density_map_cm2: Forward-rendered electron column-density map in cm^-2.
"""
    with open(os.path.join(out_path, 'density_los_profile_contents.txt'), 'w', encoding='utf-8') as f:
        f.write(description)


def main():
    args = build_parser().parse_args()
    if args.out_path is None:
        args.out_path = os.path.join(os.path.dirname(args.sunerf_path), 'density_los_profile')
    os.makedirs(args.out_path, exist_ok=True)

    loader = ThomsonSuNeRFLoader(args.sunerf_path)
    observer_coord = resolve_observer(args)
    resolution = (args.resolution, args.resolution) * u.pix
    occ_min = None if args.occ_min is None else args.occ_min * u.R_sun
    occ_max = None if args.occ_max is None else args.occ_max * u.R_sun
    profile_radius_min = (
        occ_min if args.profile_radius_min is None else args.profile_radius_min * u.R_sun
    )
    profile_radius_max = (
        occ_max if args.profile_radius_max is None else args.profile_radius_max * u.R_sun
    )
    if profile_radius_max is None:
        raise ValueError('--profile_radius_max is required when --occ_max is omitted')
    if profile_radius_min is not None and profile_radius_min >= profile_radius_max:
        raise ValueError('--profile_radius_min must be smaller than --profile_radius_max')

    points = resolve_points(args)
    rendered = None
    profiles = []
    for label, tx, ty in points:
        hpc_tx, hpc_ty = tx * u.arcsec, ty * u.arcsec

        point_rendered = render_profile(
            loader,
            observer_coord,
            args.time,
            hpc_tx,
            hpc_ty,
            resolution,
            occ_min,
            occ_max,
            profile_radius_min,
            profile_radius_max,
            args.profile_samples,
            args.batch_size,
        )
        if rendered is None:
            rendered = point_rendered
        profiles.append(point_rendered['profile'])

    save_render_figure(rendered, profiles, args.out_path, points, args.time, observer_coord, args.dpi)
    save_npz(rendered, profiles, args, observer_coord, args.out_path, points)
    save_content_description(args.out_path)
    print(f'Wrote density LOS profile outputs for {len(points)} point(s) to {args.out_path}')


if __name__ == '__main__':
    main()
