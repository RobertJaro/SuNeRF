"""Solar Orbiter validation figures of an EUV plasma reconstruction.

The held-out EUI FSI frame sets the view: the model is rendered with the rays of
that frame (its observer, pointing and pixel grid, as in training), and two
figures are written to ``--out_path``:

* ``comparison.png``: the EUI observation; SuNeRF FSI 174 render; the AIA and
  EUVI-A input images reprojected to the Solar Orbiter view (on-disk only, the
  solar surface is the projection screen). The FSI gain is not trained and the
  inputs are other instruments, so the render and the reprojections are
  calibrated to the observation: their mean and standard deviation in log10
  counts are matched to the EUI frame over the pixels both cover (as the
  ``normalize`` option of the euv_tomography callback), and all four panels share
  the EUI color map and scale. A Carrington grid (every GRID_SPACING_DEG) is
  drawn on the full-disk panels; the second row shows the north polar subframe
  (POLAR_XLIM, POLAR_YLIM), marked by a red box in the first row.
* ``thermal.png``: the AIA 193 render from Solar Orbiter with the Stonyhurst
  +-60 deg region (black) and the 0 deg Stonyhurst meridian (red); the radially
  integrated emission measure per temperature bin and (with a cool absorber)
  the radially integrated cool hydrogen column over that region in
  Stonyhurst longitude and latitude; second row: temperature, electron density
  and (with a cool absorber) cool hydrogen density in the meridional plane of
  the 0 deg Stonyhurst meridian (radius vs latitude).

Example::

    python -m sunerf.evaluation.plasma.solo_comparison --sunerf_path /path/save_state.snf \
        --eui_file eui.prepared.fits --aia_file aia_171.prepared.fits --euvi_file euvi_171.prepared.fits
"""
import argparse
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from sunpy.coordinates import frames
from sunpy.map import Map

from sunerf.data.euv.observation import load_prepared_observation
from sunerf.data.utils import get_azimuthal_equidistant_coordinates
from sunerf.evaluation.loader import PlasmaSuNeRFLoader
from sunerf.evaluation.plasma.radial import integrate_columns, radial_columns, rendered_shell
from sunerf.evaluation.plasma.video import (
    CHANNEL_PERCENTILES,
    EM_CMAP,
    EM_PERCENTILES,
    channel_norm,
    frame_range,
    resolve_channel,
    resolve_instrument,
    set_channel_colorbar_ticks,
)
from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.euv_callback import _match_mean_and_std

# rendered channels: held-out FSI channel and the thermal-structure channel
EUI_CHANNEL = ('EUI', '174')
THERMAL_CHANNEL = ('AIA', '193')
# observer labels of the input files
INPUT_OBSERVERS = {'AIA': 'SDO', 'EUVI-A': 'STEREO-A'}
# Stonyhurst region of the emission measure maps [deg] and the meridian of the slice
REGION_DEG = 60.0
SLICE_LONGITUDE_DEG = 0.0
DENSITY_CMAP = 'viridis'
TEMPERATURE_CMAP = 'plasma'
COOL_CMAP = 'cividis'
SLICE_PERCENTILES = (1.0, 99.0)
# radius [R_sun] at the center of the polar slice plots
SLICE_ORIGIN_RADIUS = 0.6
# Carrington grid of the comparison panels
GRID_SPACING_DEG = 30.0
# north polar subframe of the comparison [R_sun], 2:1 width to height
POLAR_XLIM = (-0.6, 0.6)
POLAR_YLIM = (0.5, 1.1)


def build_parser():
    parser = argparse.ArgumentParser(
        description='Solar Orbiter validation figures: render vs. EUI and reprojected inputs, the input images, '
                    'and the thermal structure seen from Solar Orbiter.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--sunerf_path', type=str, required=True, help='Path to the plasma save state')
    parser.add_argument('--eui_file', type=str, required=True,
                        help='Prepared EUI FSI 174 file; its observer, time and pixel grid set the view')
    parser.add_argument('--aia_file', type=str, required=True, help='Prepared AIA input file (SDO)')
    parser.add_argument('--euvi_file', type=str, required=True, help='Prepared EUVI-A input file (STEREO-A)')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output directory; default: <run directory>/solo_comparison')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Resample the EUI view to this many pixels per side; also the grid size of the '
                             'emission measure maps and the latitude samples of the slice; default: the EUI '
                             'pixel grid')
    parser.add_argument('--em_edges', type=float, nargs='+', default=[5.7, 6.2, 7.0],
                        help='Temperature bin edges of the emission measure maps in log10 K')
    parser.add_argument('--n_radius', type=int, default=256,
                        help='Radial samples of the emission measure integration and of the slice')
    parser.add_argument('--slice_max_radius', type=float, default=1.4,
                        help='Outer radius of the slice in solar radii (the inner radius is the rendered shell)')
    parser.add_argument('--batch_size', type=int, default=4096, help='Rays or points per model batch')
    parser.add_argument('--dpi', type=int, default=300, help='Saved figure DPI')
    return parser


################################################################################
# Observations and rendering
################################################################################

def load_observation(path):
    """Prepared map with the invalid pixels as NaN, rotated north up if needed."""
    s_map, valid = load_prepared_observation(path)
    data = np.where(valid, s_map.data, np.nan).astype(np.float32)
    s_map = Map(data, s_map.meta)
    if not np.allclose(s_map.rotation_matrix, np.eye(2)):
        s_map = s_map.rotate(missing=np.nan)
    return s_map


def render_view(loader, s_map, instrument_key, channel_id, batch_size):
    """Rendered channel on the rays of ``s_map`` (observer and pixel grid of the frame, as in training)."""
    pose = pose_spherical(s_map.carrington_longitude.to_value(u.rad), s_map.carrington_latitude.to_value(u.rad),
                          s_map.dsun.to_value(u.solRad) / float(loader.Rs_per_ds))
    out = loader.load_pose(get_azimuthal_equidistant_coordinates(s_map), pose, s_map.date.datetime,
                           batch_size=batch_size, instrument_key=instrument_key, progress=True,
                           model_outputs=['image'])
    channel_ids = [c['id'] for c in loader.channel_metadata(instrument_key)]
    return out['image'][..., channel_ids.index(channel_id)]


def calibrate_to_observation(rendered, observed):
    """Render with the mean and standard deviation of the observation in log10 counts (over the pixels where
    both are positive); the log image scaling of training is affine in log10, so this matches the loss space."""
    with np.errstate(invalid='ignore', divide='ignore'):
        log_rendered = np.log10(np.where(rendered > 0, rendered, np.nan))
        log_observed = np.log10(np.where(observed > 0, observed, np.nan))
    return 10 ** _match_mean_and_std(log_rendered, log_observed)


def reproject(s_map, target_map):
    """``s_map`` seen from the observer of ``target_map``; off-disk pixels are NaN."""
    return s_map.reproject_to(target_map.wcs).data


################################################################################
# Geometry on the solar surface
################################################################################

def surface_coord(lat_deg, lon_deg, obstime):
    """Stonyhurst points on the photosphere."""
    return SkyCoord(lon=np.asarray(lon_deg) * u.deg, lat=np.asarray(lat_deg) * u.deg, radius=1 * u.R_sun,
                    frame=frames.HeliographicStonyhurst(obstime=obstime))


def visible_pixels(coord, s_map):
    """Pixel coordinates of photospheric points in ``s_map``; NaN on the far side, so lines break there."""
    observer = s_map.observer_coordinate.transform_to(coord.frame).cartesian.xyz.to_value(u.R_sun)
    points = coord.cartesian.xyz.to_value(u.R_sun)
    visible = np.tensordot(observer, points, axes=(0, 0)) > 1.0
    x, y = s_map.world_to_pixel(coord)
    x = np.where(visible, x.to_value(u.pix), np.nan)
    y = np.where(visible, y.to_value(u.pix), np.nan)
    return x, y


class ImageFrame:
    """Plot coordinates of a map in projected solar radii from Sun center (tangent-plane offsets times the
    observer distance)."""

    def __init__(self, s_map):
        self.per_pixel = [float(s.to_value(u.rad / u.pix)) * s_map.dsun.to_value(u.R_sun) for s in s_map.scale]
        center = s_map.world_to_pixel(SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=s_map.coordinate_frame))
        self.center = [float(center.x.to_value(u.pix)), float(center.y.to_value(u.pix))]
        ny, nx = s_map.data.shape
        self.extent = [*self.to_plot(np.array([-0.5, nx - 0.5]), 0), *self.to_plot(np.array([-0.5, ny - 0.5]), 1)]

    def to_plot(self, pixel, axis):
        return (np.asarray(pixel) - self.center[axis]) * self.per_pixel[axis]

    def plot_line(self, ax, x, y, **kwargs):
        return ax.plot(self.to_plot(x, 0), self.to_plot(y, 1), **kwargs)


################################################################################
# Model queries in Stonyhurst coordinates
################################################################################

def to_carrington(lat_deg, lon_deg, observer):
    """Carrington latitude and longitude [rad] of Stonyhurst surface points (``lat_deg``, ``lon_deg``)."""
    carrington = surface_coord(lat_deg, lon_deg, observer.obstime).transform_to(
        frames.HeliographicCarrington(observer=observer, obstime=observer.obstime))
    return carrington.lat.to_value(u.rad), carrington.lon.to_value(u.rad)


def stonyhurst_em_maps(loader, observer, time, n, edges, n_radius, batch_size):
    """Radially integrated emission measure [cm^-5] per temperature bin and cool-absorber hydrogen column
    [cm^-2] (None without a cool absorber) on a Stonyhurst latitude-longitude grid of +-REGION_DEG with
    n x n cells."""
    centers = np.linspace(-REGION_DEG, REGION_DEG, 2 * n + 1)[1::2]
    lon, lat = np.meshgrid(centers, centers)
    lat_rad, lon_rad = to_carrington(lat, lon, observer)
    radii = np.linspace(*rendered_shell(loader), n_radius)
    bins = list(zip(edges[:-1], edges[1:]))
    maps, _, cool = integrate_columns(loader, radial_columns(lat_rad, lon_rad, radii), radii, time, bins,
                                      batch_size, cool_column=True)
    return bins, maps, cool


def meridional_slice(loader, observer, time, n_lat, radius_range, n_radius, batch_size):
    """Electron density [cm^-3], log10 temperature [K] and cool-absorber hydrogen density [cm^-3] (None
    without a cool absorber) in the half-plane of the Stonyhurst meridian SLICE_LONGITUDE_DEG, shape
    (n_lat, n_radius), and the latitude [deg] and radius [R_sun] samples."""
    latitudes = np.linspace(-90.0, 90.0, n_lat)
    radii = np.linspace(radius_range[0], radius_range[1], n_radius)
    lat_rad, lon_rad = to_carrington(latitudes, np.full(n_lat, SLICE_LONGITUDE_DEG), observer)
    points = radial_columns(lat_rad, lon_rad, radii) / float(loader.Rs_per_ds)
    coords = np.concatenate([points, np.full(points.shape[:-1] + (1,), loader.normalize_datetime(time))], -1)
    out = loader.load_coords(coords.astype(np.float32), batch_size=batch_size)
    cool = np.squeeze(out['cool_hydrogen_density'], -1) if 'cool_hydrogen_density' in out else None
    return np.squeeze(out['total_ne'], -1), np.squeeze(out['mean_log_T'], -1), cool, latitudes, radii


################################################################################
# Figures
################################################################################

def image_panel(fig, ax, frame, data, cmap, norm, title, channel_ticks=False, ylabel=True):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('white')
    im = ax.imshow(data, origin='lower', extent=frame.extent, cmap=cmap, norm=norm)
    ax.set_aspect('equal')
    ax.set_xlabel(r'HPC-x [R$_\odot$]')
    if ylabel:
        ax.set_ylabel(r'HPC-y [R$_\odot$]')
    else:
        ax.tick_params(labelleft=False)
    if title is None:
        return im
    colorbar = fig.colorbar(im, ax=ax, location='top', aspect=30, pad=0.02)
    colorbar.set_label(title)
    if channel_ticks:
        set_channel_colorbar_ticks(colorbar, norm)
    return im


def carrington_grid(ax, frame, s_map):
    """Carrington latitude and longitude lines on the visible photosphere of ``s_map``."""
    carrington = frames.HeliographicCarrington(observer=s_map.observer_coordinate, obstime=s_map.date)
    samples = np.linspace(-180.0, 180.0, 721)
    lines = [(np.full_like(samples, lat), samples)
             for lat in np.arange(-90.0 + GRID_SPACING_DEG, 90.0, GRID_SPACING_DEG)]
    lines += [(samples / 2, np.full_like(samples, lon)) for lon in np.arange(0.0, 360.0, GRID_SPACING_DEG)]
    for lat, lon in lines:
        coord = SkyCoord(lon=lon * u.deg, lat=lat * u.deg, radius=1 * u.R_sun, frame=carrington)
        frame.plot_line(ax, *visible_pixels(coord, s_map), color='black', alpha=0.4, linewidth=0.5)


def unit_label(s_map):
    unit = s_map.meta.get('bunit')
    return f' [{unit}]' if unit else ''


def plot_comparison(path, eui_map, rendered, inputs, reprojected, dpi):
    """EUI observation, SuNeRF render calibrated to it, reprojected inputs."""
    frame = ImageFrame(eui_map)
    norm = channel_norm(0.0, frame_range(eui_map.data, CHANNEL_PERCENTILES)[1])
    time = eui_map.date.datetime.isoformat(' ', timespec='minutes')

    panels = [(eui_map.data, f'EUI FSI 174 Å\nheld out{unit_label(eui_map)}'),
              (rendered, f'SuNeRF FSI 174 Å\ncalibrated to EUI{unit_label(eui_map)}')]
    panels += [(reprojected[key], f'{INPUT_OBSERVERS[key]} {s_map.wavelength.to_value(u.AA):g} Å reprojected\n'
                                  f'calibrated to EUI{unit_label(eui_map)}') for key, s_map in inputs.items()]
    polar_ratio = (POLAR_YLIM[1] - POLAR_YLIM[0]) / (POLAR_XLIM[1] - POLAR_XLIM[0])
    fig, axes = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 4.0 + 3.0 * polar_ratio),
                             constrained_layout=True, sharex='row', sharey='row',
                             gridspec_kw={'height_ratios': [1.0, polar_ratio]})
    for i, (data, title) in enumerate(panels):
        image_panel(fig, axes[0, i], frame, data, eui_map.cmap, norm, title, channel_ticks=True, ylabel=i == 0)
        image_panel(fig, axes[1, i], frame, data, eui_map.cmap, norm, None, ylabel=i == 0)
    for ax in axes[0]:
        carrington_grid(ax, frame, eui_map)
        ax.add_patch(Rectangle((POLAR_XLIM[0], POLAR_YLIM[0]), POLAR_XLIM[1] - POLAR_XLIM[0],
                               POLAR_YLIM[1] - POLAR_YLIM[0], fill=False, edgecolor='red', linewidth=1.2))
    axes[0, 0].set_xlim(frame.extent[:2])
    axes[0, 0].set_ylim(frame.extent[2:])
    axes[1, 0].set_xlim(POLAR_XLIM)
    axes[1, 0].set_ylim(POLAR_YLIM)
    fig.suptitle(f'Solar Orbiter view, {time} '
                 f'(lat {eui_map.heliographic_latitude.to_value(u.deg):.1f}°, '
                 f'{eui_map.dsun.to_value(u.AU):.2f} au)')
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def polar_slice_panel(fig, ax, latitudes, radii, data, cmap, norm, label):
    """Half-plane of the meridian: polar angle is latitude (north up), radius is the heliocentric distance.
    The plot center is at SLICE_ORIGIN_RADIUS instead of 0, which widens the atmosphere above the surface."""
    theta = np.deg2rad(latitudes)
    im = ax.pcolormesh(theta, radii, data.T, cmap=cmap, norm=norm, shading='nearest', rasterized=True)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_rorigin(SLICE_ORIGIN_RADIUS)
    ax.set_rlim(1.0, radii[-1])
    # the solar surface of the slice is the red meridian of the disk and map panels
    ax.plot(np.linspace(-np.pi / 2, np.pi / 2, 181), np.ones(181), color='red', linewidth=2.0, clip_on=False,
            zorder=5)
    ax.set_rticks(np.arange(1.0, radii[-1] + 1e-6, 0.1))
    ax.set_rlabel_position(-90)
    ax.set_thetagrids([-90, -60, -30, 0, 30, 60, 90])
    ax.tick_params(labelsize=9)
    ax.grid(color='white', alpha=0.3, linewidth=0.5)
    colorbar = fig.colorbar(im, ax=ax, location='top', aspect=12, pad=0.04)
    colorbar.set_label(label)
    colorbar.ax.xaxis.set_minor_formatter(NullFormatter())


def plot_thermal(path, eui_map, rendered, em, em_bins, cool_column, slice_data, dpi):
    """193 render from Solar Orbiter with the Stonyhurst region and meridian, emission measure maps and cool
    hydrogen column map; slices in the second row."""
    frame = ImageFrame(eui_map)
    obstime = eui_map.date
    n_e, log_T, cool, latitudes, radii = slice_data
    maps = [(em_map, EM_CMAP, LogNorm(*frame_range(em_map, EM_PERCENTILES)),
             rf'EM, log$_{{10}}$T in [{lo:.2f}, {hi:.2f}) [cm$^{{-5}}$]') for (lo, hi), em_map in zip(em_bins, em)]
    if cool_column is not None:
        maps.append((cool_column, COOL_CMAP, LogNorm(*frame_range(cool_column, EM_PERCENTILES)),
                     r'cool $N_H$ [cm$^{-2}$]'))
    n_columns = max(1 + len(maps), 2 if cool is None else 3)
    fig = plt.figure(figsize=(3.0 * n_columns, 2 * 4.2), constrained_layout=True)
    grid = GridSpec(2, n_columns, figure=fig)

    ax = fig.add_subplot(grid[0, 0])
    norm = channel_norm(0.0, frame_range(rendered, CHANNEL_PERCENTILES)[1])
    image_panel(fig, ax, frame, rendered, 'sdoaia193', norm, 'SuNeRF AIA 193 Å from Solar Orbiter [DN/s]',
                channel_ticks=True)
    edge = np.linspace(-REGION_DEG, REGION_DEG, 241)
    for lat, lon in [(np.full_like(edge, -REGION_DEG), edge), (np.full_like(edge, REGION_DEG), edge),
                     (edge, np.full_like(edge, -REGION_DEG)), (edge, np.full_like(edge, REGION_DEG))]:
        frame.plot_line(ax, *visible_pixels(surface_coord(lat, lon, obstime), eui_map), color='black', linewidth=1.0)
    meridian = np.linspace(-90, 90, 361)
    frame.plot_line(ax, *visible_pixels(surface_coord(meridian, np.full_like(meridian, SLICE_LONGITUDE_DEG),
                                                      obstime), eui_map), color='red', linewidth=1.2)
    ax.set_xlim(frame.extent[:2])
    ax.set_ylim(frame.extent[2:])

    first_em_ax = None
    for i, (data, cmap, norm, label) in enumerate(maps):
        ax = fig.add_subplot(grid[0, 1 + i], sharex=first_em_ax, sharey=first_em_ax)
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad('white')
        im = ax.imshow(data, origin='lower', extent=[-REGION_DEG, REGION_DEG, -REGION_DEG, REGION_DEG],
                       cmap=cmap, norm=norm)
        ax.axvline(SLICE_LONGITUDE_DEG, color='red', linewidth=1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('Stonyhurst longitude [deg]')
        if first_em_ax is None:
            first_em_ax = ax
            ax.set_ylabel('Stonyhurst latitude [deg]')
        else:
            ax.tick_params(labelleft=False)
        colorbar = fig.colorbar(im, ax=ax, location='top', aspect=30, pad=0.02)
        colorbar.set_label(label)
        colorbar.ax.xaxis.set_minor_formatter(NullFormatter())

    finite_T = log_T[np.isfinite(log_T)]
    temperature_norm = Normalize(*np.percentile(finite_T, SLICE_PERCENTILES))
    polar_slice_panel(fig, fig.add_subplot(grid[1, 0], projection='polar'), latitudes, radii, log_T,
                      TEMPERATURE_CMAP, temperature_norm, r'log$_{10}$(T / K)')
    density_norm = LogNorm(*frame_range(n_e, SLICE_PERCENTILES))
    polar_slice_panel(fig, fig.add_subplot(grid[1, 1], projection='polar'), latitudes, radii, n_e,
                      DENSITY_CMAP, density_norm, r'$n_e$ [cm$^{-3}$]')
    if cool is not None:
        polar_slice_panel(fig, fig.add_subplot(grid[1, 2], projection='polar'), latitudes, radii, cool,
                          COOL_CMAP, LogNorm(*frame_range(cool, SLICE_PERCENTILES)),
                          r'cool $n_H$ [cm$^{-3}$]')
    fig.suptitle(f'Thermal structure, {obstime.datetime.isoformat(" ", timespec="minutes")}; slice: Stonyhurst '
                 f'longitude {SLICE_LONGITUDE_DEG:g}° (red), maps: ±{REGION_DEG:g}° (black)')
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if len(args.em_edges) < 2:
        raise ValueError('--em_edges needs at least two edges')
    out_path = args.out_path or os.path.join(os.path.dirname(args.sunerf_path), 'solo_comparison')
    os.makedirs(out_path, exist_ok=True)

    loader = PlasmaSuNeRFLoader(args.sunerf_path)
    eui_map = load_observation(args.eui_file)
    if args.resolution is not None:
        eui_map = eui_map.resample([args.resolution, args.resolution] * u.pix)
    n = eui_map.data.shape[0]
    inputs = {'AIA': load_observation(args.aia_file), 'EUVI-A': load_observation(args.euvi_file)}
    observed = np.isfinite(eui_map.data)

    eui_key = resolve_instrument(loader, EUI_CHANNEL[0])
    eui_channel = resolve_channel(loader, eui_key, EUI_CHANNEL[1])['id']
    rendered = render_view(loader, eui_map, eui_key, eui_channel, args.batch_size)
    rendered = np.where(observed, rendered, np.nan)
    both = observed & (rendered > 0) & (eui_map.data > 0)
    print(f'Median render / observation on the valid EUI pixels before calibration: '
          f'{np.median(rendered[both] / eui_map.data[both]):.3f}')
    rendered = calibrate_to_observation(rendered, eui_map.data)
    # on-disk pixels only: each reprojection is matched to the EUI frame over the disk it covers
    reprojected = {key: calibrate_to_observation(reproject(s_map, eui_map), eui_map.data)
                   for key, s_map in inputs.items()}
    plot_comparison(os.path.join(out_path, 'comparison.png'), eui_map, rendered, inputs, reprojected, args.dpi)

    thermal_key = resolve_instrument(loader, THERMAL_CHANNEL[0])
    thermal_channel = resolve_channel(loader, thermal_key, THERMAL_CHANNEL[1])['id']
    rendered_193 = np.where(observed, render_view(loader, eui_map, thermal_key, thermal_channel, args.batch_size),
                            np.nan)
    time = eui_map.date.datetime
    observer = eui_map.observer_coordinate
    em_bins, em, cool_column = stonyhurst_em_maps(loader, observer, time, n, args.em_edges, args.n_radius, args.batch_size)
    slice_data = meridional_slice(loader, observer, time, n, (rendered_shell(loader)[0], args.slice_max_radius),
                                  args.n_radius, args.batch_size)
    plot_thermal(os.path.join(out_path, 'thermal.png'), eui_map, rendered_193, em, em_bins, cool_column,
                 slice_data, args.dpi)
    print(f'Wrote comparison.png and thermal.png to {out_path}')


if __name__ == '__main__':
    main()
