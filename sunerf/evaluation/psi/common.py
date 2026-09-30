"""Shared loading and drawing routines of the GT evaluation plots.

The plots read the SuNeRF exports (``sunerf.evaluation.cme.export_*``), the GT
density cubes and the GT image series; no plot loads a SuNeRF model.
"""
from dataclasses import replace
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.colors import LogNorm
from sunpy.map import Map
from sunpy.visualization.colormaps import cm

from sunerf.data.psi.psi_test_paths import channel_file
from sunerf.evaluation.cme.density_series import load_export_frame, load_psi_frame
from sunerf.evaluation.cme.export_layout import SERIES_FILE, dump_name

TRUTH_TITLE = "Ground truth"
TRUTH_COLOR = "black"
RUN_COLORS = ("C0", "C1", "C2", "C3", "C4", "C5")
PRODUCT_LABELS = {"tb": "tB", "pb": "pB"}
# Stonyhurst longitudes of the viewpoints marked in the equatorial cut.
VIEWPOINTS = {"Earth": 0.0, "L4": 60.0, "L5": -60.0}
RADIUS_RANGE = (2.0, 30.0)
DENSITY_LABEL = r"$n_e\,(r/R_\odot)^2$ [cm$^{-3}$]"
DENSITY_CMAP = plt.get_cmap("plasma").copy()
DENSITY_CMAP.set_bad("white")
FIGURE_WIDTH = 7.2
# The leading edge is the outer end of the outermost enhancement of at least the minimum width [R_sun]
# in which a quantity exceeds its pre-event value by this factor.
LEADING_EDGE_RATIO = 1.1
LEADING_EDGE_MIN_WIDTH = 1.0


##########################################################
# data

def to_psi_order(frame):
    """Density (r, latitude, longitude) as (phi, theta, r), cropped to the evaluated shell."""
    r_idx = np.flatnonzero((frame.radius >= RADIUS_RANGE[0]) & (frame.radius <= RADIUS_RANGE[1]))
    # Latitude ascends in the frames; theta grows southward.
    return (frame.rho[r_idx].transpose(2, 1, 0)[:, ::-1], frame.radius[r_idx],
            np.pi / 2.0 - frame.latitude[::-1], frame.longitude)


def _grid_indices(axis, subset, name):
    index = np.clip(np.searchsorted(axis, subset), 0, axis.size - 1)
    if not np.allclose(axis[index], subset):
        raise ValueError(f"The {name} axis of the export is not part of the GT grid.")
    return index


def load_densities(density_dir: Path, dump: int, runs):
    """GT and SuNeRF densities in cm^-3 on the grid of the exports.

    Returns ``truth, cubes, radius, theta, phi, time`` with arrays in the order
    (phi, theta, r).  Exports written with a stride select the same points of
    the GT cube.
    """
    name = dump_name(dump)
    exports = [load_export_frame(export_dir, name) for _, export_dir in runs]
    grid = exports[0]
    for (label, _), export in zip(runs, exports):
        if any(not np.array_equal(getattr(export, axis), getattr(grid, axis))
               for axis in ("radius", "latitude", "longitude")):
            raise ValueError(f"The grid of the export '{label}' differs from the other exports.")
    truth = load_psi_frame(density_dir / f"{name}.hdf")
    index = [_grid_indices(getattr(truth, axis), getattr(grid, axis), axis)
             for axis in ("radius", "latitude", "longitude")]
    truth_cube, radius, theta, phi = to_psi_order(replace(grid, rho=truth.rho[np.ix_(*index)]))
    return truth_cube, [to_psi_order(export)[0] for export in exports], radius, theta, phi, grid.time


def load_series(path: Path):
    """CME diagnostics written by ``export_cme_series`` (file or export directory)."""
    path = Path(path)
    series = dict(np.load(path / SERIES_FILE if path.is_dir() else path))
    series["times"] = series["times"].astype("datetime64[s]").astype(object)
    return series


def outermost_above(profile, heights):
    """Outer end of the outermost enhancement above the leading-edge ratio.

    Enhancements narrower than the minimum width are ignored.  Also returns
    whether the end lies at the outer limit of the profile.
    """
    valid = np.flatnonzero(np.isfinite(profile))
    above = np.nan_to_num(profile) >= LEADING_EDGE_RATIO
    limits = np.flatnonzero(np.diff(np.concatenate([[0], above.astype(int), [0]])))
    for start, end in zip(limits[::2][::-1], limits[1::2][::-1] - 1):
        if heights[end] - heights[start] < LEADING_EDGE_MIN_WIDTH:
            continue
        if end == valid[-1] or not np.isfinite(profile[end + 1]):
            return heights[end], end == valid[-1]
        # position of the threshold between the two heights that enclose it
        fraction = (profile[end] - LEADING_EDGE_RATIO) / (profile[end] - profile[end + 1])
        return heights[end] + fraction * (heights[end + 1] - heights[end]), False
    return np.nan, False


def leading_edge(ratio, heights):
    """Leading edge for every time of ``ratio[time, height]``, the ratio to the pre-event frame.

    The track ends when the edge reaches the outer limit of the field of view.
    """
    edge = np.full(ratio.shape[0], np.nan)
    for index, profile in enumerate(ratio):
        height, at_limit = outermost_above(profile, heights)
        if at_limit:
            if np.isfinite(edge).any():
                break
            continue
        edge[index] = height
    return edge


def series_leading_edge(entry, key):
    """Leading edge of ``entry[key][time, radius]`` relative to the first frame of the series."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return leading_edge(entry[key] / entry[key][0], entry["radius"])


def load_maps(root: Path, views, frames):
    return {view: {product: Map(channel_file(root, view, product, f"{frames[view]:04d}"))
                   for product in PRODUCT_LABELS} for view in views}


def cme_longitude_index(truth, radius, theta) -> int:
    """Longitude whose density exceeds the longitudinal median the most."""
    excess = np.log(truth / np.median(truth, axis=0, keepdims=True))
    latitude_band = np.abs(np.pi / 2.0 - theta) < np.deg2rad(60.0)
    return int(np.argmax(excess[:, latitude_band][:, :, radius >= 5.0].mean(axis=(1, 2))))


##########################################################
# figure elements

def density_norm(values) -> LogNorm:
    positive = values[np.isfinite(values) & (values > 0)]
    return LogNorm(*np.percentile(positive, (1.0, 99.5)))


def image_norm(images) -> LogNorm:
    # The brightness falls by orders of magnitude over the coronagraph fields.
    finite = np.concatenate([image[np.isfinite(image) & (image > 0)] for image in images])
    return LogNorm(*np.percentile(finite, (2.0, 99.8)), clip=True)


def add_panel_metric(axis, text, color="white") -> None:
    axis.text(0.04, 0.96, text, transform=axis.transAxes, va="top", ha="left", fontsize=7, color=color)


def add_colorbar(fig, artist, slot, label) -> None:
    colorbar = fig.colorbar(artist, cax=fig.add_subplot(slot))
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=6.5)


def map_extent(s_map) -> list[float]:
    """Tangent-plane extent of a Sun-centred map in solar radii."""
    half = [0.5 * n * float(scale.to_value(u.rad / u.pix)) * float(s_map.dsun.to_value(u.R_sun))
            for n, scale in zip(s_map.data.shape[::-1], s_map.scale)]
    return [-half[0], half[0], -half[1], half[1]]


def add_sun(axis) -> None:
    """Solar disk at the image center as the size reference."""
    axis.add_patch(plt.Circle((0.0, 0.0), 1.0, color="yellow", linewidth=0))


def plot_image(axis, image, norm, extent):
    artist = axis.imshow(image, origin="lower", cmap=cm.soholasco2, norm=norm, interpolation="none",
                         extent=extent)
    axis.set_facecolor("black")
    axis.tick_params(labelsize=6.5)
    return artist


def label_plane_axes(axis, xlabel, ylabel, show_x=True, show_y=True) -> None:
    if show_x:
        axis.set_xlabel(xlabel, fontsize=7)
    else:
        axis.set_xticklabels([])
    if show_y:
        axis.set_ylabel(ylabel, fontsize=7)
    else:
        axis.set_yticklabels([])


def format_time_axis(axis, max_ticks=6) -> None:
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=max_ticks))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axis.tick_params(labelsize=6.5)


def save_figure(fig, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=200)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    return output_dir / f"{name}.png"


##########################################################
# density cuts

def finish_polar_axis(axis, outer) -> None:
    axis.set_rlim(0, outer)
    axis.set_rticks([10, 20])
    axis.set_rlabel_position(135)
    axis.set_thetagrids([])
    axis.tick_params(labelsize=6.5)
    axis.grid(color="white", linewidth=0.4, alpha=0.5)
    axis.set_facecolor("white")


def plot_equatorial(axis, cube, radius, theta, phi, earth_longitude, norm, meridional_phi, label_viewpoints):
    """Polar plot of the equatorial cut with Earth at the bottom and L4 on the right."""
    equator = int(np.argmin(np.abs(theta - np.pi / 2.0)))
    # Close the longitude seam.
    values = np.concatenate([cube[:, equator], cube[:1, equator]]) * radius ** 2
    longitude = np.concatenate([phi, phi[:1] + 2.0 * np.pi]) - earth_longitude
    lon_grid, r_grid = np.meshgrid(longitude, radius, indexing="ij")
    artist = axis.pcolormesh(lon_grid, r_grid, values, norm=norm, cmap=DENSITY_CMAP,
                             shading="gouraud", rasterized=True)
    axis.set_theta_zero_location("S")
    axis.set_theta_direction(1)
    outer = radius[-1]
    for name, stonyhurst in VIEWPOINTS.items():
        angle = np.deg2rad(stonyhurst)
        axis.plot([angle, angle], [0.93 * outer, 1.0 * outer], color="white", linewidth=1.0)
        if not label_viewpoints:
            continue
        offset = 9.0 * np.array([np.sin(angle), -np.cos(angle)])
        axis.annotate(name, (angle, outer), xytext=offset, textcoords="offset points",
                      ha="center", va="center", fontsize=6.5, annotation_clip=False)
    cut = meridional_phi - earth_longitude
    axis.plot([cut, cut], [0, outer], color="white", linewidth=0.6, linestyle=":")
    axis.plot([cut + np.pi, cut + np.pi], [0, outer], color="white", linewidth=0.6, linestyle=":")
    finish_polar_axis(axis, outer)
    return artist


def plot_meridional(axis, cube, radius, theta, index, norm):
    """Polar plot of the meridional cut: the CME longitude on the right, its antipode on the left."""
    opposite = (index + cube.shape[0] // 2) % cube.shape[0]
    theta_grid, r_grid = np.meshgrid(theta, radius, indexing="ij")
    artist = None
    for sign, slab in ((1.0, index), (-1.0, opposite)):
        artist = axis.pcolormesh(sign * theta_grid, r_grid, cube[slab] * radius ** 2, norm=norm,
                                 cmap=DENSITY_CMAP, shading="gouraud", rasterized=True)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    finish_polar_axis(axis, radius[-1])
    return artist


def plot_carrington(axis, cube, radius, theta, phi, map_radius, norm):
    shell = int(np.argmin(np.abs(radius - map_radius)))
    extent = [0.0, 360.0, -90.0, 90.0]
    # theta grows southward; flip so that latitude increases upward.
    artist = axis.imshow((cube[:, ::-1, shell] * radius[shell] ** 2).T, origin="lower", extent=extent,
                         aspect="auto", cmap=DENSITY_CMAP, norm=norm, interpolation="none")
    axis.set_xticks([0, 120, 240, 360])
    axis.set_yticks([-90, 0, 90])
    axis.tick_params(labelsize=6.5)
    axis.set_xlabel("Carr. longitude [deg]", fontsize=7)
    axis.set_ylabel("latitude [deg]", fontsize=7)
    return artist
