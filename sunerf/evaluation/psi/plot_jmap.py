#!/usr/bin/env python3
"""Plot 7: J-maps of the CME from the synthetic observations and from the 3D density.

Top row, naive construction: for every view the total brightness along the
position angle of the CME direction is stacked over time as the ratio to the
first frame of the CME period; the height is the plane-of-sky distance.  The
leading edge (outermost height above the ratio threshold) gives the speed an
observer would derive and is shown in red.  The ground-truth body edge is
shown in cyan at its true radial distance only.

Bottom row, 3D construction: compensated density ``r^2 n_e`` along the CME
direction over time for GT and every run; the height is the radial distance.
The tracks use ``body_edge_apex`` from the exported series, the same dense-body
edge used in the mass propagation plots (90th percentile over the cone rays).
The GT reference tracks in both rows use this same body edge.

Reads the GT image series and the CME series of GT and the runs
(``sunerf.evaluation.cme.export_cme_series``).  Writes ``psi_jmap`` and the
speeds of the linear fits of the tracks to ``psi_jmap.txt``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.ndimage import map_coordinates
from sunpy.coordinates import frames
from sunpy.map import Map

from sunerf.data.psi.render_psi_thomson import carrington_to_hci_longitude
from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.cme.mass_transport import fit_speed
from sunerf.evaluation.psi.common import (
    DENSITY_CMAP, DENSITY_LABEL, FIGURE_WIDTH, TRUTH_TITLE, add_colorbar, density_norm, format_time_axis,
    leading_edge, load_series, save_figure,
)

HEIGHTS = np.arange(2.0, 30.0 + 1e-6, 0.25)
# position angles of the slit relative to the CME direction; the slit is their median
SLIT_ANGLES = np.deg2rad(np.linspace(-4.0, 4.0, 5))
TRUTH_LINE = "cyan"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True, help="GT image series <view>/tb/*.fts.")
    parser.add_argument("--views", nargs="+", default=["L4", "L5", "L5_wide"], help="Views of the naive J-maps.")
    parser.add_argument("--psi-series", type=Path, required=True, help="CME series of the GT cubes (npz).")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def cme_direction(s_map, latitude, longitude):
    """Unit vector of the CME direction (Carrington, radians) in the inertial frame at the time of the map."""
    lon = longitude + carrington_to_hci_longitude(s_map.date.datetime)
    return np.array([np.cos(latitude) * np.cos(lon), np.cos(latitude) * np.sin(lon), np.sin(latitude)])


def slit_profile(s_map, direction):
    """Brightness along the projected CME direction on the plane-of-sky height grid.

    Returns the profile and the distance [R_sun] and unit vector of the observer.
    """
    inertial = frames.HeliocentricInertial(obstime=s_map.date)
    observer = s_map.observer_coordinate.transform_to(inertial).cartesian.xyz.to_value(u.R_sun)
    point = SkyCoord(*(10.0 * direction) * u.R_sun, frame=inertial, representation_type="cartesian")
    center = np.array([v.to_value(u.pix) for v in s_map.world_to_pixel(
        SkyCoord(0 * u.arcsec, 0 * u.arcsec, frame=s_map.coordinate_frame))])
    offset = np.array([v.to_value(u.pix) for v in s_map.world_to_pixel(point)]) - center
    angle = np.arctan2(offset[1], offset[0])
    # tangent-plane scale of the image in R_sun per pixel
    rsun_per_pixel = float(s_map.scale[0].to_value(u.rad / u.pix)) * float(s_map.dsun.to_value(u.R_sun))
    pixels = HEIGHTS / rsun_per_pixel
    lines = [map_coordinates(np.asarray(s_map.data, dtype=float),
                             [center[1] + pixels * np.sin(angle + delta), center[0] + pixels * np.cos(angle + delta)],
                             order=1, mode="constant", cval=np.nan) for delta in SLIT_ANGLES]
    distance = np.linalg.norm(observer)
    return np.nanmedian(lines, axis=0), distance, observer / distance


def projected_height(radius, direction, observer_distance, observer_direction):
    """Plane-of-sky height at which a point at ``radius`` along ``direction`` appears to the observer."""
    cos_psi = float(np.dot(direction, observer_direction))
    sin_psi = np.sqrt(1.0 - cos_psi ** 2)
    return observer_distance * radius * sin_psi / (observer_distance - radius * cos_psi)


def naive_jmap(image_dir: Path, view, start, end, latitude, longitude):
    """Base-ratio J-map of one view over the CME period with the geometry of every frame."""
    times, profiles, geometry = [], [], []
    for file in sorted((image_dir / view / "tb").glob("*.fts")):
        s_map = Map(file)
        if not start <= s_map.date.datetime <= end:
            continue
        direction = cme_direction(s_map, latitude, longitude)
        profile, distance, observer_direction = slit_profile(s_map, direction)
        times.append(s_map.date.datetime)
        profiles.append(profile)
        geometry.append((direction, distance, observer_direction))
    if not times:
        raise FileNotFoundError(f"No {view} tB images of the CME period found in {image_dir}.")
    profiles = np.array(profiles)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = profiles / profiles[0]
    return np.array(times), ratio, geometry


def seconds(times):
    return np.array([(time - times[0]).total_seconds() for time in times])


def plot_track(axis, times, track, style, color, label):
    """Draw a height-time track; returns the speed of its linear fit in km/s."""
    axis.plot(times, track, style, color=color, linewidth=0.8, label=label)
    return fit_speed(seconds(times), track)


def write_speeds(path: Path, speeds) -> None:
    """Speed of the linear fit of every track, one line per panel and track."""
    lines = ["Speed of the linear fit of the height-time tracks in the J-maps [km/s].", "",
             f"{'panel':<22}{'track':<20}{'speed':>10}"]
    for (panel, track), speed in speeds.items():
        lines.append(f"{panel:<22}{track:<20}" + (f"{speed:>10.1f}" if np.isfinite(speed) else f"{'-':>10}"))
    path.write_text("\n".join(lines) + "\n")


def finish_axis(axis, title, ylabel, first, period, max_ticks=6):
    axis.set_title(title, fontsize=8)
    axis.set_xlim(*period)
    axis.set_ylim(HEIGHTS[0], HEIGHTS[-1])
    axis.set_xlabel("Time [UT]", fontsize=7)
    if first:
        axis.set_ylabel(ylabel, fontsize=7)
    else:
        axis.set_yticklabels([])
    format_time_axis(axis, max_ticks)


def main() -> None:
    args = parse_args()
    psi = load_series(args.psi_series)
    series = [psi] + [load_series(export_dir) for _, export_dir in args.runs]
    labels = [TRUTH_TITLE] + [label for label, _ in args.runs]
    latitude, longitude = float(psi["target_latitude"]), float(psi["target_longitude"])
    psi_seconds = np.array([time.timestamp() for time in psi["times"]])
    psi_edge = psi["body_edge_apex"]
    period = psi["times"][0], psi["times"][-1]
    speeds = {}

    fig = plt.figure(figsize=(FIGURE_WIDTH, 2 * FIGURE_WIDTH / (len(series) + 0.4) + 1.6), constrained_layout=True)
    rows = fig.add_gridspec(2, 1)

    naive = rows[0].subgridspec(1, len(args.views) + 1, width_ratios=(*[1] * len(args.views), 0.04))
    for column, view in enumerate(args.views):
        times, ratio, _ = naive_jmap(args.image_dir, view, *period, latitude, longitude)
        axis = fig.add_subplot(naive[0, column])
        artist = axis.pcolormesh(times, HEIGHTS, ratio.T, norm=LogNorm(0.8, 10.0), cmap="gray",
                                 shading="nearest", rasterized=True)
        # GT body edge along the CME direction at the times of the images
        edge = np.interp([time.timestamp() for time in times], psi_seconds, psi_edge)
        speeds[view, "Plane-of-sky fit"] = plot_track(axis, times, leading_edge(ratio, HEIGHTS), "-", "red",
                                                      "Plane-of-sky fit")
        plot_track(axis, times, edge, "--", TRUTH_LINE, "GT body edge")
        finish_axis(axis, f"{view}: tB / tB(t$_0$)", r"Plane-of-sky height [R$_\odot$]", column == 0, period)
    add_colorbar(fig, artist, naive[0, -1], "brightness ratio")

    volume = rows[1].subgridspec(1, len(series) + 1, width_ratios=(*[1] * len(series), 0.05))
    norm = density_norm(psi["beam_density"])
    for column, (entry, label) in enumerate(zip(series, labels)):
        axis = fig.add_subplot(volume[0, column])
        artist = axis.pcolormesh(entry["times"], entry["radius"], entry["beam_density"].T, norm=norm,
                                 cmap=DENSITY_CMAP, shading="nearest", rasterized=True)
        if column == 0:
            speeds[label, "GT body edge"] = plot_track(axis, entry["times"], entry["body_edge_apex"], "--",
                                                       TRUTH_LINE, "GT body edge")
        else:
            plot_track(axis, psi["times"], psi_edge, "--", TRUTH_LINE, "GT body edge")
            speeds[label, "SuNeRF body edge"] = plot_track(axis, entry["times"], entry["body_edge_apex"], "--",
                                                           "white", "SuNeRF body edge")
        finish_axis(axis, label, r"Radial distance [R$_\odot$]", column == 0, period, max_ticks=3)
    add_colorbar(fig, artist, volume[0, -1], DENSITY_LABEL)
    # one legend for all panels: the lines mean the same everywhere
    handles = [Line2D([], [], color="red", linestyle="-", linewidth=0.8, label="Plane-of-sky fit"),
               Line2D([], [], color=TRUTH_LINE, linestyle="--", linewidth=0.8, label="GT body edge"),
               Line2D([], [], color="white", linestyle="--", linewidth=0.8, label="SuNeRF body edge")]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles), fontsize=6, labelcolor="white",
               facecolor="0.35", edgecolor="none")
    print(save_figure(fig, args.output_dir, "psi_jmap"))
    write_speeds(args.output_dir / "psi_jmap.txt", speeds)
    print(args.output_dir / "psi_jmap.txt")


if __name__ == "__main__":
    main()
