#!/usr/bin/env python3
"""Plot 5: tracking of the CME leading edge, body and cavity.

The maps show the compensated density of GT and of every run as a function of
radius and latitude (at the CME longitude) and of radius and longitude (at the
CME latitude) for the selected dumps.  Along every ray of the shown density the
markers give

* the leading edge: outermost radius where the density exceeds the pre-event
  density (reference dump) by 10 %,
* the body edge and the body peak: outer edge and maximum of the outermost
  enhancement of ``r^2 n_e``,
* the cavity: minimum between this enhancement and the next one inside.

The GT leading edge is repeated in the panels of the runs.  The bottom rows
show the kinematics inside the CME cone: height, speed and acceleration over
time.  The bands give the 16th to 84th percentile of the radius over the rays
of the cone.  Speed and acceleration curves come from quadratic fits in a
sliding window of 45 min and 90 min.  The cavity is found on few rays and is
shown as height only.

Reads the GT cubes, the density exports of the selected dumps and of the
reference dump (``sunerf.evaluation.cme.export_density``) and the CME series of
GT and the runs (``sunerf.evaluation.cme.export_cme_series``).  Writes
``psi_cme_tracking`` and, to ``psi_cme_tracking.txt``, the speed at 20 solar
radii and the acceleration of a constant-acceleration fit to every track.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from sunerf.evaluation.cme.export_cme_series import DETECTION
from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.cme.mass_transport import fit_constant_acceleration, local_kinematics, ray_structure
from sunerf.evaluation.psi.common import (
    DENSITY_CMAP, DENSITY_LABEL, FIGURE_WIDTH, RUN_COLORS, TRUTH_COLOR, TRUTH_TITLE, add_colorbar, density_norm,
    format_time_axis, load_densities, load_series, outermost_above, save_figure, series_leading_edge,
)

# track of the series (None: leading edge of the mass per radius), spread of the rays, panel title,
# speed and acceleration shown
TRACKS = ((None, None, "Leading edge", True), ("body_edge_apex", "body_edge", "Body edge (apex)", True),
          ("body_peak_median", "body_peak", "Body peak", True), ("cavity_median", "cavity", "Cavity", False))
SPEED_WINDOW = 45 * 60.0  # s
ACCELERATION_WINDOW = 90 * 60.0  # s
REFERENCE_RADIUS = 20.0  # R_sun
# structure of a ray: attribute of mass_transport.RayStructure, label, color
STRUCTURES = (("front", "body edge", "cyan"), ("sheath", "body peak", "lime"), ("cavity", "cavity", "magenta"))
MAP_HALF_WIDTH = np.deg2rad(60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True, help="GT density cubes rho*.hdf.")
    parser.add_argument("--psi-series", type=Path, required=True, help="CME series of the GT cubes (npz).")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--dumps", type=int, nargs="+", required=True, help="Simulation dumps of the maps.")
    parser.add_argument("--reference-dump", type=int, default=1, help="Pre-event dump of the leading edge.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def angular_separation(angle, target):
    return np.angle(np.exp(1j * (angle - target)))


def resample(cube, grid, target_grid):
    """Density ``cube[phi, theta, r]`` on another grid: nearest direction, log-linear in radius."""
    (radius, theta, phi), (target_radius, target_theta, target_phi) = grid, target_grid
    phi_index = np.abs(angular_separation(phi[None, :], target_phi[:, None])).argmin(axis=1)
    theta_index = np.abs(theta[None, :] - target_theta[:, None]).argmin(axis=1)
    rays = np.log(cube[phi_index][:, theta_index])
    return np.exp(interp1d(radius, rays, axis=-1, bounds_error=False, fill_value=np.nan)(target_radius))


def plot_map(axis, rays, reference_rays, radius, angle_deg, norm, psi_edge=None):
    """Compensated density ``rays[angle, r]`` with the structures detected along every ray."""
    artist = axis.pcolormesh(angle_deg, radius, (rays * radius ** 2).T, norm=norm, cmap=DENSITY_CMAP,
                             shading="nearest", rasterized=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        edges = [outermost_above(ratio, radius) for ratio in rays / reference_rays]
    # rays whose leading edge has reached the outer boundary are not marked
    edge = np.array([height if not at_limit else np.nan for height, at_limit in edges])
    axis.plot(angle_deg, edge, ".", markersize=1.5, color="white", label="leading edge")
    structures = [ray_structure(ray, radius, **DETECTION) for ray in rays]
    for attribute, label, color in STRUCTURES:
        axis.plot(angle_deg, [getattr(s, attribute) for s in structures], ".", markersize=1.5, color=color,
                  label=label)
    if psi_edge is not None:
        axis.plot(angle_deg, psi_edge, ".", markersize=1.5, color="black", label="GT leading edge")
    axis.tick_params(labelsize=6.5)
    return artist, edge


def write_fits(path: Path, labels, fits) -> None:
    """Speed at the reference radius and acceleration of the constant-acceleration fit of every track."""
    lines = [
        "Constant-acceleration fit r(t) = r0 + v0 t + a t^2 / 2 to every height-time track.",
        f"speed: km/s at {REFERENCE_RADIUS:g} R_sun (- if the track does not cross it); acceleration: m/s^2.",
        "",
        f"{'structure':<18}{'quantity':<14}" + "".join(f"  {label:>18}" for label in labels),
    ]
    for _, _, title, _ in TRACKS:
        for index, quantity in enumerate(("speed", "acceleration")):
            values = [fits[title, label][index] for label in labels]
            cells = [f"{value:>18.1f}" if np.isfinite(value) else f"{'-':>18}" for value in values]
            lines.append(f"{title:<18}{quantity:<14}" + "".join(f"  {cell}" for cell in cells))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    psi_series = load_series(args.psi_series)
    series = [psi_series] + [load_series(export_dir) for _, export_dir in args.runs]
    labels = [TRUTH_TITLE] + [label for label, _ in args.runs]
    target_latitude, target_longitude, half_angle = (
        float(psi_series[key]) for key in ("target_latitude", "target_longitude", "cone_half_angle"))
    reference_truth, reference_cubes, *reference_grid, _ = load_densities(
        args.density_dir, args.reference_dump, args.runs)

    n_rows = 2 * len(args.dumps)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 0.85 * n_rows + 4.4), constrained_layout=True)
    grid = GridSpec(n_rows + 1, len(labels) + 1, figure=fig, width_ratios=(*[1] * len(labels), 0.05),
                    height_ratios=(*[1] * n_rows, 5.0))
    cut_times = []
    for block, dump in enumerate(args.dumps):
        truth, cubes, radius, theta, phi, time = load_densities(args.density_dir, dump, args.runs)
        references = [resample(cube, reference_grid, (radius, theta, phi))
                      for cube in (reference_truth, *reference_cubes)]
        cut_times.append(time)
        if block == 0:
            norm = density_norm(truth * radius ** 2)
        latitude = np.pi / 2.0 - theta
        lat_index = int(np.argmin(np.abs(latitude - target_latitude)))
        lon_index = int(np.argmin(np.abs(angular_separation(phi, target_longitude))))
        # ascending latitude / longitude inside the window around the CME direction
        in_latitude = np.flatnonzero(np.abs(latitude - target_latitude) <= MAP_HALF_WIDTH)[::-1]
        separation = angular_separation(phi, target_longitude)
        in_longitude = np.flatnonzero(np.abs(separation) <= MAP_HALF_WIDTH)
        in_longitude = in_longitude[np.argsort(separation[in_longitude])]
        planes = (
            (f"lon. {np.rad2deg(phi[lon_index]):.0f}°", "Latitude [deg]", np.rad2deg(latitude[in_latitude]),
             np.rad2deg(target_latitude), lambda cube: cube[lon_index, in_latitude]),
            (f"lat. {np.rad2deg(latitude[lat_index]):.0f}°", "Carr. longitude [deg]",
             np.rad2deg(target_longitude + separation[in_longitude]), np.rad2deg(target_longitude),
             lambda cube: cube[in_longitude, lat_index]),
        )
        for plane, (cut, xlabel, angle_deg, center, select) in enumerate(planes):
            row = 2 * block + plane
            psi_edge = None
            for column, (cube, reference, label) in enumerate(zip([truth, *cubes], references, labels)):
                axis = fig.add_subplot(grid[row, column])
                artist, edge = plot_map(axis, select(cube), select(reference), radius, angle_deg, norm, psi_edge)
                if column == 0:
                    psi_edge = edge
                    axis.set_ylabel(f"{time:%H:%M} UT, {cut}\n" + r"Radius [R$_\odot$]", fontsize=6.5)
                else:
                    axis.set_yticklabels([])
                for side in (-1, 1):
                    axis.axvline(center + side * np.rad2deg(half_angle), color="white", linewidth=0.4,
                                 linestyle=":")
                if row == 0:
                    axis.set_title(label, fontsize=8)
                if row == 0 and column == len(labels) - 1:
                    # the last column holds all markers, including the GT leading edge
                    fig.legend(*axis.get_legend_handles_labels(), loc="outside upper center", ncols=5, fontsize=6,
                               markerscale=3, labelcolor="white", facecolor="0.35", edgecolor="none",
                               handletextpad=0.1, columnspacing=1.0)
                if block == len(args.dumps) - 1:
                    axis.set_xlabel(xlabel, fontsize=6.5, labelpad=1)
    add_colorbar(fig, artist, grid[:n_rows, -1], DENSITY_LABEL)

    rows = (r"Radius [R$_\odot$]", "Speed [km/s]", r"Accel. [m/s$^2$]")
    tracks = grid[n_rows, :].subgridspec(len(rows), len(TRACKS), height_ratios=(1.2, 1, 1))
    fits = {}
    for column, (key, spread, title, kinematics) in enumerate(TRACKS):
        axes = [fig.add_subplot(tracks[row, column]) for row in range(len(rows) if kinematics else 1)]
        for entry, label, color in zip(series, labels, [TRUTH_COLOR, *RUN_COLORS]):
            times = entry["times"]
            seconds = np.array([(t - times[0]).total_seconds() for t in times])
            track = series_leading_edge(entry, "mass_per_radius") if key is None else entry[key]
            fits[title, label] = fit_constant_acceleration(seconds, track, REFERENCE_RADIUS)
            axes[0].plot(times, track, "-", linewidth=0.8, color=color, label=label)
            if spread is not None:
                axes[0].fill_between(times, entry[f"{spread}_p16"], entry[f"{spread}_p84"], color=color,
                                     alpha=0.15, linewidth=0)
            if kinematics:
                axes[1].plot(times, local_kinematics(seconds, track, SPEED_WINDOW)[0], "-", linewidth=0.8,
                             color=color)
                axes[2].plot(times, local_kinematics(seconds, track, ACCELERATION_WINDOW)[1], "-", linewidth=0.8,
                             color=color)
        axes[0].set_title(title, fontsize=8)
        axes[-1].set_xlabel("Time [UT]", fontsize=7)
        for row, axis in enumerate(axes):
            for time in cut_times:
                axis.axvline(time, color="0.6", linewidth=0.5, linestyle=":")
            axis.set_xlim(psi_series["times"][0], psi_series["times"][-1])
            if column == 0:
                axis.set_ylabel(rows[row], fontsize=7)
            format_time_axis(axis, max_ticks=4)
            if axis is not axes[-1]:
                axis.set_xticklabels([])
        if kinematics:
            axes[2].axhline(0.0, color="0.6", linewidth=0.5)
        else:
            # the free cells below the cavity hold the legend of all track panels
            legend_axis = fig.add_subplot(tracks[1:, column])
            legend_axis.axis("off")
            legend_axis.legend(*axes[0].get_legend_handles_labels(), loc="center", fontsize=6, frameon=False)
    write_fits(args.output_dir / "psi_cme_tracking.txt", labels, fits)
    print(save_figure(fig, args.output_dir, "psi_cme_tracking"))


if __name__ == "__main__":
    main()
