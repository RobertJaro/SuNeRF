#!/usr/bin/env python3
"""Plot 5: front and cavity of the CME in the meridional cut at selected times.

The maps show the compensated density of GT and of every run as a function of
radius and latitude at the CME longitude, one row per dump.  Along every ray of
the shown density the markers give

* the front: maximum of the outermost enhancement of ``r^2 n_e`` (the bright dense front),
* the cavity: minimum behind the front, in front of the next enhancement inside.

The GT front is repeated in the panels of the runs.  The kinematics of both
structures are shown by ``plot_cme_kinematics``.

Reads the GT cubes and the density exports of the selected dumps
(``sunerf.evaluation.cme.export_density``) and the CME direction from the GT
series (``sunerf.evaluation.cme.export_cme_series``).  Writes ``psi_cme_tracking``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from sunerf.evaluation.cme.export_cme_series import DETECTION
from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.cme.mass_transport import ray_structure
from sunerf.evaluation.psi.common import (
    DENSITY_CMAP, DENSITY_LABEL, FIGURE_WIDTH, TRUTH_TITLE, add_colorbar, density_norm, load_densities,
    load_series, save_figure, title_size,
)

# structure of a ray: attribute of mass_transport.RayStructure, label, color
STRUCTURES = (("sheath", "front", "white"), ("cavity", "cavity", "cyan"))
MAP_HALF_WIDTH = np.deg2rad(60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True, help="GT density cubes rho*.hdf.")
    parser.add_argument("--psi-series", type=Path, required=True, help="CME series of the GT cubes (npz).")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--dumps", type=int, nargs="+", required=True, help="Simulation dumps of the maps.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def angular_separation(angle, target):
    return np.angle(np.exp(1j * (angle - target)))


def plot_map(axis, rays, radius, latitude_deg, norm, truth_front=None):
    """Compensated density ``rays[latitude, r]`` with the structures detected along every ray."""
    artist = axis.pcolormesh(latitude_deg, radius, (rays * radius ** 2).T, norm=norm, cmap=DENSITY_CMAP,
                             shading="nearest", rasterized=True)
    structures = [ray_structure(ray, radius, **DETECTION) for ray in rays]
    if truth_front is not None:
        axis.plot(latitude_deg, truth_front, ".", markersize=1.5, color="black", label="GT front")
    for attribute, label, color in STRUCTURES:
        axis.plot(latitude_deg, [getattr(s, attribute) for s in structures], ".", markersize=1.5, color=color,
                  label=label)
    axis.tick_params(labelsize=6.5)
    return artist, np.array([s.sheath for s in structures])


def main() -> None:
    args = parse_args()
    psi_series = load_series(args.psi_series)
    labels = [TRUTH_TITLE] + [label for label, _ in args.runs]
    target_latitude, target_longitude, half_angle = (
        float(psi_series[key]) for key in ("target_latitude", "target_longitude", "cone_half_angle"))

    fig = plt.figure(figsize=(FIGURE_WIDTH, 0.95 * len(args.dumps) + 0.9), constrained_layout=True)
    grid = GridSpec(len(args.dumps), len(labels) + 1, figure=fig, width_ratios=(*[1] * len(labels), 0.05))
    for row, dump in enumerate(args.dumps):
        truth, cubes, radius, theta, phi, time = load_densities(args.density_dir, dump, args.runs)
        if row == 0:
            norm = density_norm(truth * radius ** 2)
        latitude = np.pi / 2.0 - theta
        lon_index = int(np.argmin(np.abs(angular_separation(phi, target_longitude))))
        # ascending latitude inside the window around the CME direction
        in_latitude = np.flatnonzero(np.abs(latitude - target_latitude) <= MAP_HALF_WIDTH)[::-1]
        latitude_deg = np.rad2deg(latitude[in_latitude])
        truth_front = None
        for column, (cube, label) in enumerate(zip([truth, *cubes], labels)):
            axis = fig.add_subplot(grid[row, column])
            artist, front = plot_map(axis, cube[lon_index, in_latitude], radius, latitude_deg, norm, truth_front)
            if column == 0:
                truth_front = front
                axis.set_ylabel(f"{time:%H:%M} UT\n" + r"Radius [R$_\odot$]", fontsize=6.5)
            else:
                axis.set_yticklabels([])
            for side in (-1, 1):
                axis.axvline(np.rad2deg(target_latitude + side * half_angle), color="white", linewidth=0.4,
                             linestyle=":")
            if row == 0:
                axis.set_title(label, fontsize=title_size(len(labels)))
            if row == 0 and column == len(labels) - 1:
                # the last column holds all markers, including the GT front
                fig.legend(*axis.get_legend_handles_labels(), loc="outside upper center", ncols=3, fontsize=6,
                           markerscale=3, labelcolor="white", facecolor="0.35", edgecolor="none",
                           handletextpad=0.1, columnspacing=1.0)
            if row == len(args.dumps) - 1:
                axis.set_xlabel("Latitude [deg]", fontsize=6.5, labelpad=1)
            else:
                axis.set_xticklabels([])
    add_colorbar(fig, artist, grid[:, -1], DENSITY_LABEL)
    print(save_figure(fig, args.output_dir, "psi_cme_tracking"))


if __name__ == "__main__":
    main()
