#!/usr/bin/env python3
"""Plot 3: density cuts of GT and every run.

Rows are the Carrington map at a fixed radius, the meridional plane through the
CME and the equatorial plane; columns are GT and the runs
(``sunerf.evaluation.cme.export_density``).  Writes ``psi_density_cuts``.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.gridspec import GridSpec
from sunpy.coordinates import sun

from sunerf.evaluation.cme.export_layout import labelled_path
from sunerf.evaluation.psi.common import (
    DENSITY_LABEL, FIGURE_WIDTH, TRUTH_TITLE, add_colorbar, cme_longitude_index, density_norm, load_densities,
    plot_carrington, plot_equatorial, plot_meridional, save_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density-dir", type=Path, required=True, help="GT density cubes rho*.hdf.")
    parser.add_argument("--dump", type=int, required=True, help="Simulation dump of the compared density.")
    parser.add_argument("--runs", type=labelled_path, nargs="+", required=True, metavar="LABEL=EXPORT",
                        help="Column title and export directory of every compared run.")
    parser.add_argument("--map-radius", type=float, default=10.0,
                        help="Radius of the Carrington map in solar radii.")
    parser.add_argument("--cme-longitude", type=float, default=None,
                        help="Carrington longitude of the meridional cut in degrees; "
                             "default is the longitude of the largest density excess.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth, cubes, radius, theta, phi, time = load_densities(args.density_dir, args.dump, args.runs)
    cme_index = (cme_longitude_index(truth, radius, theta) if args.cme_longitude is None
                 else int(np.argmin(np.abs(phi - np.deg2rad(args.cme_longitude)))))
    earth_longitude = sun.L0(time).to_value(u.rad)
    norm = density_norm(truth * radius ** 2)
    shell = int(np.argmin(np.abs(radius - args.map_radius)))
    map_norm = density_norm(truth[..., shell] * radius[shell] ** 2)
    row_labels = (f"r = {radius[shell]:.0f} R$_\\odot$\nlatitude [deg]",
                  f"Meridional plane\n(Carr. lon. {np.rad2deg(phi[cme_index]):.0f}°)",
                  "Equatorial plane")

    n_columns = len(cubes) + 1
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.75 * FIGURE_WIDTH / (n_columns + 0.4) + 0.5),
                     constrained_layout=True)
    grid = GridSpec(3, n_columns + 1, figure=fig, width_ratios=(*[1] * n_columns, 0.05),
                    height_ratios=(0.55, 1, 1))
    titles = [TRUTH_TITLE, *(label for label, _ in args.runs)]
    for column, (cube, title) in enumerate(zip([truth, *cubes], titles)):
        axes = [fig.add_subplot(grid[0, column])] + [
            fig.add_subplot(grid[row, column], projection="polar") for row in (1, 2)]
        axes[0].set_title(title, fontsize=8)
        map_artist = plot_carrington(axes[0], cube, radius, theta, phi, args.map_radius, map_norm)
        cut_artist = plot_meridional(axes[1], cube, radius, theta, cme_index, norm)
        plot_equatorial(axes[2], cube, radius, theta, phi, earth_longitude, norm, phi[cme_index],
                        label_viewpoints=column == 0)
        axes[0].set_ylabel(row_labels[0] if column == 0 else "", fontsize=7)
        if column > 0:
            axes[0].set_yticklabels([])
            continue
        for axis, label in zip(axes[1:], row_labels[1:]):
            # Polar axes have no y label; the row label sits left of the disk, the r unit below.
            axis.text(-0.2, 0.5, label, transform=axis.transAxes, rotation=90, ha="center", va="center",
                      fontsize=7)
            axis.text(1.0, 0.0, r"r [R$_\odot$]", transform=axis.transAxes, ha="right", va="top", fontsize=7)
    add_colorbar(fig, map_artist, grid[0, -1], DENSITY_LABEL)
    # Both cuts share one normalization and therefore one colorbar.
    add_colorbar(fig, cut_artist, grid[1:, -1], DENSITY_LABEL)
    print(save_figure(fig, args.output_dir, "psi_density_cuts"))


if __name__ == "__main__":
    main()
