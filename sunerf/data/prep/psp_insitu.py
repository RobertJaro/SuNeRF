#!/usr/bin/env python3
import argparse
import os

import numpy as np

from sunerf.data.prep.base_insitu import (
    BaseInSituPreparer,
    as_cm3,
    as_kms,
    filter_time_range,
    nearest_values,
)


class PSPInSituPreparer(BaseInSituPreparer):
    label = "PSP"
    dataset_name = "PSP in-situ"
    default_horizons_target = "Parker Solar Probe"
    fallback_horizons_targets = {"Parker Solar Probe": ("Solar Probe Plus",)}
    density_source_labels = {1.0: "QTN", 2.0: "SPC"}
    density_source_encoding = {"qtn": 1.0, "spc": 2.0}
    qa_title = "PSP in-situ QA"
    radius_ylabel = "PSP radius [Rsun]"

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(description="Prepare PSP in-situ constraints for SuNeRF-CME.")
        BaseInSituPreparer.add_common_args(
            parser,
            default_horizons_target=PSPInSituPreparer.default_horizons_target,
            default_delta_seconds=300.0,
            description_label="PSP",
        )
        parser.add_argument("--density-source", choices=["qtn", "spc", "qtn_or_spc"], default="qtn_or_spc")
        parser.add_argument(
            "--check-horizons",
            action="store_true",
            help="Compare sampled prepared PSP HCI positions against JPL Horizons.",
        )
        parser.add_argument(
            "--horizons-check-points",
            type=int,
            default=12,
            help="Number of equal-time prepared PSP samples to compare when --check-horizons is used.",
        )
        return parser

    def run(self, args):
        raw_files = self.load_raw_files(args.raw_dir)
        spc_files, qtn_files = self.split_products(raw_files)

        qtn_t, qtn_density, qtn_units, qtn_name = self.load_series(
            qtn_files,
            ["fit_n_e", "electron_density", "n_e", "ne", "density"],
            contains=("density",),
        )
        spc_t, spc_density, spc_units, spc_density_name = self.load_series(
            spc_files,
            ["np_fit", "np1_fit", "np_moment"],
            contains=("np",),
        )
        v_t, velocity_rtn, v_units, velocity_name = self.load_series(
            spc_files,
            ["vp_fit_RTN", "vp1_fit_RTN", "vp_moment_RTN"],
            contains=("vp", "rtn"),
            vector=True,
        )

        arrays, duplicate_count = self.build_arrays(args, qtn_t, qtn_density, qtn_units, spc_t, spc_density,
                                                    spc_units, v_t, velocity_rtn, v_units)
        meta = {
            "qtn_density_var": qtn_name,
            "qtn_density_units": qtn_units,
            "spc_density_var": spc_density_name,
            "spc_density_units": spc_units,
            "velocity_var": velocity_name,
            "velocity_units": v_units,
        }
        if duplicate_count:
            meta["discarded_duplicate_qtn_or_spc_timestamps"] = int(duplicate_count)

        self.save_product(
            args,
            arrays,
            meta,
            check_horizons=args.check_horizons,
            horizons_check_points=args.horizons_check_points,
        )

    @staticmethod
    def split_products(raw_files):
        qtn_files = [f for f in raw_files if "sqtn" in os.path.basename(f).lower() or "qtn" in os.path.basename(f).lower()]
        coho_files = [f for f in raw_files if "coho" in os.path.basename(f).lower() or "merged_mag_plasma" in os.path.basename(f).lower()]
        spc_files = [f for f in raw_files if "spc" in os.path.basename(f).lower()]
        if not spc_files:
            spc_files = [f for f in raw_files if f not in qtn_files and f not in coho_files]
        return spc_files, qtn_files

    def build_arrays(self, args, qtn_t, qtn_density, qtn_units, spc_t, spc_density, spc_units,
                     v_t, velocity_rtn, v_units):
        start = np.datetime64(args.start)
        end = np.datetime64(args.end)

        density_parts = []
        if qtn_t is not None:
            qtn_time_unix, qtn_density = filter_time_range(qtn_t, as_cm3(qtn_density, qtn_units), start, end)
            if args.density_source in {"qtn", "qtn_or_spc"}:
                density_parts.append((qtn_time_unix, qtn_density, np.full(qtn_time_unix.shape, 1.0)))
        if spc_t is not None:
            spc_time_unix, spc_density = filter_time_range(spc_t, as_cm3(spc_density, spc_units), start, end)
            if args.density_source in {"spc", "qtn_or_spc"}:
                density_parts.append((spc_time_unix, spc_density, np.full(spc_time_unix.shape, 2.0)))
        if not density_parts:
            raise KeyError("Could not find density data in QTN or SPC products")
        if velocity_rtn is None:
            raise KeyError("Could not find vp_fit_RTN velocity data in SPC product")

        velocity_rtn = as_kms(velocity_rtn[:, :3], v_units)
        velocity_time_unix, velocity_rtn = filter_time_range(v_t, velocity_rtn, start, end)

        density_time_unix = np.concatenate([part[0] for part in density_parts])
        density_values = np.concatenate([part[1] for part in density_parts])
        density_source_values = np.concatenate([part[2] for part in density_parts])
        order = np.argsort(density_time_unix, kind="stable")
        density_time_unix = density_time_unix[order]
        density_values = density_values[order]
        density_source_values = density_source_values[order]

        duplicate_count = 0
        if args.density_source == "qtn_or_spc":
            _, unique_idx = np.unique(density_time_unix, return_index=True)
            unique_idx = np.sort(unique_idx)
            duplicate_count = density_time_unix.size - unique_idx.size
            density_time_unix = density_time_unix[unique_idx]
            density_values = density_values[unique_idx]
            density_source_values = density_source_values[unique_idx]

        dst_t = np.unique(np.concatenate([density_time_unix, velocity_time_unix])).astype(np.float64)
        density, density_dt, density_valid = nearest_values(
            density_time_unix,
            density_values,
            dst_t,
            args.max_density_time_delta_seconds,
        )
        density_source, _, density_source_valid = nearest_values(
            density_time_unix,
            density_source_values,
            dst_t,
            args.max_density_time_delta_seconds,
        )
        velocity_rtn, velocity_dt, velocity_valid = nearest_values(
            velocity_time_unix,
            velocity_rtn,
            dst_t,
            args.max_velocity_time_delta_seconds,
        )

        velocity_radial_kms = velocity_rtn[:, 0]
        finite_positive_density = density_valid & density_source_valid & np.isfinite(density) & (density > 0)
        has_velocity_mask = self.valid_radial_velocity_mask(velocity_valid, velocity_rtn, velocity_radial_kms)
        valid = finite_positive_density | has_velocity_mask

        self.print_sample_accounting(
            "PSP in-situ prep sample accounting:",
            density_time_unix.size,
            velocity_time_unix.size,
            dst_t,
            finite_positive_density,
            has_velocity_mask,
            args,
            duplicate_count=duplicate_count,
        )

        density[~finite_positive_density] = np.nan
        density_source[~finite_positive_density] = 0.0
        density_dt[~finite_positive_density] = np.nan
        velocity_rtn[~has_velocity_mask] = np.nan
        velocity_radial_kms[~has_velocity_mask] = np.nan
        velocity_dt[~has_velocity_mask] = np.nan

        arrays = {
            "time_unix": dst_t[valid].astype(np.float64),
            "density_cm3": density[valid].astype(np.float32),
            "density_source": density_source[valid].astype(np.float32),
            "density_time_delta_seconds": density_dt[valid].astype(np.float32),
            "has_density": finite_positive_density[valid].astype(np.float32),
            "has_velocity": has_velocity_mask[valid].astype(np.float32),
            "velocity_rtn_kms": velocity_rtn[valid].astype(np.float32),
            "velocity_radial_kms": velocity_radial_kms[valid].astype(np.float32),
            "velocity_time_delta_seconds": velocity_dt[valid].astype(np.float32),
        }
        if arrays["time_unix"].size == 0:
            raise ValueError("No PSP samples remain after requiring density or velocity")
        return arrays, duplicate_count


def main():
    args = PSPInSituPreparer.build_parser().parse_args()
    PSPInSituPreparer().run(args)


if __name__ == "__main__":
    main()
