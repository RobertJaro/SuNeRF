#!/usr/bin/env python3
import argparse

import numpy as np

from sunerf.data.prep.base_insitu import (
    BaseInSituPreparer,
    as_cm3,
    as_kms,
    filter_time_range,
    nearest_values,
)


class SolarOrbiterInSituPreparer(BaseInSituPreparer):
    label = "Solar Orbiter"
    dataset_name = "SOLO_COHO1HR_MERGED_MAG_PLASMA"
    default_horizons_target = "Solar Orbiter"
    density_source_labels = {3.0: "COHO proton proxy"}
    density_source_encoding = {"solo_coho_proton_proxy": 3.0}
    qa_title = "Solar Orbiter COHO in-situ QA"
    radius_ylabel = "Solar Orbiter radius [Rsun]"

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(description="Prepare Solar Orbiter COHO in-situ constraints for SuNeRF-CME.")
        BaseInSituPreparer.add_common_args(
            parser,
            default_horizons_target=SolarOrbiterInSituPreparer.default_horizons_target,
            default_delta_seconds=3600.0,
            description_label="Solar Orbiter",
        )
        parser.add_argument(
            "--electron-density-factor",
            type=float,
            default=1.0,
            help="Multiplier converting COHO proton density to electron-density target. Default assumes n_e ~= n_p.",
        )
        return parser

    def run(self, args):
        if args.electron_density_factor <= 0:
            raise ValueError("--electron-density-factor must be positive")

        raw_files = self.load_raw_files(args.raw_dir)
        density_t, proton_density, density_units, density_name = self.load_series(
            raw_files,
            ["protonDensity", "ProtonDensity", "SWA_PAS_DENSITY"],
            contains=("density",),
        )
        vr_t, vr, vr_units, vr_name = self.load_series(raw_files, ["VR", "SWA_PAS_VELOCITY_RTN"], contains=("vr",))
        vt_t, vt, vt_units, vt_name = self.load_series(raw_files, ["VT"], contains=("vt",))
        vn_t, vn, vn_units, vn_name = self.load_series(raw_files, ["VN"], contains=("vn",))

        self.require_series(
            {
                "protonDensity": density_t,
                "VR": vr_t,
                "VT": vt_t,
                "VN": vn_t,
            }
        )
        arrays = self.build_arrays(
            args,
            density_t,
            proton_density,
            density_units,
            vr_t,
            vr,
            vr_units,
            vt_t,
            vt,
            vt_units,
            vn_t,
            vn,
            vn_units,
        )
        meta = {
            "dataset": self.dataset_name,
            "density_var": density_name,
            "density_units": density_units,
            "density_kind": "proton_density_as_electron_proxy",
            "electron_density_factor": args.electron_density_factor,
            "velocity_vars": {"r": vr_name, "t": vt_name, "n": vn_name},
            "velocity_units": {"r": vr_units, "t": vt_units, "n": vn_units},
            "output_units": {
                "raw_density_cm3": "cm^-3",
                "density_cm3": "cm^-3 electron-density target",
            },
        }
        self.save_product(args, arrays, meta)

    @staticmethod
    def require_series(series_by_name):
        missing = [name for name, values in series_by_name.items() if values is None]
        if missing:
            raise KeyError(f"Missing required Solar Orbiter COHO variables: {', '.join(missing)}")

    def build_arrays(self, args, density_t, proton_density, density_units, vr_t, vr, vr_units,
                     vt_t, vt, vt_units, vn_t, vn, vn_units):
        start = np.datetime64(args.start)
        end = np.datetime64(args.end)

        density_time_unix, raw_density_values = filter_time_range(
            density_t,
            as_cm3(proton_density, density_units),
            start,
            end,
        )
        vr_t, vr = filter_time_range(vr_t, as_kms(vr, vr_units), start, end)
        vt_t, vt = filter_time_range(vt_t, as_kms(vt, vt_units), start, end)
        vn_t, vn = filter_time_range(vn_t, as_kms(vn, vn_units), start, end)

        velocity_time_unix = np.unique(np.concatenate([vr_t, vt_t, vn_t])).astype(np.float64)
        dst_t = np.unique(np.concatenate([density_time_unix, velocity_time_unix])).astype(np.float64)

        raw_density_cm3, density_dt, density_valid = nearest_values(
            density_time_unix,
            raw_density_values,
            dst_t,
            args.max_density_time_delta_seconds,
        )
        density_cm3 = raw_density_cm3 * args.electron_density_factor
        vr_nearest, vr_dt, vr_valid = nearest_values(vr_t, vr, dst_t, args.max_velocity_time_delta_seconds)
        vt_nearest, vt_dt, vt_valid = nearest_values(vt_t, vt, dst_t, args.max_velocity_time_delta_seconds)
        vn_nearest, vn_dt, vn_valid = nearest_values(vn_t, vn, dst_t, args.max_velocity_time_delta_seconds)
        velocity_rtn = np.stack([vr_nearest, vt_nearest, vn_nearest], axis=-1)
        velocity_time_delta = np.maximum.reduce([vr_dt, vt_dt, vn_dt])
        velocity_radial_kms = velocity_rtn[:, 0]

        finite_positive_density = density_valid & np.isfinite(density_cm3) & (density_cm3 > 0)
        has_velocity_mask = self.valid_radial_velocity_mask(vr_valid & vt_valid & vn_valid, velocity_rtn, velocity_radial_kms)
        valid = finite_positive_density | has_velocity_mask

        self.print_sample_accounting(
            "Solar Orbiter in-situ prep sample accounting:",
            density_time_unix.size,
            velocity_time_unix.size,
            dst_t,
            finite_positive_density,
            has_velocity_mask,
            args,
            velocity_label="native velocity timestamps",
        )

        raw_density_cm3[~finite_positive_density] = np.nan
        density_cm3[~finite_positive_density] = np.nan
        density_dt[~finite_positive_density] = np.nan
        velocity_rtn[~has_velocity_mask] = np.nan
        velocity_radial_kms[~has_velocity_mask] = np.nan
        velocity_time_delta[~has_velocity_mask] = np.nan

        arrays = {
            "time_unix": dst_t[valid].astype(np.float64),
            "raw_density_cm3": raw_density_cm3[valid].astype(np.float32),
            "density_cm3": density_cm3[valid].astype(np.float32),
            "density_source": np.full(np.count_nonzero(valid), 3.0, dtype=np.float32),
            "density_time_delta_seconds": density_dt[valid].astype(np.float32),
            "has_density": finite_positive_density[valid].astype(np.float32),
            "has_velocity": has_velocity_mask[valid].astype(np.float32),
            "velocity_rtn_kms": velocity_rtn[valid].astype(np.float32),
            "velocity_radial_kms": velocity_radial_kms[valid].astype(np.float32),
            "velocity_time_delta_seconds": velocity_time_delta[valid].astype(np.float32),
        }
        if arrays["time_unix"].size == 0:
            raise ValueError("No Solar Orbiter samples remain after requiring density or velocity")
        return arrays


def main():
    args = SolarOrbiterInSituPreparer.build_parser().parse_args()
    SolarOrbiterInSituPreparer().run(args)


if __name__ == "__main__":
    main()
