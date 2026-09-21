import json
import os
from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from astropy.time import Time


def require_cdflib(label="in-situ"):
    try:
        import cdflib
    except ImportError as exc:
        raise ImportError(f"{label} prep requires cdflib. Install it with `pip install cdflib`.") from exc
    return cdflib


def cdf_vars(path, label="in-situ"):
    cdflib = require_cdflib(label)
    cdf = cdflib.CDF(path)
    info = cdf.cdf_info()
    zvars = getattr(info, "zVariables", None)
    rvars = getattr(info, "rVariables", None)
    if zvars is None and isinstance(info, dict):
        zvars = info.get("zVariables", [])
        rvars = info.get("rVariables", [])
    return cdf, list(zvars or []) + list(rvars or [])


def find_var(varnames, aliases, contains=()):
    lower_map = {v.lower(): v for v in varnames}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    for var in varnames:
        v = var.lower()
        if all(token.lower() in v for token in contains):
            return var
    return None


def var_units(cdf, name):
    try:
        attrs = cdf.varattsget(name)
    except Exception:
        return ""
    for key in ("UNITS", "Units", "units"):
        if key in attrs:
            value = attrs[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            return str(value)
    return ""


def read_var(cdf, name):
    data = np.asarray(cdf.varget(name))
    try:
        attrs = cdf.varattsget(name)
    except Exception:
        attrs = {}

    fill_values = []
    for key in ("FILLVAL", "FillVal"):
        if key not in attrs:
            continue
        value = attrs[key]
        if isinstance(value, (list, tuple, np.ndarray)):
            fill_values.extend(np.asarray(value).reshape(-1).tolist())
        else:
            fill_values.append(value)

    if not np.issubdtype(data.dtype, np.number):
        return data

    data = data.astype(np.float64, copy=False)
    if np.issubdtype(data.dtype, np.floating):
        for fill in fill_values:
            try:
                data[np.isclose(data, float(fill), rtol=0.0, atol=0.0)] = np.nan
            except Exception:
                pass
    return data


def read_epoch(cdf, varnames, label="in-situ"):
    cdflib = require_cdflib(label)
    epoch_name = find_var(varnames, ["Epoch", "epoch", "Epoch_1", "Epoch_time"])
    if epoch_name is None:
        raise KeyError("Could not find Epoch variable")
    epoch = np.asarray(cdf.varget(epoch_name))
    return np.asarray(cdflib.cdfepoch.to_datetime(epoch), dtype="datetime64[ns]")


def datetime64_to_unix_seconds(times):
    return times.astype("datetime64[ns]").astype(np.int64) / 1e9


def sort_unique_series(times, values):
    times = np.asarray(times)
    values = np.asarray(values)
    order = np.argsort(times.astype("datetime64[ns]").astype(np.int64), kind="stable")
    times = times[order]
    values = values[order]
    time_ns = times.astype("datetime64[ns]").astype(np.int64)
    keep = np.ones(time_ns.shape, dtype=bool)
    keep[1:] = time_ns[1:] != time_ns[:-1]
    return times[keep], values[keep]


def load_series(files, aliases, contains=(), vector=False, label="in-situ"):
    times_all, values_all = [], []
    units = ""
    used_name = None
    for path in sorted(files):
        cdf, varnames = cdf_vars(path, label=label)
        name = find_var(varnames, aliases, contains=contains)
        if name is None:
            continue
        values = read_var(cdf, name)
        if vector and values.ndim > 2:
            values = values.reshape(values.shape[0], -1)
        if vector and values.ndim == 1:
            continue
        if not vector and values.ndim > 1:
            values = values.reshape(values.shape[0], -1)[:, 0]
        times_all.append(read_epoch(cdf, varnames, label=label))
        values_all.append(values)
        units = units or var_units(cdf, name)
        used_name = used_name or name
    if not times_all:
        return None, None, "", None
    times = np.concatenate(times_all)
    values = np.concatenate(values_all)
    times, values = sort_unique_series(times, values)
    return times, values, units, used_name


def filter_time_range(times, values, start, end):
    unix = datetime64_to_unix_seconds(times)
    start_s = start.astype("datetime64[ns]").astype(np.int64) / 1e9
    end_s = end.astype("datetime64[ns]").astype(np.int64) / 1e9
    mask = np.isfinite(unix) & (unix >= start_s) & (unix < end_s)
    return unix[mask].astype(np.float64), np.asarray(values)[mask]


def nearest_values(src_t, src_y, dst_t, max_delta_seconds):
    src_t = np.asarray(src_t, dtype=np.float64)
    src_y = np.asarray(src_y, dtype=np.float64)
    dst_t = np.asarray(dst_t, dtype=np.float64)
    if src_t.size == 0:
        raise ValueError("Cannot match nearest values from an empty source time series")

    order = np.argsort(src_t, kind="stable")
    src_t = src_t[order]
    src_y = src_y[order]

    positions = np.searchsorted(src_t, dst_t)
    next_pos = np.clip(positions, 0, src_t.size - 1)
    prev_pos = np.clip(positions - 1, 0, src_t.size - 1)
    next_delta = np.abs(src_t[next_pos] - dst_t)
    prev_delta = np.abs(src_t[prev_pos] - dst_t)
    use_prev = prev_delta <= next_delta
    nearest_pos = np.where(use_prev, prev_pos, next_pos)
    delta = np.where(use_prev, prev_delta, next_delta)
    values = src_y[nearest_pos]
    valid = np.isfinite(delta) & (delta <= max_delta_seconds)
    if values.ndim == 1:
        valid &= np.isfinite(values)
    else:
        valid &= np.all(np.isfinite(values), axis=-1)
    return values, delta, valid


def as_kms(values, units):
    values = np.asarray(values, dtype=np.float64)
    units_l = (units or "").lower().replace(" ", "")
    if "m/s" in units_l and "km/s" not in units_l:
        return values / 1000.0
    return values


def as_cm3(values, units):
    values = np.asarray(values, dtype=np.float64)
    units_l = (units or "").lower().replace(" ", "")
    if "cm^-3" in units_l or "cm-3" in units_l or "/cm3" in units_l or "cc" in units_l:
        return values
    if "m^-3" in units_l or "m-3" in units_l or "/m3" in units_l:
        return values / 1e6
    return values


def iso_time(t):
    return np.datetime_as_string(np.datetime64(int(round(float(t) * 1_000_000_000)), "ns"), unit="s")


def print_loss_reason(label, mask, time_unix, max_examples=5):
    count = int(np.count_nonzero(mask))
    if count == 0:
        print(f"  lost {label}: 0")
        return
    lost_times = np.asarray(time_unix)[mask]
    examples = ", ".join(iso_time(t) for t in lost_times[:max_examples])
    print(
        f"  lost {label}: {count} "
        f"(first={iso_time(lost_times[0])}, last={iso_time(lost_times[-1])}, examples={examples})"
    )


def sample_time_indices(time_unix, n_points):
    time_unix = np.asarray(time_unix, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(time_unix))
    if finite_idx.size == 0 or n_points <= 0:
        return finite_idx
    if finite_idx.size <= n_points:
        return finite_idx

    finite_times = time_unix[finite_idx]
    target_times = np.linspace(finite_times[0], finite_times[-1], n_points)
    positions = np.searchsorted(finite_times, target_times)
    positions = np.clip(positions, 0, finite_idx.size - 1)
    previous_positions = np.clip(positions - 1, 0, finite_idx.size - 1)

    next_delta = np.abs(finite_times[positions] - target_times)
    previous_delta = np.abs(finite_times[previous_positions] - target_times)
    nearest_positions = np.where(previous_delta < next_delta, previous_positions, positions)
    return finite_idx[np.unique(nearest_positions)]


def inertial_coords_from_hci(position_hci_rsun):
    position_hci_rsun = np.atleast_2d(position_hci_rsun).astype(np.float64)
    radius = np.linalg.norm(position_hci_rsun, axis=-1)
    lat = np.arcsin(np.clip(position_hci_rsun[:, 2] / np.clip(radius, 1e-12, None), -1.0, 1.0))
    lon = np.arctan2(position_hci_rsun[:, 1], position_hci_rsun[:, 0])
    return np.stack([radius, lat, lon], axis=-1)


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


def spherical_from_hci_deg(position_hci_rsun):
    sph = inertial_coords_from_hci(position_hci_rsun)
    return sph[:, 0], np.rad2deg(sph[:, 2]), np.rad2deg(sph[:, 1])


def angle_delta_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class BaseInSituPreparer:
    label = "in-situ"
    dataset_name = "in-situ"
    default_horizons_target = None
    density_source_labels = {}
    density_source_encoding = {}
    fallback_horizons_targets = {}
    qa_title = "In-situ QA"
    radius_ylabel = "radius [Rsun]"

    def load_raw_files(self, raw_dir):
        import glob

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.cdf")))
        if not raw_files:
            raise FileNotFoundError(f"No CDF files found in {raw_dir}")
        return raw_files

    def load_series(self, files, aliases, contains=(), vector=False):
        return load_series(files, aliases, contains=contains, vector=vector, label=self.label)

    @staticmethod
    def add_common_args(parser, default_horizons_target, default_delta_seconds, description_label):
        parser.add_argument("--raw-dir", required=True, help=f"Directory containing downloaded {description_label} CDF files")
        parser.add_argument("--out", required=True, help="Output .npz file")
        parser.add_argument("--start", default="2025-09-01T00:00:00")
        parser.add_argument("--end", default="2025-10-01T00:00:00")
        parser.add_argument(
            "--max-velocity-time-delta-seconds",
            type=float,
            default=default_delta_seconds,
            help="Maximum allowed native velocity time offset from a selected density sample.",
        )
        parser.add_argument(
            "--max-density-time-delta-seconds",
            type=float,
            default=default_delta_seconds,
            help="Maximum allowed native density time offset from a selected velocity sample.",
        )
        parser.add_argument("--horizons-target", default=default_horizons_target, help="JPL Horizons target used for HCI positions")
        parser.add_argument(
            "--position-cadence-seconds",
            type=float,
            default=3600.0,
            help="Cadence for Horizons position lookup. Native measurements are matched to the nearest position sample.",
        )
        parser.add_argument("--horizons-chunk-size", type=int, default=1000, help="Number of timestamps per Horizons request")
        parser.add_argument("--plot-dir", default=None, help=f"Optional directory for {description_label} prep QA plots")

    def _horizons_hci_positions(self, time_unix, target, chunk_size):
        from sunpy.coordinates import frames, get_horizons_coord

        positions = []
        fallback_targets = self.fallback_horizons_targets.get(target, ())
        for start in range(0, len(time_unix), chunk_size):
            chunk = np.asarray(time_unix[start:start + chunk_size], dtype=np.float64)
            obstime = Time(chunk, format="unix", scale="utc")
            targets = (target, *fallback_targets)
            last_exc = None
            for target_name in targets:
                try:
                    coord = get_horizons_coord(target_name, obstime)
                    break
                except Exception as exc:
                    last_exc = exc
            else:
                raise last_exc

            hci = coord.transform_to(frames.HeliocentricInertial(obstime=obstime))
            positions.append(
                np.stack(
                    [
                        hci.cartesian.x.to_value(u.R_sun),
                        hci.cartesian.y.to_value(u.R_sun),
                        hci.cartesian.z.to_value(u.R_sun),
                    ],
                    axis=-1,
                )
            )
            print(f"Fetched Horizons HCI positions for {min(start + chunk_size, len(time_unix))}/{len(time_unix)} samples", flush=True)
        return np.concatenate(positions, axis=0)

    @staticmethod
    def _position_times_for_range(sample_time_unix, cadence_seconds):
        sample_time_unix = np.asarray(sample_time_unix, dtype=np.float64)
        start = np.floor(np.nanmin(sample_time_unix) / cadence_seconds) * cadence_seconds
        end = np.ceil(np.nanmax(sample_time_unix) / cadence_seconds) * cadence_seconds
        return np.arange(start, end + cadence_seconds, cadence_seconds, dtype=np.float64)

    @staticmethod
    def _interpolate_positions(src_t, src_position, dst_t):
        src_t = np.asarray(src_t, dtype=np.float64)
        src_position = np.asarray(src_position, dtype=np.float64)
        dst_t = np.asarray(dst_t, dtype=np.float64)
        order = np.argsort(src_t, kind="stable")
        src_t = src_t[order]
        src_position = src_position[order]
        if src_t.size < 2:
            raise ValueError("At least two Horizons position samples are required for position interpolation")
        if np.nanmin(dst_t) < src_t[0] or np.nanmax(dst_t) > src_t[-1]:
            raise ValueError("Horizons position samples do not cover the full measurement time range")
        if not np.all(np.isfinite(src_position)):
            raise ValueError("Horizons position samples contain non-finite values")
        position = np.stack([np.interp(dst_t, src_t, src_position[:, i]) for i in range(src_position.shape[1])], axis=-1)
        _, nearest_dt, valid = nearest_values(src_t, src_position, dst_t, np.inf)
        return position, nearest_dt, valid

    def observer_positions_for_samples(self, sample_time_unix, target, cadence_seconds, chunk_size):
        position_time_unix = self._position_times_for_range(sample_time_unix, cadence_seconds)
        position_samples = self._horizons_hci_positions(position_time_unix, target, chunk_size)
        try:
            position, position_dt, position_valid = self._interpolate_positions(position_time_unix, position_samples, sample_time_unix)
        except ValueError as exc:
            print(f"  Horizons support interpolation failed: {exc}")
            position = np.full((sample_time_unix.size, 3), np.nan, dtype=np.float64)
            position_dt = np.full(sample_time_unix.shape, np.nan, dtype=np.float64)
            position_valid = np.zeros(sample_time_unix.shape, dtype=bool)

        missing = ~position_valid | ~np.all(np.isfinite(position), axis=-1)
        if np.any(missing):
            print(f"  querying exact Horizons positions for {int(np.count_nonzero(missing))} missing observer samples")
            position[missing] = self._horizons_hci_positions(sample_time_unix[missing], target, chunk_size)
            position_dt[missing] = 0.0

        still_missing = ~np.all(np.isfinite(position), axis=-1)
        if np.any(still_missing):
            missing_times = sample_time_unix[still_missing]
            examples = ", ".join(iso_time(t) for t in missing_times[:5])
            raise ValueError(
                f"Missing observer locations for {int(np.count_nonzero(still_missing))} samples "
                f"after exact Horizons queries. Examples: {examples}"
            )
        return position, position_dt, position_time_unix.size

    def attach_positions(self, args, arrays):
        time_unix = arrays["time_unix"]
        position_i, position_dt, position_support_count = self.observer_positions_for_samples(
            time_unix,
            args.horizons_target,
            args.position_cadence_seconds,
            args.horizons_chunk_size,
        )
        print(f"  Horizons position support samples: {position_support_count}")
        print(f"  max position interpolation support distance: {np.nanmax(position_dt):.1f} s")
        print("  lost missing observer locations: 0")

        radius = np.linalg.norm(position_i, axis=-1)
        arrays["inertial_coords_r_lat_lon"] = inertial_coords_from_hci(position_i).astype(np.float32)
        arrays["radius_rsun"] = radius.astype(np.float32)
        arrays["r_hat"] = (position_i / np.clip(radius[:, None], 1e-8, None)).astype(np.float32)
        arrays["position_time_delta_seconds"] = position_dt.astype(np.float32)
        arrays["quality"] = np.ones(time_unix.size, dtype=np.float32)
        arrays["position_source"] = np.array("jpl_horizons")
        arrays["position_coordinate_system"] = np.array("SunPy HeliocentricInertial spherical r_lat_lon")
        return position_i

    @staticmethod
    def valid_radial_velocity_mask(velocity_valid, velocity_rtn, velocity_radial_kms, min_abs_kms=50.0, max_abs_kms=1500.0):
        finite_velocity = np.all(np.isfinite(velocity_rtn), axis=-1) & np.isfinite(velocity_radial_kms)
        velocity_in_bounds = (np.abs(velocity_radial_kms) >= min_abs_kms) & (np.abs(velocity_radial_kms) <= max_abs_kms)
        return velocity_valid & finite_velocity & velocity_in_bounds

    @staticmethod
    def print_sample_accounting(title, n_density, n_velocity, dst_t, finite_positive_density, has_velocity_mask, args,
                                velocity_label="native velocity samples", duplicate_count=0):
        print(title)
        print(f"  native density samples: {n_density}")
        print(f"  {velocity_label}: {n_velocity}")
        print(f"  union timestamps: {dst_t.size}")
        if duplicate_count:
            print(f"  discarded duplicate qtn_or_spc timestamps before filtering: {duplicate_count}")
        print_loss_reason(
            f"no valid density within {args.max_density_time_delta_seconds:g} s",
            ~finite_positive_density,
            dst_t,
        )
        print_loss_reason(
            f"no valid velocity within {args.max_velocity_time_delta_seconds:g} s or velocity out of bounds",
            ~has_velocity_mask,
            dst_t,
        )
        print(f"  rows with density target: {int(np.count_nonzero(finite_positive_density))}")
        print(f"  rows with velocity target: {int(np.count_nonzero(has_velocity_mask))}")
        print(f"  kept with density or velocity: {int(np.count_nonzero(finite_positive_density | has_velocity_mask))}")

    def save_product(self, args, arrays, meta, check_horizons=False, horizons_check_points=12):
        position_i = self.attach_positions(args, arrays)
        dst_times = (arrays["time_unix"].astype("int64") * 1_000_000_000).astype("datetime64[ns]")
        arrays["time_iso"] = np.datetime_as_string(dst_times, unit="s")

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(args.out, **arrays)

        meta = dict(meta)
        meta["created_utc"] = datetime.now(timezone.utc).isoformat()
        meta["start"] = args.start
        meta["end"] = args.end
        meta["sample_policy"] = "native density and velocity timestamp union; missing targets stored as NaN"
        meta["max_velocity_time_delta_seconds"] = args.max_velocity_time_delta_seconds
        meta["max_density_time_delta_seconds"] = args.max_density_time_delta_seconds
        meta["position_cadence_seconds"] = args.position_cadence_seconds
        meta["position_policy"] = "Horizons HCI sampled over full range and linearly interpolated to native measurement timestamps"
        meta["n_valid"] = int(arrays["time_unix"].size)
        meta["position"] = {
            "position_source": "jpl_horizons",
            "coordinate_system": "SunPy HeliocentricInertial",
            "horizons_target": args.horizons_target,
        }
        meta["output_units"] = self.output_units(meta.get("output_units", {}))
        meta["density_source_encoding"] = self.density_source_encoding
        self.write_summary(args.out, meta)

        if args.plot_dir is not None:
            self.write_verification_plot(args.out, args.plot_dir)
        if check_horizons:
            self.check_horizons_positions(
                arrays["time_unix"],
                position_i,
                horizons_check_points,
                out_path=args.out,
                plot_dir=args.plot_dir,
            )
        print(f"Wrote {arrays['time_unix'].size} {self.label} samples to {args.out}")

    @staticmethod
    def output_units(extra=None):
        units = {
            "density_cm3": "cm^-3",
            "inertial_coords_r_lat_lon": "[R_sun, rad, rad]",
            "radius_rsun": "R_sun",
            "velocity_rtn_kms": "km/s",
            "velocity_radial_kms": "km/s",
            "density_time_delta_seconds": "s",
            "velocity_time_delta_seconds": "s",
            "position_time_delta_seconds": "s",
        }
        if extra:
            units.update(extra)
        return units

    @staticmethod
    def write_summary(out_path, meta):
        summary_path = os.path.splitext(out_path)[0] + ".json"
        with open(summary_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def write_verification_plot(self, out_path, plot_dir):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = np.load(out_path)
        time_unix = data["time_unix"].astype(np.float64)
        days = (time_unix - np.nanmin(time_unix)) / 86400.0
        density = data["density_cm3"].astype(np.float64)
        velocity = data["velocity_radial_kms"].astype(np.float64)
        radius = data["radius_rsun"].astype(np.float64)
        position = cartesian_from_inertial_coords(data["inertial_coords_r_lat_lon"].astype(np.float64))

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, os.path.splitext(os.path.basename(out_path))[0] + "_qa.png")

        fig, axs = plt.subplots(3, 2, figsize=(13, 10), dpi=150)

        if "raw_density_cm3" in data:
            raw_density = data["raw_density_cm3"].astype(np.float64)
            axs[0, 0].plot(days, raw_density, color="0.45", lw=0.7, label="raw density")
            axs[0, 0].plot(days, density, color="black", lw=0.7, label="electron target")
            axs[0, 0].legend()
        else:
            axs[0, 0].plot(days, density, color="black", lw=0.7)
        axs[0, 0].set_yscale("log")
        axs[0, 0].set_ylabel("density [cm^-3]")
        axs[0, 0].set_xlabel("days from first sample")

        axs[0, 1].plot(days, velocity, color="tab:blue", lw=0.7)
        axs[0, 1].axhline(0.0, color="black", lw=0.6, ls="--")
        axs[0, 1].set_ylabel("radial velocity [km s^-1]")
        axs[0, 1].set_xlabel("days from first sample")

        axs[1, 0].plot(days, radius, color="tab:green", lw=0.7)
        axs[1, 0].set_ylabel(self.radius_ylabel)
        axs[1, 0].set_xlabel("days from first sample")

        if time_unix.size > 1:
            dt = np.diff(time_unix)
            axs[1, 1].hist(dt[np.isfinite(dt)], bins=80, color="0.25")
            axs[1, 1].set_xlabel("sample spacing [s]")
            axs[1, 1].set_ylabel("count")
        else:
            axs[1, 1].text(0.5, 0.5, "single sample", ha="center", va="center")
            axs[1, 1].set_axis_off()

        source = data["density_source"].astype(np.float64)
        axs[2, 0].plot(days, source, ".", color="tab:purple", ms=2)
        source_ticks = sorted(self.density_source_labels)
        if source_ticks:
            axs[2, 0].set_yticks(source_ticks)
            axs[2, 0].set_yticklabels([self.density_source_labels[t] for t in source_ticks])
        axs[2, 0].set_ylabel("density source")
        axs[2, 0].set_xlabel("days from first sample")

        sc = axs[2, 1].scatter(position[:, 0], position[:, 1], c=days, s=4, cmap="viridis")
        axs[2, 1].scatter([0], [0], marker="+", color="orange", s=60)
        axs[2, 1].set_aspect("equal", adjustable="box")
        axs[2, 1].set_xlabel("HCI X [Rsun]")
        axs[2, 1].set_ylabel("HCI Y [Rsun]")
        cbar = fig.colorbar(sc, ax=axs[2, 1])
        cbar.set_label("days from first sample")

        fig.suptitle(f"{self.qa_title}: {time_unix.size} samples", y=0.995)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Wrote {self.label} verification plot to {plot_path}")

    def check_horizons_positions(self, time_unix, position_hci_rsun, n_points, out_path=None, plot_dir=None):
        from sunpy.coordinates import frames, get_horizons_coord

        check_idx = sample_time_indices(time_unix, n_points)
        print(f"Checking {len(check_idx)} {self.label} position samples against JPL Horizons...", flush=True)
        checked_time_unix = []
        checked_prep_position = []
        checked_horizons_position = []
        target = self.default_horizons_target
        fallback_targets = self.fallback_horizons_targets.get(target, ())
        for idx in check_idx:
            time = datetime.fromtimestamp(float(time_unix[idx]), tz=timezone.utc)
            stored_radius, stored_lon, stored_lat = spherical_from_hci_deg(position_hci_rsun[idx])
            last_exc = None
            for target_name in (target, *fallback_targets):
                try:
                    horizons = get_horizons_coord(target_name, time)
                    break
                except Exception as exc:
                    last_exc = exc
            else:
                raise last_exc
            horizons_hci = horizons.transform_to(frames.HeliocentricInertial(obstime=time))
            horizons_radius = horizons_hci.spherical.distance.to_value(u.R_sun)
            horizons_lon = horizons_hci.lon.to_value(u.deg)
            horizons_lat = horizons_hci.lat.to_value(u.deg)
            horizons_position = np.array(
                [
                    horizons_hci.cartesian.x.to_value(u.R_sun),
                    horizons_hci.cartesian.y.to_value(u.R_sun),
                    horizons_hci.cartesian.z.to_value(u.R_sun),
                ],
                dtype=np.float64,
            )
            checked_time_unix.append(float(time_unix[idx]))
            checked_prep_position.append(np.asarray(position_hci_rsun[idx], dtype=np.float64))
            checked_horizons_position.append(horizons_position)
            print(
                f"  {time.isoformat(timespec='minutes')} | "
                f"prep r/lon/lat=({stored_radius[0]:7.2f}, {stored_lon[0]:8.2f}, {stored_lat[0]:7.2f}) | "
                f"Horizons=({horizons_radius:7.2f}, {horizons_lon:8.2f}, {horizons_lat:7.2f}) | "
                f"delta=({stored_radius[0] - horizons_radius:7.2f}, "
                f"{angle_delta_deg(stored_lon[0], horizons_lon):8.2f}, {stored_lat[0] - horizons_lat:7.2f})",
                flush=True,
            )
        if plot_dir is not None and checked_time_unix:
            self.plot_horizons_check(
                out_path,
                plot_dir,
                {
                    "time_unix": np.asarray(checked_time_unix, dtype=np.float64),
                    "prep_position_hci_rsun": np.asarray(checked_prep_position, dtype=np.float64),
                    "horizons_position_hci_rsun": np.asarray(checked_horizons_position, dtype=np.float64),
                },
            )

    def plot_horizons_check(self, out_path, plot_dir, check):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, os.path.splitext(os.path.basename(out_path))[0] + "_horizons.png")

        time_unix = check["time_unix"]
        days = (time_unix - np.nanmin(time_unix)) / 86400.0
        prep_position = check["prep_position_hci_rsun"]
        horizons_position = check["horizons_position_hci_rsun"]
        delta_position = prep_position - horizons_position
        prep_radius, prep_lon, prep_lat = spherical_from_hci_deg(prep_position)
        horizons_radius, horizons_lon, horizons_lat = spherical_from_hci_deg(horizons_position)

        fig, axs = plt.subplots(2, 2, figsize=(12, 9), dpi=150)

        axs[0, 0].plot(prep_position[:, 0], prep_position[:, 1], "o-", ms=3, lw=0.8, label="prepared")
        axs[0, 0].plot(horizons_position[:, 0], horizons_position[:, 1], "x-", ms=4, lw=0.8, label="Horizons")
        axs[0, 0].scatter([0], [0], marker="+", color="orange", s=60)
        axs[0, 0].set_aspect("equal", adjustable="box")
        axs[0, 0].set_xlabel("HCI X [Rsun]")
        axs[0, 0].set_ylabel("HCI Y [Rsun]")
        axs[0, 0].legend(loc="best")

        axs[0, 1].plot(days, prep_radius - horizons_radius, ".-", ms=4, lw=0.8)
        axs[0, 1].axhline(0.0, color="black", lw=0.6)
        axs[0, 1].set_xlabel("days from first checked sample")
        axs[0, 1].set_ylabel("radius delta [Rsun]")

        axs[1, 0].plot(days, angle_delta_deg(prep_lon, horizons_lon), ".-", ms=4, lw=0.8)
        axs[1, 0].axhline(0.0, color="black", lw=0.6)
        axs[1, 0].set_xlabel("days from first checked sample")
        axs[1, 0].set_ylabel("longitude delta [deg]")

        axs[1, 1].plot(days, prep_lat - horizons_lat, ".-", ms=4, lw=0.8, label="latitude")
        axs[1, 1].plot(days, np.linalg.norm(delta_position, axis=-1), ".-", ms=4, lw=0.8, label="|xyz|")
        axs[1, 1].axhline(0.0, color="black", lw=0.6)
        axs[1, 1].set_xlabel("days from first checked sample")
        axs[1, 1].set_ylabel("lat delta [deg] / xyz delta [Rsun]")
        axs[1, 1].legend(loc="best")

        fig.suptitle(f"{self.label} Horizons check: {time_unix.size} samples", y=0.995)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Wrote {self.label} Horizons check plot to {plot_path}")
