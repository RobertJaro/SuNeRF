from datetime import datetime, timezone

import numpy as np
from astropy import units as u
from dateutil.parser import parse

from sunerf.data.date_util import normalize_datetime
from sunerf.data.dataset import TensorsDataset
from sunerf.train.coordinate_transformation import spherical_to_cartesian

INERTIAL_COORDS_KEY = "inertial_coords_r_lat_lon"


def _load_inertial_position(data):
    if INERTIAL_COORDS_KEY not in data:
        raise KeyError(f"In-situ data must contain '{INERTIAL_COORDS_KEY}'.")

    inertial_coords = data[INERTIAL_COORDS_KEY].astype(np.float32)
    position = spherical_to_cartesian(inertial_coords, np).astype(np.float32)

    radius_rsun = inertial_coords[:, 0].astype(np.float32)
    r_hat = position / np.clip(radius_rsun[:, None], 1e-8, None)
    return inertial_coords, position, radius_rsun, r_hat


class InSituDataset(TensorsDataset):
    def __init__(
            self,
            data_path,
            Rs_per_ds,
            seconds_per_dt,
            ref_date,
            work_directory,
            drho_cm3,
            batch_size=4096,
            ds_key="insitu",
            instrument_key=None,
            max_distance=np.inf,
            time_range=None,
            shuffle=True,
            filter_nans=True,
            **kwargs):
        instrument_key = ds_key if instrument_key is None else instrument_key
        data = np.load(data_path)

        time_unix = data["time_unix"].astype(np.float64)
        time_mask = np.ones(time_unix.shape, dtype=bool)
        if time_range is not None:
            if isinstance(time_range, (str, datetime)):
                time_range = [time_range]
            parsed_times = [parse(t) if isinstance(t, str) else t for t in time_range]
            parsed_times = [
                t.astimezone(timezone.utc).replace(tzinfo=None)
                if t.tzinfo is not None else t
                for t in parsed_times
            ]
            if len(parsed_times) == 1:
                target_unix = parsed_times[0].replace(tzinfo=timezone.utc).timestamp()
                closest_idx = int(np.argmin(np.abs(time_unix - target_unix)))
                time_mask[:] = False
                time_mask[closest_idx] = True
            elif len(parsed_times) == 2:
                start, end = parsed_times
                if start > end:
                    raise ValueError("time_range start must be earlier than or equal to end.")
                start_unix = start.replace(tzinfo=timezone.utc).timestamp()
                end_unix = end.replace(tzinfo=timezone.utc).timestamp()
                time_mask = (time_unix >= start_unix) & (time_unix <= end_unix)
            else:
                raise ValueError("time_range must contain one value for closest-date selection "
                                 "or two values for start/end filtering.")
            if not np.any(time_mask):
                raise ValueError(f"No in-situ samples found in time_range {time_range}.")
        if ref_date is None:
            ref_date = datetime.fromtimestamp(
                float(np.nanmin(time_unix)), timezone.utc
            ).replace(tzinfo=None)
        elif ref_date.tzinfo is not None:
            ref_date = ref_date.astimezone(timezone.utc).replace(tzinfo=None)
        self.ref_date = ref_date

        times = np.array([
            datetime.fromtimestamp(float(t), timezone.utc).replace(tzinfo=None)
            for t in time_unix
        ])
        time_norm = np.array([normalize_datetime(t, seconds_per_dt, ref_date) for t in times], dtype=np.float32)
        time_days = ((time_unix - np.nanmin(time_unix)) / 86400.0).astype(np.float32)

        inertial_coords, position, radius_rsun, r_hat = _load_inertial_position(data)
        distance_mask = radius_rsun <= float(max_distance)
        coords_xyz = position / float(Rs_per_ds)
        query_points = np.concatenate([coords_xyz, time_norm[:, None]], axis=-1).astype(np.float32)

        density_cm3 = data["density_cm3"].astype(np.float32)
        density = (density_cm3 / float(drho_cm3)).astype(np.float32)

        velocity_scale = float((1.0 * u.km / u.s).to_value(u.R_sun / u.s) / Rs_per_ds * seconds_per_dt)
        velocity_radial = (data["velocity_radial_kms"].astype(np.float32) * velocity_scale).astype(np.float32)
        if "has_density" in data:
            has_density = data["has_density"].astype(np.float32)
        else:
            has_density = np.isfinite(density_cm3).astype(np.float32)
        if "has_velocity" in data:
            has_velocity = data["has_velocity"].astype(np.float32)
        else:
            has_velocity = np.isfinite(data["velocity_radial_kms"].astype(np.float32)).astype(np.float32)
        target_mask = (has_density > 0.5) | (has_velocity > 0.5)
        finite_position_mask = np.all(np.isfinite(position), axis=-1) & np.isfinite(radius_rsun)
        distance_mask = distance_mask & target_mask & finite_position_mask & time_mask

        tensors = {
            "query_points": query_points[distance_mask],
            "density": density[distance_mask, None],
            "density_cm3": density_cm3[distance_mask, None],
            "velocity_radial": velocity_radial[distance_mask, None],
            "velocity_radial_kms": data["velocity_radial_kms"].astype(np.float32)[distance_mask, None],
            "r_hat": r_hat.astype(np.float32)[distance_mask],
            "radius_rsun": radius_rsun[distance_mask, None],
            INERTIAL_COORDS_KEY: inertial_coords[distance_mask],
            "time_norm": time_norm[distance_mask, None],
            "time_days": time_days[distance_mask, None],
            "density_source": data["density_source"].astype(np.float32)[distance_mask, None],
        }
        tensors["has_density"] = has_density[distance_mask, None]
        tensors["has_velocity"] = has_velocity[distance_mask, None]

        # Density and radial velocity are independently sampled observables.  A
        # missing value in one target must not make an otherwise valid row
        # unusable, while geometry and the target selected by its availability
        # flag still need to be finite.
        finite_geometry = (
            np.all(np.isfinite(tensors["query_points"]), axis=-1)
            & np.all(np.isfinite(tensors["r_hat"]), axis=-1)
            & np.all(np.isfinite(tensors[INERTIAL_COORDS_KEY]), axis=-1)
            & np.isfinite(tensors["radius_rsun"][:, 0])
            & np.isfinite(tensors["time_norm"][:, 0])
            & np.isfinite(tensors["time_days"][:, 0])
        )
        finite_density_target = (
            (tensors["has_density"][:, 0] > 0.5)
            & np.isfinite(tensors["density"][:, 0])
            & np.isfinite(tensors["density_cm3"][:, 0])
        )
        finite_velocity_target = (
            (tensors["has_velocity"][:, 0] > 0.5)
            & np.isfinite(tensors["velocity_radial"][:, 0])
            & np.isfinite(tensors["velocity_radial_kms"][:, 0])
        )
        valid_mask = finite_geometry & (finite_density_target | finite_velocity_target)

        super().__init__(
            tensors,
            work_directory=work_directory,
            filter_nans=filter_nans,
            valid_mask=valid_mask,
            shuffle=shuffle,
            ds_name=ds_key,
            batch_size=batch_size,
            instrument=instrument_key,
        )

        self.data_path = data_path
        self.instrument_key = instrument_key
        self.max_distance = float(max_distance)
        self.drho_cm3 = float(drho_cm3)


class PSPDataset(InSituDataset):
    pass


class SolarOrbiterDataset(InSituDataset):
    pass
