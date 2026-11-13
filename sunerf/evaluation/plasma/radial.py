"""Radial columns of the plasma model: points on heliographic footpoints and their integrals."""
import numpy as np
from astropy import units as u

from sunerf.evaluation.util import convert_spherical_to_cartesian

R_SUN_CM = (1 * u.R_sun).to_value(u.cm)
# model points evaluated per call of the loader; bounds the memory of dense radial grids
POINTS_PER_CHUNK = 2 ** 22


def rendered_shell(loader):
    """Inner and outer radius [R_sun] of the shell the renderer samples (sampling.min_distance and
    max_distance). Below the inner radius the model is never rendered, so its plasma state is unconstrained."""
    sampler = loader.rendering.sampler
    if not hasattr(sampler, 'min_distance'):
        raise ValueError(f'{type(sampler).__name__} defines no rendered shell; pass the radial range explicitly')
    return (float(sampler.min_distance) * loader.Rs_per_ds, float(sampler.max_distance) * loader.Rs_per_ds)


def radial_columns(lat_rad, lon_rad, radii):
    """Cartesian points (..., n_radius, 3) in solar radii of the radial columns above the Carrington
    footpoints ``lat_rad``/``lon_rad`` (any shape, NaN where no footpoint exists)."""
    lat = np.asarray(lat_rad, dtype=float)[..., None]
    lon = np.asarray(lon_rad, dtype=float)[..., None]
    r = np.asarray(radii, dtype=float)
    x, y, z = convert_spherical_to_cartesian(r, lat, lon)
    return np.stack(np.broadcast_arrays(x, y, z), -1)


def trapezoid_weights(radii):
    """Trapezoid quadrature weights of the samples ``radii``: half an interval at each end."""
    dr = np.diff(np.asarray(radii, dtype=float))
    weights = np.zeros(len(dr) + 1)
    weights[:-1] += 0.5 * dr
    weights[1:] += 0.5 * dr
    return weights


def integrate_columns(loader, points, radii, time, bins, batch_size, cool_column=False):
    """Radial integrals of the columns ``points`` (..., n_radius, 3) at ``time``.

    Returns the emission measure n_e^2 dr [cm^-5] of every (lo, hi) temperature bin in ``bins`` (log10 K,
    the highest upper edge inclusive) and the column electron density n_e dr [cm^-2]; NaN columns stay NaN.
    Each sample enters its bin with its trapezoid weight, so the temperature resolution of the bins along
    the column is the radial sample spacing. With ``cool_column``, the cool-absorber hydrogen column
    n_H dr [cm^-2] is returned third (None for a model without a cool absorber).
    """
    radii = np.asarray(radii, dtype=float)
    columns = points.reshape(-1, len(radii), 3)
    n_e = np.full(columns.shape[:-1], np.nan, dtype=np.float32)
    log_T = np.full(columns.shape[:-1], np.nan, dtype=np.float32)
    n_cool = np.full(columns.shape[:-1], np.nan, dtype=np.float32) if cool_column else None
    normalized_time = loader.normalize_datetime(time)
    chunk = max(1, POINTS_PER_CHUNK // len(radii))
    for start in range(0, len(columns), chunk):
        block = columns[start:start + chunk] / float(loader.Rs_per_ds)
        if not np.isfinite(block).all(-1).any():
            continue  # no footpoint in this chunk (e.g. off-limb rows of a disk image)
        coords = np.concatenate([block, np.full(block.shape[:-1] + (1,), normalized_time)], -1)
        out = loader.load_coords(coords.astype(np.float32), batch_size=batch_size)
        n_e[start:start + chunk] = np.squeeze(out['total_ne'], -1)
        log_T[start:start + chunk] = np.squeeze(out['mean_log_T'], -1)
        if n_cool is not None:
            if 'cool_hydrogen_density' not in out:
                n_cool = None
            else:
                n_cool[start:start + chunk] = np.squeeze(out['cool_hydrogen_density'], -1)
    n_e = n_e.reshape(points.shape[:-1])
    log_T = log_T.reshape(points.shape[:-1])

    dr_cm = trapezoid_weights(radii) * R_SUN_CM
    column_density = (n_e * dr_cm).sum(-1)
    em = n_e.astype(float) ** 2 * dr_cm
    top = max(hi for _, hi in bins)
    maps = []
    for lo, hi in bins:
        inside = (log_T >= lo) & ((log_T < hi) | ((hi >= top) & (log_T <= hi)))
        maps.append(np.where(inside, em, 0.0).sum(-1))
    # columns without a footpoint (NaN model output) are NaN, not zero
    missing = ~np.isfinite(n_e).all(-1)
    column_density[missing] = np.nan
    for em_map in maps:
        em_map[missing] = np.nan
    if not cool_column:
        return maps, column_density
    cool_density = None
    if n_cool is not None:
        cool_density = (n_cool.reshape(points.shape[:-1]) * dr_cm).sum(-1)
        cool_density[missing] = np.nan
    return maps, column_density, cool_density
