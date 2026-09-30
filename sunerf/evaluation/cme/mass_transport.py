"""Mass transport and three-part structure diagnostics for CME density cubes.

All functions work on density sampled on a spherical grid ``rho[r, lat, lon]``
in electrons per cm^3 with ``radius`` in solar radii and ``latitude`` /
``longitude`` in radians.  Nothing here subtracts a pre-event background: the
ambient fall-off is removed by the radially compensated density ``r^2 rho``,
and the ambient reference of every ray is taken ahead of the detected front in
the same frame.
"""
from dataclasses import dataclass

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

CM_PER_RSUN = (1 * u.R_sun).to_value(u.cm)
KM_PER_RSUN = (1 * u.R_sun).to_value(u.km)
PROTON_MASS_G = const.m_p.to_value(u.g)


##########################################################
# mass distribution

def axis_widths(x):
    """Cell widths of a 1D axis (midpoints between neighbours, half cells at the edges)."""
    if x.size < 2:
        return np.ones_like(x, dtype=float)
    edges = np.concatenate([[x[0]], 0.5 * (x[1:] + x[:-1]), [x[-1]]])
    return np.abs(np.diff(edges))


def solid_angle_weights(latitude, longitude):
    """Return ``cos(lat) dlat dlon`` for every grid cell."""
    return np.cos(latitude)[:, None] * axis_widths(latitude)[:, None] * axis_widths(longitude)[None, :]


def mass_per_radius(rho, radius, latitude, longitude, angular_mask=None):
    """Return ``dM/dr`` in electrons per solar radius for every radial shell.

    ``angular_mask[lat, lon]`` restricts the integration to a sector.
    """
    weights = solid_angle_weights(latitude, longitude)
    if angular_mask is not None:
        weights = weights * angular_mask
    column = np.nansum(rho * weights[None], axis=(1, 2))
    return column * radius ** 2 * CM_PER_RSUN ** 3


def integrate_mass(mass_per_r, radius, r_min=None, r_max=None):
    """Integrate ``dM/dr`` between two radii (electrons)."""
    mask = np.ones_like(radius, dtype=bool)
    if r_min is not None:
        mask &= radius >= r_min
    if r_max is not None:
        mask &= radius <= r_max
    if mask.sum() < 2:
        return np.nan
    return np.trapezoid(mass_per_r[mask], radius[mask])


def cone_mask(latitude, longitude, target_latitude, target_longitude, half_angle):
    """Boolean ``[lat, lon]`` mask of rays within ``half_angle`` of the target direction (radians)."""
    lat = latitude[:, None]
    lon = longitude[None, :]
    cos_sep = (np.sin(lat) * np.sin(target_latitude)
               + np.cos(lat) * np.cos(target_latitude) * np.cos(lon - target_longitude))
    return np.arccos(np.clip(cos_sep, -1, 1)) <= half_angle


##########################################################
# three-part structure along a ray

@dataclass
class RayStructure:
    front: float = np.nan
    sheath: float = np.nan
    cavity: float = np.nan
    ejecta: float = np.nan
    ambient: float = np.nan  # compensated density r^2 rho ahead of the front [cm^-3 Rs^2]
    excess_mass_per_sr: float = np.nan  # positive excess of r^2 rho inside the front [cm^-3 Rs^3]
    net_mass_per_sr: float = np.nan


def compensated_density(rho, radius):
    """Return ``r^2 rho`` which is flat for an undisturbed radial outflow."""
    return rho * radius[:, None, None] ** 2


def _uniform_ray(radius, values, dr):
    r_uniform = np.arange(radius.min(), radius.max() + 0.5 * dr, dr)
    return r_uniform, np.interp(r_uniform, radius, values)


def _vertex_offset(values, index):
    """Offset of the extremum of the parabola through ``values[index - 1:index + 2]`` in grid steps."""
    if index == 0 or index == len(values) - 1:
        return 0.0
    left, center, right = values[index - 1:index + 2]
    curvature = left - 2 * center + right
    if curvature == 0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


def ray_structure(rho_ray, radius, dr=0.5, smooth=1.0, min_prominence=0.15, min_gradient=0.05,
                  ambient_offset=1.0, ambient_window=3.0, r_search_min=None):
    """Locate front, sheath, cavity and ejecta along one radial density profile.

    Parameters
    ----------
    rho_ray: density along the ray (cm^-3), ``radius`` in R_sun (any spacing).
    dr: uniform resampling step [R_sun].
    smooth: Gaussian smoothing of ``log10(r^2 rho)`` [R_sun].
    min_prominence: peak prominence in dex required for a density enhancement.
    min_gradient: steepest outward decrease in dex per R_sun required at the front.
    ambient_offset, ambient_window: the ambient level is the median of the
        profile between ``front + offset`` and ``front + offset + window`` [R_sun].
    r_search_min: ignore structure below this radius (inner boundary effects).

    The outermost enhancement is the sheath (pile-up); the front is the point of
    steepest outward decrease between the sheath peak and its outer base.  The
    cavity is the minimum between the sheath and the next enhancement inside,
    which is the ejecta.
    """
    valid = np.isfinite(rho_ray) & (rho_ray > 0)
    result = RayStructure()
    if valid.sum() < 4:
        return result
    r_uni, q = _uniform_ray(radius[valid], rho_ray[valid] * radius[valid] ** 2, dr)
    log_q = np.log10(q)
    if smooth > 0:
        log_q = gaussian_filter1d(log_q, smooth / dr, mode='nearest')
    if r_search_min is not None:
        search = r_uni >= r_search_min
        if search.sum() < 4:
            return result
        r_uni, log_q = r_uni[search], log_q[search]

    peaks, props = find_peaks(log_q, prominence=min_prominence)
    if peaks.size == 0:
        return result
    order = np.argsort(peaks)[::-1]  # outermost first
    peaks, right_bases = peaks[order], props['right_bases'][order]

    gradient = np.gradient(log_q, r_uni)
    sheath_idx, base_idx = peaks[0], right_bases[0]
    if base_idx <= sheath_idx + 1:
        return result
    front_idx = sheath_idx + np.argmin(gradient[sheath_idx:base_idx + 1])
    if -gradient[front_idx] < min_gradient:
        return result
    # refine: the front is where the profile has dropped to the ambient level
    # (halfway between sheath peak and outer base in log space)
    half_level = 0.5 * (log_q[sheath_idx] + log_q[base_idx])
    below = np.nonzero(log_q[sheath_idx:base_idx + 1] <= half_level)[0]
    front = r_uni[front_idx]
    if below.size:
        front_idx = sheath_idx + below[0]
        front = r_uni[front_idx]
        if front_idx > sheath_idx:
            # position of the half level between the two grid points that enclose it
            drop = log_q[front_idx - 1] - log_q[front_idx]
            front = r_uni[front_idx - 1] + dr * (log_q[front_idx - 1] - half_level) / drop

    result.sheath = r_uni[sheath_idx] + dr * _vertex_offset(log_q, sheath_idx)
    result.front = front
    ambient_mask = (r_uni >= result.front + ambient_offset) & (r_uni <= result.front + ambient_offset + ambient_window)
    if not ambient_mask.any():
        # no ambient reference ahead of the front (front at the outer grid edge): no detection
        return RayStructure()
    result.ambient = 10 ** np.median(log_q[ambient_mask])

    inner = r_uni <= result.front
    excess = 10 ** log_q[inner] - result.ambient
    result.net_mass_per_sr = np.trapezoid(excess, r_uni[inner])
    result.excess_mass_per_sr = np.trapezoid(np.clip(excess, 0, None), r_uni[inner])

    if peaks.size > 1:
        ejecta_idx = peaks[1]
        cavity_idx = ejecta_idx + np.argmin(log_q[ejecta_idx:sheath_idx + 1])
        if ejecta_idx < cavity_idx < sheath_idx:
            result.ejecta = r_uni[ejecta_idx]
            result.cavity = r_uni[cavity_idx]
    return result


STRUCTURE_FIELDS = ('front', 'sheath', 'cavity', 'ejecta', 'ambient', 'excess_mass_per_sr', 'net_mass_per_sr')


def structure_maps(rho, radius, angular_mask=None, **kwargs):
    """Apply :func:`ray_structure` to every ray; returns ``{field: array[lat, lon]}``."""
    n_lat, n_lon = rho.shape[1:]
    maps = {field: np.full((n_lat, n_lon), np.nan) for field in STRUCTURE_FIELDS}
    for i in range(n_lat):
        for j in range(n_lon):
            if angular_mask is not None and not angular_mask[i, j]:
                continue
            structure = ray_structure(rho[:, i, j], radius, **kwargs)
            for field in STRUCTURE_FIELDS:
                maps[field][i, j] = getattr(structure, field)
    return maps


def cone_summary(maps, latitude, longitude, cone, apex_percentile=90):
    """Reduce structure maps over a cone to scalar kinematic quantities.

    Returns radii of the front apex (upper percentile over the cone), the median
    front / sheath / cavity / ejecta radii, the detection fraction, the solid
    angle of rays with a detected front [sr], the positive excess mass inside the
    front [electrons] and its net counterpart, and the median ambient
    compensated density ahead of the front [cm^-3 Rs^2].
    """
    weights = solid_angle_weights(latitude, longitude)
    front = np.where(cone, maps['front'], np.nan)
    detected = np.isfinite(front)
    summary = {
        'front_apex': np.nanpercentile(front, apex_percentile) if detected.any() else np.nan,
        'detection_fraction': detected.sum() / max(cone.sum(), 1),
        'front_solid_angle': float((weights * detected).sum()),
        'excess_mass': float(np.nansum(np.where(detected, maps['excess_mass_per_sr'], 0.0) * weights)
                             * CM_PER_RSUN ** 3),
        'net_mass': float(np.nansum(np.where(detected, maps['net_mass_per_sr'], 0.0) * weights)
                          * CM_PER_RSUN ** 3),
    }
    for field in ('front', 'sheath', 'cavity', 'ejecta', 'ambient'):
        values = np.where(cone, maps[field], np.nan)
        summary[f'{field}_median'] = np.nanmedian(values) if np.isfinite(values).any() else np.nan
    return summary


def ambient_ahead_of_front(field, radius, front_map, cone, offset=(1.0, 4.0)):
    """Median of ``field[r, lat, lon]`` in the shell ``front + offset`` along every cone ray with a front."""
    samples = []
    for i, j in zip(*np.nonzero(cone & np.isfinite(front_map))):
        r_front = front_map[i, j]
        mask = (radius >= r_front + offset[0]) & (radius <= r_front + offset[1])
        if mask.any():
            samples.append(np.nanmedian(field[mask, i, j]))
    return np.nanmedian(samples) if samples else np.nan


##########################################################
# kinematics and drag

def fit_speed(times_s, radius_rsun):
    """Linear least-squares speed in km/s of a height-time track (NaNs ignored)."""
    mask = np.isfinite(radius_rsun)
    if mask.sum() < 2:
        return np.nan
    slope = np.polyfit(times_s[mask], radius_rsun[mask], 1)[0]
    return slope * KM_PER_RSUN


def fit_constant_acceleration(times_s, radius_rsun, reference_radius):
    """Constant-acceleration fit of a height-time track (NaNs ignored).

    Returns the speed in km/s at ``reference_radius`` [R_sun] and the
    acceleration in m/s^2; the speed is NaN when the track does not cross the
    reference radius.
    """
    mask = np.isfinite(radius_rsun)
    if mask.sum() < 4:
        return np.nan, np.nan
    half_acceleration, speed, offset = np.polyfit(times_s[mask], radius_rsun[mask], 2)
    acceleration = 2 * half_acceleration * KM_PER_RSUN * 1e3
    if not radius_rsun[mask].min() <= reference_radius <= radius_rsun[mask].max():
        return np.nan, acceleration
    # speed at the time the fit reaches the reference radius inside the tracked interval
    roots = np.roots([half_acceleration, speed, offset - reference_radius])
    roots = roots[np.isreal(roots)].real
    roots = roots[(roots >= times_s[mask].min()) & (roots <= times_s[mask].max())]
    if not roots.size:
        return np.nan, acceleration
    return (speed + 2 * half_acceleration * roots[0]) * KM_PER_RSUN, acceleration


def local_kinematics(times_s, radius_rsun, window_s, min_points=5):
    """Speed [km/s] and acceleration [m/s^2] from quadratic fits in a sliding time window.

    Every sample is fitted with the finite samples within ``window_s / 2`` on
    both sides; samples without ``min_points`` or without neighbours on both
    sides stay NaN.
    """
    speed = np.full(len(times_s), np.nan)
    acceleration = np.full(len(times_s), np.nan)
    finite = np.isfinite(radius_rsun)
    for index in np.flatnonzero(finite):
        dt = times_s - times_s[index]
        window = finite & (np.abs(dt) <= 0.5 * window_s)
        if window.sum() < min_points or not (dt[window] < 0).any() or not (dt[window] > 0).any():
            continue
        half_acceleration, slope, _ = np.polyfit(dt[window], radius_rsun[window], 2)
        speed[index] = slope * KM_PER_RSUN
        acceleration[index] = 2 * half_acceleration * KM_PER_RSUN * 1e3
    return speed, acceleration


def finite_difference_speed(times_s, radius_rsun):
    """Speed in km/s from central differences of a height-time track."""
    mask = np.isfinite(radius_rsun)
    speed = np.full_like(radius_rsun, np.nan, dtype=float)
    if mask.sum() >= 2:
        speed[mask] = np.gradient(radius_rsun[mask], times_s[mask]) * KM_PER_RSUN
    return speed


def dbm_distance(t_s, r0_km, v0_kms, w_kms, gamma_per_km):
    """Drag-based model (Vrsnak et al. 2013) heliocentric distance in km.

    ``dv/dt = -gamma (v - w) |v - w|`` with constant ambient wind ``w`` and drag
    parameter ``gamma``.
    """
    dv = v0_kms - w_kms
    sign = np.sign(dv) if dv != 0 else 1.0
    arg = 1 + sign * gamma_per_km * dv * t_s
    arg = np.clip(arg, 1e-12, None)
    return r0_km + w_kms * t_s + sign / max(gamma_per_km, 1e-30) * np.log(arg)


def fit_dbm(times_s, radius_rsun, w_kms=None, gamma_init=1e-7):
    """Fit the drag-based model to a height-time track.

    Returns ``{'r0', 'v0', 'w', 'gamma', 'radius_fit'}`` with distances in
    R_sun, speeds in km/s and ``gamma`` in 1/km.  With ``w_kms`` given, only the
    initial speed and the drag parameter are free.
    """
    mask = np.isfinite(radius_rsun)
    if mask.sum() < 3:
        return {'r0': np.nan, 'v0': np.nan, 'w': np.nan, 'gamma': np.nan,
                'radius_fit': np.full_like(radius_rsun, np.nan, dtype=float)}
    t = times_s[mask] - times_s[mask][0]
    r_km = radius_rsun[mask] * KM_PER_RSUN
    v_guess = fit_speed(times_s, radius_rsun)
    if w_kms is None:
        def model(t, r0, v0, w, log_gamma):
            return dbm_distance(t, r0, v0, w, 10 ** log_gamma)
        p0 = [r_km[0], v_guess, 0.5 * v_guess, np.log10(gamma_init)]
        bounds = ([r_km[0] - 5 * KM_PER_RSUN, 0, 0, -10], [r_km[0] + 5 * KM_PER_RSUN, 6000, 3000, -4])
    else:
        def model(t, r0, v0, log_gamma):
            return dbm_distance(t, r0, v0, w_kms, 10 ** log_gamma)
        p0 = [r_km[0], v_guess, np.log10(gamma_init)]
        bounds = ([r_km[0] - 5 * KM_PER_RSUN, 0, -10], [r_km[0] + 5 * KM_PER_RSUN, 6000, -4])
    try:
        params, _ = curve_fit(model, t, r_km, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError):
        return {'r0': np.nan, 'v0': np.nan, 'w': np.nan if w_kms is None else w_kms, 'gamma': np.nan,
                'radius_fit': np.full_like(radius_rsun, np.nan, dtype=float)}
    if w_kms is None:
        r0, v0, w, log_gamma = params
    else:
        r0, v0, log_gamma = params
        w = w_kms
    radius_fit = np.full_like(radius_rsun, np.nan, dtype=float)
    radius_fit[mask] = dbm_distance(t, r0, v0, w, 10 ** log_gamma) / KM_PER_RSUN
    return {'r0': r0 / KM_PER_RSUN, 'v0': v0, 'w': w, 'gamma': 10 ** log_gamma, 'radius_fit': radius_fit}


def physical_drag_parameter(ambient_density_cm3, cross_section_rsun2, cme_mass_electrons, drag_coefficient=1.0):
    """``gamma = C_d rho_sw A / M`` in 1/km for a hydrogen plasma.

    ``ambient_density_cm3`` is the electron density ahead of the front, the
    cross section is in R_sun^2 and the CME mass in electrons.
    """
    rho_sw = ambient_density_cm3 * PROTON_MASS_G  # g cm^-3
    area_cm2 = cross_section_rsun2 * CM_PER_RSUN ** 2
    mass_g = cme_mass_electrons * PROTON_MASS_G
    with np.errstate(divide='ignore', invalid='ignore'):
        gamma_per_cm = drag_coefficient * rho_sw * area_cm2 / mass_g
    return gamma_per_cm * 1e5
