import numpy as np
import pytest

from sunerf.evaluation.cme.mass_transport import (
    CM_PER_RSUN, KM_PER_RSUN, PROTON_MASS_G, ambient_ahead_of_front, cone_mask, cone_summary, dbm_distance,
    fit_dbm, fit_speed, integrate_mass, mass_per_radius, physical_drag_parameter, ray_structure, structure_maps,
)


def three_part_profile(radius, r_front=60.0, ambient=1.0, sheath=4.0, cavity=0.3, core=3.0):
    """r^2 rho profile: pile-up behind the front, cavity, dense core; ambient r^-2 fall-off."""
    q = np.full_like(radius, ambient)
    q += (sheath - ambient) * np.exp(-0.5 * ((radius - (r_front - 4)) / 2.0) ** 2)
    q += (cavity - ambient) * np.exp(-0.5 * ((radius - (r_front - 14)) / 3.0) ** 2)
    q += (core - ambient) * np.exp(-0.5 * ((radius - (r_front - 24)) / 2.5) ** 2)
    q[radius > r_front] = ambient
    return q / radius ** 2 * 1e4  # cm^-3 with q in units of 1e4 cm^-3 Rs^2


def test_ray_structure_recovers_front_cavity_and_ejecta():
    radius = np.linspace(30, 120, 361)
    rho = three_part_profile(radius, r_front=60.0)
    structure = ray_structure(rho, radius, dr=0.25, smooth=0.5)
    assert abs(structure.sheath - 56.0) < 1.0
    assert 56.0 < structure.front <= 60.5
    assert abs(structure.cavity - 46.0) < 1.5
    assert abs(structure.ejecta - 36.0) < 1.0
    assert structure.ambient == pytest.approx(1e4, rel=0.05)
    assert structure.excess_mass_per_sr > structure.net_mass_per_sr > 0


def test_ray_structure_returns_nan_for_undisturbed_wind():
    radius = np.linspace(30, 120, 181)
    rho = 5e3 / radius ** 2
    structure = ray_structure(rho, radius)
    assert np.isnan(structure.front) and np.isnan(structure.cavity) and np.isnan(structure.ejecta)


def test_ray_structure_handles_nonuniform_radius_and_nans():
    radius = np.sort(np.concatenate([np.linspace(30, 70, 200), np.linspace(70.5, 120, 60)]))
    rho = three_part_profile(radius, r_front=60.0)
    rho[:3] = np.nan
    structure = ray_structure(rho, radius, dr=0.25, smooth=0.5)
    assert 56.0 < structure.front <= 60.5


def test_mass_per_radius_matches_analytic_shell_mass():
    radius = np.linspace(30, 60, 121)
    latitude = np.deg2rad(np.linspace(-60, 60, 61))
    longitude = np.deg2rad(np.linspace(0, 90, 46))
    rho = np.full((radius.size, latitude.size, longitude.size), 10.0)
    dm_dr = mass_per_radius(rho, radius, latitude, longitude)
    solid_angle = (np.sin(latitude[-1]) - np.sin(latitude[0])) * (longitude[-1] - longitude[0])
    expected = 10.0 * solid_angle * radius ** 2 * CM_PER_RSUN ** 3
    # trapezoidal cos(lat) weights differ from the exact integral by the grid discretization
    np.testing.assert_allclose(dm_dr, expected, rtol=2e-3)
    mass = integrate_mass(dm_dr, radius, 40, 50)
    expected_mass = 10.0 * solid_angle * (50 ** 3 - 40 ** 3) / 3 * CM_PER_RSUN ** 3
    np.testing.assert_allclose(mass, expected_mass, rtol=5e-3)


def test_cone_mask_selects_rays_around_target():
    latitude = np.deg2rad(np.arange(-60, 61, 5))
    longitude = np.deg2rad(np.arange(0, 360, 5))
    cone = cone_mask(latitude, longitude, np.deg2rad(0), np.deg2rad(135), np.deg2rad(20))
    assert cone[np.argmin(np.abs(np.rad2deg(latitude))), np.argmin(np.abs(np.rad2deg(longitude) - 135))]
    assert not cone[np.argmin(np.abs(np.rad2deg(latitude))), np.argmin(np.abs(np.rad2deg(longitude) - 180))]
    # wrap-around
    cone = cone_mask(latitude, longitude, 0.0, np.deg2rad(2), np.deg2rad(10))
    assert cone[np.argmin(np.abs(latitude)), np.argmin(np.abs(np.rad2deg(longitude) - 355))]


def test_structure_maps_and_cone_summary():
    radius = np.linspace(30, 120, 181)
    latitude = np.deg2rad(np.arange(-40, 41, 4))
    longitude = np.deg2rad(np.arange(100, 171, 4))
    cone = cone_mask(latitude, longitude, 0.0, np.deg2rad(135), np.deg2rad(25))
    rho = np.empty((radius.size, latitude.size, longitude.size))
    fronts = np.empty((latitude.size, longitude.size))
    for i, lat in enumerate(latitude):
        for j, lon in enumerate(longitude):
            separation = np.hypot(lat, lon - np.deg2rad(135))
            fronts[i, j] = 70 - 40 * separation  # front recedes away from the axis
            rho[:, i, j] = three_part_profile(radius, r_front=fronts[i, j])
    maps = structure_maps(rho, radius, angular_mask=cone, dr=0.5, smooth=0.5)
    assert np.isnan(maps['front'][~cone]).all()
    summary = cone_summary(maps, latitude, longitude, cone)
    assert summary['detection_fraction'] > 0.9
    assert abs(summary['front_apex'] - np.nanpercentile(np.where(cone, fronts, np.nan), 90)) < 2.0
    assert summary['excess_mass'] > 0
    ambient = ambient_ahead_of_front(rho, radius, maps['front'], cone)
    r_ref = summary['front_median'] + 2.5
    assert ambient == pytest.approx(1e4 / r_ref ** 2, rel=0.3)


def test_dbm_fit_recovers_drag_parameter():
    times_s = np.arange(0, 20 * 3600, 1200.0)
    gamma, w, v0, r0 = 2e-7, 400.0, 1200.0, 30 * KM_PER_RSUN
    radius = dbm_distance(times_s, r0, v0, w, gamma) / KM_PER_RSUN
    fit = fit_dbm(times_s, radius, w_kms=w)
    assert fit['gamma'] == pytest.approx(gamma, rel=0.05)
    assert fit['v0'] == pytest.approx(v0, rel=0.02)
    fit_free = fit_dbm(times_s, radius)
    np.testing.assert_allclose(fit_free['radius_fit'], radius, rtol=1e-3)
    assert fit_speed(times_s, radius) < v0


def test_physical_drag_parameter_units():
    # gamma = rho_sw A / M: 1e3 cm^-3 protons, cross-section of 1 Rs^2, mass 1e15 g
    mass_electrons = 1e15 / PROTON_MASS_G
    gamma = physical_drag_parameter(1e3, 1.0, mass_electrons)
    expected = 1e3 * PROTON_MASS_G * CM_PER_RSUN ** 2 / 1e15 * 1e5
    assert gamma == pytest.approx(expected)
