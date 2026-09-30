from datetime import datetime

import numpy as np

from sunerf.evaluation.cme.density_series import DensityFrame, load_export_frame
from sunerf.evaluation.cme.export_cme_series import RADIUS_GRID, beam_profile, reduce_frame
from sunerf.evaluation.cme.export_layout import coordinates_file, density_file
from sunerf.evaluation.psi.common import leading_edge, outermost_above
from sunerf.evaluation.psi.plot_jmap import projected_height


def _shell_frame(shell_radius, target_latitude, target_longitude):
    """Radial outflow with a dense shell inside 25 deg of the target direction."""
    radius = np.linspace(2.0, 30.0, 113)
    latitude = np.deg2rad(np.arange(-88.0, 89.0, 2.0))
    longitude = np.deg2rad(np.arange(1.0, 360.0, 2.0))
    r, lat, lon = np.meshgrid(radius, latitude, longitude, indexing='ij')
    cos_separation = (np.sin(lat) * np.sin(target_latitude)
                      + np.cos(lat) * np.cos(target_latitude) * np.cos(lon - target_longitude))
    inside = np.arccos(np.clip(cos_separation, -1, 1)) < np.deg2rad(25.0)
    shell = 4.0 * np.exp(-0.5 * ((r - shell_radius) / 1.5) ** 2) * inside
    rho = 3e5 * (1.0 + shell) / r ** 2
    return DensityFrame(rho.astype(np.float32), radius, latitude, longitude, datetime(2021, 10, 28, 18))


def test_reduce_frame_locates_shell_and_direction():
    target = np.deg2rad(-20.0), np.deg2rad(245.0)
    record = reduce_frame(_shell_frame(15.0, *target), *target, np.deg2rad(30.0))

    assert abs(record['body_peak_median'] - 15.0) < 0.5
    assert 15.0 < record['body_edge_median'] < 19.0
    assert abs(RADIUS_GRID[np.argmax(record['mass_per_radius'])] - 15.0) < 0.5
    assert abs(np.rad2deg(record['direction_latitude']) + 20.0) < 2.0
    assert abs(np.rad2deg(record['direction_longitude']) - 245.0) < 2.0
    assert record['detection_fraction'] > 0.5
    assert abs(record['beam_body_peak'] - 15.0) < 0.5
    assert abs(RADIUS_GRID[np.argmax(record['beam_density'])] - 15.0) < 0.5


def test_mass_per_radius_of_radial_outflow_is_flat():
    target = np.deg2rad(10.0), np.deg2rad(100.0)
    frame = _shell_frame(15.0, *target)
    frame.rho = (3e5 / frame.radius[:, None, None] ** 2 * np.ones_like(frame.rho)).astype(np.float32)
    record = reduce_frame(frame, *target, np.deg2rad(30.0))

    mass = record['mass_per_radius']
    assert np.allclose(mass, mass.mean(), rtol=1e-3)
    assert np.isnan(record['body_edge_apex'])


def test_export_frame_round_trip(tmp_path):
    frame = _shell_frame(12.0, 0.0, 1.0)
    for file in (density_file(tmp_path, 'rho000070'), coordinates_file(tmp_path, 'rho000070')):
        (tmp_path / file).parent.mkdir(exist_ok=True)
    np.save(density_file(tmp_path, 'rho000070'), frame.rho)
    np.savez(coordinates_file(tmp_path, 'rho000070'), radius=frame.radius, latitude=frame.latitude,
             longitude=frame.longitude, time=np.datetime64(frame.time, 's'))

    loaded = load_export_frame(tmp_path, 'rho000070')
    assert loaded.time == frame.time
    assert np.array_equal(loaded.rho, frame.rho)
    assert np.array_equal(loaded.longitude, frame.longitude)


def test_leading_edge_follows_front_and_ends_at_the_field_edge():
    heights = np.arange(2.0, 20.0, 0.5)
    fronts = np.array([0.0, 5.0, 10.0, 15.0, 25.0, 8.0])
    ratio = np.where(heights[None] <= fronts[:, None], 2.0, 1.0)
    ratio[:, heights > 18.0] = np.nan  # outside the field of view

    edge = leading_edge(ratio, heights)

    assert np.isnan(edge[0])
    assert np.allclose(edge[1:4], np.array([5.0, 10.0, 15.0]) + 0.5 * 0.9)
    # the front left the field of view; later frames are no longer tracked
    assert np.isnan(edge[4:]).all()


def test_projected_height_of_plane_of_sky_and_halo_directions():
    observer = np.array([1.0, 0.0, 0.0])
    in_plane = projected_height(10.0, np.array([0.0, 1.0, 0.0]), 215.0, observer)
    toward = projected_height(10.0, np.array([1.0, 0.0, 0.0]), 215.0, observer)
    oblique = projected_height(10.0, np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0]), 1e9, observer)

    assert np.isclose(in_plane, 10.0)
    assert np.isclose(toward, 0.0)
    assert np.isclose(oblique, 10.0 * np.sqrt(0.5))


def test_density_metrics_recover_calibration_factor_from_all_points():
    from sunerf.evaluation.psi.plot_histograms import density_metrics

    generator = np.random.default_rng(0)
    truth = 10 ** generator.uniform(2.0, 6.0, size=(20, 15, 10))
    reconstruction = 3.0 * truth
    truth[0, 0, 0] = -1.0  # excluded

    metrics = density_metrics(truth, reconstruction)

    assert metrics["n_points"] == truth.size - 1
    assert metrics["n_excluded"] == 1
    assert np.isclose(metrics["normalization_factor"], 3.0)
    assert np.isclose(metrics["fit_slope_log10"], 1.0)
    assert np.isclose(metrics["fit_intercept_log10"], np.log10(3.0))
    assert np.isclose(metrics["rmse_log10"], np.log10(3.0))
    assert np.isclose(metrics["rmse_log10_normalized"], 0.0, atol=1e-9)
    assert np.isclose(metrics["correlation_log10"], 1.0)


def test_leading_edge_ignores_narrow_enhancements_ahead():
    heights = np.arange(2.0, 30.0, 0.25)
    profile = np.where(heights <= 8.0, 2.0, 1.0)
    profile[np.abs(heights - 15.0) < 0.3] = 1.5  # isolated fluctuation ahead of the front

    height, at_limit = outermost_above(profile, heights)

    # the threshold 1.1 lies between the samples at 8 (ratio 2) and 8.25 (ratio 1)
    assert np.isclose(height, 8.0 + 0.25 * 0.9)
    assert not at_limit


def _analytic_frame(n_latitude, n_longitude):
    """Smooth density with angular structure on a mesh of the given angular size."""
    radius = np.linspace(2.0, 30.0, 85)
    latitude = np.linspace(-np.pi / 2 + 0.02, np.pi / 2 - 0.02, n_latitude)
    longitude = np.linspace(0.01, 2 * np.pi - 0.01, n_longitude)
    r, lat, lon = np.meshgrid(radius, latitude, longitude, indexing='ij')
    rho = 3e5 / r ** 2 * (1.0 + 0.8 * np.cos(3 * lon) * np.cos(lat) ** 2 + 0.5 * np.sin(2 * lat))
    return DensityFrame(rho.astype(np.float32), radius, latitude, longitude, datetime(2021, 10, 28, 18))


def test_beam_profile_and_cone_mass_do_not_depend_on_the_mesh():
    target = np.deg2rad(-20.0), np.deg2rad(245.0)
    fine, coarse = _analytic_frame(251, 368), _analytic_frame(142, 231)

    assert np.allclose(beam_profile(fine, *target), beam_profile(coarse, *target), rtol=0.01)
    mass = [reduce_frame(frame, *target, np.deg2rad(30.0))['mass_per_radius'] for frame in (fine, coarse)]
    assert np.allclose(mass[0], mass[1], rtol=0.02)


def test_structure_positions_are_not_tied_to_the_grid():
    from sunerf.evaluation.cme.mass_transport import ray_structure

    radius = np.linspace(2.0, 30.0, 400)
    peaks = np.linspace(14.0, 15.0, 9)
    found = []
    for peak in peaks:
        rho = 3e5 * (1.0 + 4.0 * np.exp(-0.5 * ((radius - peak) / 1.5) ** 2)) / radius ** 2
        structure = ray_structure(rho, radius, dr=0.25, smooth=0.5, r_search_min=3.0)
        found.append((structure.sheath, structure.front))
    found = np.array(found)

    # peak and edge follow the shift of the profile more closely than the 0.25 R_sun grid
    assert np.abs(found[:, 0] - peaks).max() < 0.06
    assert np.abs(np.diff(found[:, 1]) - np.diff(peaks)).max() < 0.06


def test_kinematic_fits_recover_constant_acceleration():
    from sunerf.evaluation.cme.mass_transport import KM_PER_RSUN, fit_constant_acceleration, local_kinematics

    times = np.arange(0.0, 4 * 3600.0, 300.0)
    speed_kms, acceleration_ms2 = 900.0, -20.0
    radius = 3.0 + (speed_kms * times + 0.5 * acceleration_ms2 * 1e-3 * times ** 2) / KM_PER_RSUN
    radius[5] = np.nan

    speed_at_10, acceleration = fit_constant_acceleration(times, radius, 10.0)
    expected = np.sqrt(speed_kms ** 2 + 2 * acceleration_ms2 * 1e-3 * (10.0 - 3.0) * KM_PER_RSUN)
    assert np.isclose(speed_at_10, expected, rtol=1e-6)
    assert np.isclose(acceleration, acceleration_ms2, rtol=1e-6)
    assert np.isnan(fit_constant_acceleration(times, radius, 100.0)[0])

    speed, local_acceleration = local_kinematics(times, radius, window_s=2700.0)
    inner = np.isfinite(speed)
    assert inner.sum() > 30 and not inner[0] and not inner[-1] and not inner[5]
    assert np.allclose(speed[inner], speed_kms + acceleration_ms2 * 1e-3 * times[inner], rtol=1e-6)
    assert np.allclose(local_acceleration[inner], acceleration_ms2, rtol=1e-5)
