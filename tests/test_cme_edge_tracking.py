import numpy as np

from sunerf.evaluation.cme.plot_cme_mass_budget import _axis_track_statistics, axis_tracks, track_kinematics
from sunerf.evaluation.cme.plot_cme_propagation import core_tracks, half_maximum


def moving_ridge():
    radius = np.arange(3., 60.)
    seconds = np.arange(12) * 1800.
    centers = 15. + np.arange(len(seconds))
    excess = 3 * np.exp(-0.5 * ((radius - centers[:, None]) / 2) ** 2)
    ambient = np.ones_like(excess) / radius ** 2
    return (1 + excess) * ambient, ambient, radius, seconds


def test_front_loss_keeps_rear_and_core_and_recovers():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    for t in range(3, 7):
        density[t, radius >= expected[t, 1] + 2] = np.nan
    tracks = core_tracks(density, ambient, radius, seconds)
    np.testing.assert_allclose(tracks[:, :2], expected[:, :2])
    assert np.isnan(tracks[3:7, 2]).all()
    np.testing.assert_allclose(tracks[7:], expected[7:])


def test_rear_clipping_and_implausible_jump_keep_front():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    for t in (3, 4):
        # A shoulder above half height extends to the grid edge, then far behind the core.
        shoulder = radius < expected[t, 1]
        if t == 4:
            shoulder &= radius > 5
        density[t, shoulder] = 2.8 * ambient[t, shoulder]
    tracks = core_tracks(density, ambient, radius, seconds)
    np.testing.assert_allclose(tracks[:, 1:], expected[:, 1:])
    assert np.isnan(tracks[3:5, 0]).all()
    np.testing.assert_allclose(tracks[5:], expected[5:])


def test_weak_residual_does_not_extend_track():
    density, ambient, radius, seconds = moving_ridge()
    density[3:5] = ambient[3:5] + 0.01 * (density[3:5] - ambient[3:5])
    tracks = core_tracks(density, ambient, radius, seconds)
    assert np.isfinite(tracks[:3]).all()
    assert np.isnan(tracks[3:]).all()


def test_half_height_ignores_single_bin_dip():
    profile = np.array([0., 0., 1., 3.5, 1., 4., 6., 4., 1., 3.5, 1., 0.])
    assert np.isclose(half_maximum(profile, 0, 6), 2.8)
    assert np.isclose(half_maximum(profile, 11, 6), 9.2)


def test_first_zero_edges_use_nearest_crossing_and_stop_at_missing_data():
    radius = np.arange(10., 19.)
    profile = np.array([1., -0.5, 0.5, 2., 3., 1., 0., 1., -1.])
    excess = np.tile(profile, (4, 3, 1))
    excess[1, :, 6] = np.nan
    excess[2, :, :3] = 0.5
    ambient = np.ones_like(excess) / radius**2
    rays = np.full((4, 3, 3), np.nan)
    tracks, _, offsets = _axis_track_statistics(
        ambient * (1 + excess), ambient, radius, np.arange(4) * 1800., median_profile=True,
        edge_fraction=0., ray_edges=rays)
    expected = np.array([[11.5, 14., 16.], [11.5, 14., np.nan],
                         [np.nan, 14., 16.], [11.5, 14., 16.]])
    np.testing.assert_allclose(tracks, expected)
    np.testing.assert_allclose(rays, np.repeat(expected[:, None], 3, axis=1))
    np.testing.assert_allclose(offsets[:, :, 0], expected[:, [0, 2]] - expected[:, 1:2])


def test_kinematics_need_support_on_each_side_of_gap():
    track = np.arange(25, dtype=float)
    track[12] = np.nan
    speed, acceleration = track_kinematics(np.arange(25) * 1800., track, 6 * 3600.)
    assert np.isnan(speed[11:14]).all()
    assert np.isnan(acceleration[11:14]).all()
    assert np.isfinite(speed[[5, 19]]).all()


def test_edges_cross_local_background_enhancement_on_each_ray():
    radius = np.arange(10., 17.)
    # The compensated background varies with radius and across rays.
    bg = (1. + .05 * np.arange(7)) / radius**2
    ambient = np.broadcast_to(bg * np.array([1., 2., 3.])[:, None], (5, 3, 7))
    ratio = np.array([.98, 1.04, 1.16, 3., 1.25, 1.05, 1.])
    density = ambient * ratio
    density[-1] = 1.02 * ambient[-1]
    rays = np.full((5, 3, 3), np.nan)
    tracks, _, offsets = _axis_track_statistics(
        density, ambient, radius, np.arange(5) * 1800., above_background=.1,
        median_profile=True, continue_to_index=4, ray_edges=rays)
    expected = np.array([11.5, 13., 14.75])
    np.testing.assert_allclose(tracks[:4], np.tile(expected, (4, 1)))
    np.testing.assert_allclose(rays[:4], np.tile(expected, (4, 3, 1)))
    np.testing.assert_allclose(offsets[:4, :, 0], np.tile([-1.5, 1.75], (4, 1)))
    assert np.isnan(tracks[4]).all()


def test_optional_continuation_uses_measured_weak_peaks_and_stops_at_requested_time():
    density, ambient, radius, seconds = moving_ridge()
    density[6:] = ambient[6:] + .04 * (density[6:] - ambient[6:])
    density[7] = ambient[7]
    rays = density[:, None]
    background = ambient[:, None]
    baseline, _, _ = _axis_track_statistics(rays, background, radius, seconds, median_profile=True, edge_fraction=.1)
    weak = np.zeros(len(seconds), dtype=bool)
    tracks, _, _ = _axis_track_statistics(rays, background, radius, seconds, median_profile=True, edge_fraction=.1,
                                           continue_to_index=9, weak_frames=weak)
    np.testing.assert_allclose(tracks[:6], baseline[:6])
    np.testing.assert_array_equal(np.flatnonzero(weak), [6, 8, 9])
    np.testing.assert_allclose(tracks[weak, 1], 15 + np.flatnonzero(weak))
    assert np.isnan(tracks[[7, 10, 11]]).all()


def test_core_is_independent_of_flank_threshold_and_first_crossing_wins():
    radius = np.arange(10., 19.)
    ambient = np.broadcast_to(1 / radius**2, (5, 3, 9))
    ratio = np.array([1., 1.4, 1.05, 1.25, 3., 1.25, 1.05, 1.4, 1.])
    density = ambient * ratio
    density[-1] = ambient[-1] * (1 + .04 * (ratio - 1))
    ray_cores = []
    for threshold in (.1, 3.):
        rays = np.full((5, 3, 3), np.nan)
        tracks, _, _ = _axis_track_statistics(
            density, ambient, radius, np.arange(5) * 1800., above_background=threshold,
            median_profile=True, continue_to_index=4, ray_edges=rays)
        np.testing.assert_allclose(tracks[:, 1], 14.)
        ray_cores.append(rays[:, :, 1])
        if threshold == .1:
            np.testing.assert_allclose(tracks[:4, [0, 2]], np.tile([12.25, 15.75], (4, 1)))
            assert np.isnan(tracks[-1, [0, 2]]).all()
        else:
            assert np.isnan(tracks[:, [0, 2]]).all()
            assert np.isnan(rays[:, :, [0, 2]]).all()
    np.testing.assert_allclose(ray_cores[0][:4], 14.)
    np.testing.assert_equal(*ray_cores)


def test_axis_rejects_a_bright_minority_ridge():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    rays = np.repeat(density[:, None], 5, axis=1)
    background = np.repeat(ambient[:, None], 5, axis=1)
    extra = np.exp(-0.5 * ((radius - expected[:, 1, None] - 10) / 2) ** 2)
    rays[:, :2] += (100 * extra * ambient)[:, None]
    averaged = core_tracks(rays.mean(axis=1), ambient, radius, seconds)
    assert np.all(averaged[:, 1] - expected[:, 1] >= 9)
    tracks, spread = axis_tracks(rays, background, radius, seconds)
    np.testing.assert_allclose(tracks, expected)
    np.testing.assert_allclose(spread, [expected, expected])


def test_median_profile_tracks_a_resolved_lobe_and_requires_coverage_on_each_flank():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    # Two rays have a distinct peak. The other three have sloping backgrounds,
    # two above the ridge and one below, so the median still resolves the lobe.
    bright = ambient * (10 + 0.1 * radius)
    faint = ambient * (1 + 0.001 * radius)
    rays = np.stack([density, density, bright, bright, faint], axis=1)
    rays[5, :3, radius >= expected[5, 1] + 2] = np.nan
    # At frame 6 the median front survives, but only one ray measures that front.
    for ray in (0, 2):
        rays[6, ray, radius >= expected[6, 1] + 2] = np.nan
    tracks, spread, offsets = _axis_track_statistics(
        rays, np.broadcast_to(ambient[:, None], rays.shape), radius, seconds, median_profile=True)
    expected[5, 2] = np.nan
    np.testing.assert_allclose(tracks, expected)
    assert np.isnan(spread).all()
    expected_offsets = expected[:, [0, 2]] - expected[:, 1:2]
    expected_offsets[6, 1] = np.nan
    np.testing.assert_allclose(offsets[:, :, 0], expected_offsets)
    np.testing.assert_allclose(offsets[:, :, 1], np.where(np.isfinite(expected_offsets), 0., np.nan))


def test_axis_requires_a_majority_for_each_edge_and_resumes():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    rays = np.repeat(density[:, None], 4, axis=1)
    for t in range(2, 5):
        rays[t, :2, radius >= expected[t, 1] + 2] = np.nan
    tracks, spread = axis_tracks(rays, np.repeat(ambient[:, None], 4, axis=1), radius, seconds)
    np.testing.assert_allclose(tracks[:, :2], expected[:, :2])
    assert np.isnan(tracks[2:5, 2]).all()
    assert np.isnan(spread[:, 2:5, 2]).all()
    np.testing.assert_allclose(tracks[5:], expected[5:])


def test_core_offsets_use_paired_rays_and_preserve_one_sided_gaps():
    _, ambient, radius, seconds = moving_ridge()
    densities, per_ray = [], []
    for shift, width in [(-1, 2), (0, 2.5), (1, 3)]:
        centers = 15 + np.arange(len(seconds)) + shift
        excess = 3 * np.exp(-0.5 * ((radius - centers[:, None]) / width) ** 2)
        density = ambient * (1 + excess)
        if shift <= 0:
            density[4, radius >= centers[4] + 2] = np.nan
        densities.append(density)
        positions = core_tracks(density, ambient, radius, seconds)
        per_ray.append(positions[:, [0, 2]] - positions[:, 1:2])
    rays = np.stack(densities, axis=1)
    _, _, offsets = _axis_track_statistics(rays, np.broadcast_to(ambient[:, None], rays.shape), radius, seconds)
    expected = np.stack([np.nanmean(per_ray, axis=0), np.nanstd(per_ray, axis=0)], axis=-1)
    complete = np.arange(len(seconds)) != 4
    np.testing.assert_allclose(offsets[complete], expected[complete], atol=1e-10)
    assert np.isfinite(offsets[4, 0]).all()
    assert np.isnan(offsets[4, 1]).all()  # Only one ray still measures the front.


def test_axis_has_no_position_between_two_disjoint_equal_groups():
    density, ambient, radius, seconds = moving_ridge()
    offset = np.roll((density - ambient) * radius ** 2, 12, axis=1)
    rays = np.stack([density, density, ambient + offset / radius ** 2, ambient + offset / radius ** 2], axis=1)
    tracks, spread = axis_tracks(rays, np.repeat(ambient[:, None], 4, axis=1), radius, seconds)
    assert np.isnan(tracks).all()
    assert np.isnan(spread).all()


def test_axis_recovers_sustained_ridge_after_unrelated_early_detection():
    density, ambient, radius, seconds = moving_ridge()
    clean = core_tracks(density, ambient, radius, seconds)
    early = 30 * np.exp(-0.5 * ((radius - 48) / 2) ** 2)
    density[:2] = ambient[:2] * (1 + early)
    density[2:4] = ambient[2:4]
    rays = np.repeat(density[:, None], 5, axis=1)
    tracks, _ = axis_tracks(rays, np.repeat(ambient[:, None], 5, axis=1), radius, seconds)
    assert np.isnan(tracks[:4]).all()
    np.testing.assert_allclose(tracks[4:], clean[4:])


def test_axis_path_can_cross_a_missing_frame_without_filling_it():
    density, ambient, radius, seconds = moving_ridge()
    expected = core_tracks(density, ambient, radius, seconds)
    density[5] = np.nan
    tracks, spread = axis_tracks(density[:, None], ambient[:, None], radius, seconds)
    expected[5] = np.nan
    np.testing.assert_allclose(tracks, expected)
    assert np.isnan(spread[:, 5]).all()


def test_moving_patch_majority_counts_only_selected_rays():
    density, ambient, radius, seconds = moving_ridge()
    rays = np.repeat(density[:, None], 8, axis=1)
    reference = np.repeat(ambient[:, None], 8, axis=1)
    counts = np.full(len(seconds), 8)
    counts[2:7] = 3
    rays[2:7, 3:] = np.nan
    reference[2:7, 3:] = np.nan
    tracks, _, _ = _axis_track_statistics(rays, reference, radius, seconds, counts)
    expected = core_tracks(density, ambient, radius, seconds)
    np.testing.assert_allclose(tracks, expected)
    # Two failed measurements on selected rays should still suppress that frame.
    rays[4, :2] = np.nan
    tracks, _, _ = _axis_track_statistics(rays, reference, radius, seconds, counts)
    assert np.isnan(tracks[4]).all()
    np.testing.assert_allclose(tracks[5:], expected[5:])


def test_start_index_fits_only_later_frames_and_preserves_output_times():
    density, ambient, radius, seconds = moving_ridge()
    rays = density[:, None]
    background = ambient[:, None]
    expected = _axis_track_statistics(rays[5:], background[5:], radius, seconds[5:])
    ray_edges = np.full((len(seconds), 1, 3), np.nan)
    tracks, spread, offsets = _axis_track_statistics(rays, background, radius, seconds,
                                                    start_index=5, ray_edges=ray_edges)
    assert np.isnan(tracks[:5]).all() and np.isnan(ray_edges[:5]).all()
    np.testing.assert_allclose(tracks[5:], expected[0])
    np.testing.assert_allclose(spread[:, 5:], expected[1])
    np.testing.assert_allclose(offsets[5:], expected[2])
