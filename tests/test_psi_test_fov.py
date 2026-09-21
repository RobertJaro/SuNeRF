import numpy as np
import pytest

sunpy_map = pytest.importorskip("sunpy.map")

from sunerf.data.psi.degrade_psi import (  # noqa: E402
    additive_degradation_mask,
    fill_nearest_finite,
    degradation_config,
    degrade_sequence,
    radial_gain_mask,
    resample_detector_mask,
)
from sunerf.data.psi.prepare_psi_clear import (  # noqa: E402
    discover_observers,
    radial_rsun,
)
from sunerf.evaluation.plot_psi_degradation import plot_sample  # noqa: E402


def _test_map(data=None):
    header = {
        "ctype1": "HPLN-TAN",
        "ctype2": "HPLT-TAN",
        "cunit1": "arcsec",
        "cunit2": "arcsec",
        "cdelt1": 10.0,
        "cdelt2": 12.0,
        "crpix1": 3.0,
        "crpix2": 3.0,
        "crval1": 0.0,
        "crval2": 0.0,
        "date-obs": "2021-10-28T15:30:00",
        "hgln_obs": 0.0,
        "hglt_obs": 0.0,
        "dsun_obs": 149_597_870_691.0,
        "rsun_ref": 696_000_000.0,
        "rsun_obs": 959.63,
    }
    if data is None:
        data = np.arange(25, dtype=np.float32).reshape(5, 5)
    return sunpy_map.Map(np.asarray(data, dtype=np.float32), header)


def test_degradation_keywords_hold_view_specific_settings():
    l1 = degradation_config(
        synthetic="LASCO_C2",
        inner_rsun=2.2,
        outer_rsun=8.3,
        inner_gain=0.6,
        outer_gain=1.2,
        gain_power=1,
    )
    l4 = degradation_config(
        synthetic="STEREO_A_COR2",
        inner_rsun=3,
        outer_rsun=15,
        inner_gain=1.5,
        outer_gain=1.1,
        gain_power=2,
    )
    l5 = degradation_config(
        synthetic="STEREO_B_COR2",
        inner_rsun=8,
        outer_rsun=30,
        inner_gain=1.3,
        outer_gain=0.7,
        gain_power=3,
    )

    assert (l1["inner_rsun"], l1["outer_rsun"]) == (2.2, 8.3)
    assert (l4["inner_rsun"], l4["outer_rsun"]) == (3.0, 15.0)
    assert (l5["inner_rsun"], l5["outer_rsun"]) == (8.0, 30.0)
    assert l1["mask_fraction"] == l4["mask_fraction"] == l5["mask_fraction"] == 0.1
    assert (l1["inner_gain"], l1["outer_gain"], l1["gain_power"]) == (0.6, 1.2, 1.0)
    assert (l4["inner_gain"], l4["outer_gain"], l4["gain_power"]) == (1.5, 1.1, 2.0)
    assert (l5["inner_gain"], l5["outer_gain"], l5["gain_power"]) == (1.3, 0.7, 3.0)


def test_clear_preparation_discovers_all_complete_observers(tmp_path):
    for observer in ("L1", "L4", "P1"):
        for product in ("pb", "tb"):
            path = tmp_path / observer / product
            path.mkdir(parents=True)
            (path / f"{observer}_{product}050.fts").touch()
    incomplete = tmp_path / "incomplete" / "pb"
    incomplete.mkdir(parents=True)
    (incomplete / "incomplete_pb050.fts").touch()

    assert discover_observers(tmp_path) == ["L1", "L4", "P1"]


def test_detector_mask_identity_mapping_preserves_values():
    source = _test_map()
    target = _test_map(np.zeros((5, 5), dtype=np.float32))
    mapped = resample_detector_mask(source, target)

    np.testing.assert_allclose(mapped, source.data, rtol=0, atol=1e-5)


def test_detector_mask_mapping_ignores_observer_and_date():
    source = _test_map()
    target = _test_map(np.zeros((5, 5), dtype=np.float32))
    target.meta["date-obs"] = "2030-01-01T00:00:00"
    target.meta["hgln_obs"] = 120.0

    mapped = resample_detector_mask(source, target)
    np.testing.assert_allclose(mapped, source.data, rtol=0, atol=1e-5)


def test_detector_mask_annulus_maps_onto_the_field_of_view_of_the_image():
    def annulus_map(size, inner, outer, cdelt):
        rows, columns = np.indices((size, size))
        center = (size - 1) / 2
        radius = np.hypot(columns - center, rows - center)
        # Linear ramp from 1 at the occulter edge to 0 at the outer edge.
        data = ((outer - radius) / (outer - inner)).astype(np.float32)
        data[(radius < inner) | (radius > outer)] = np.nan
        s_map = _test_map(data)
        s_map.meta.update({"crpix1": center + 1, "crpix2": center + 1, "cdelt1": cdelt, "cdelt2": cdelt})
        return sunpy_map.Map(data, s_map.meta), radius

    # Wide detector occulter and another plate scale than the image.
    mask, _ = annulus_map(200, inner=40, outer=95, cdelt=30.0)
    target, radius = annulus_map(100, inner=10, outer=48, cdelt=7.0)

    mapped = resample_detector_mask(mask, target)

    assert mapped.shape == (100, 100) and np.isfinite(mapped).all()
    field = (radius > 12) & (radius < 46)
    np.testing.assert_allclose(mapped[field], ((48 - radius) / (48 - 10))[field], atol=0.04)


def test_detector_mask_produces_only_additive_field():
    mask = np.arange(100, dtype=np.float32).reshape(10, 10)

    additive = additive_degradation_mask(mask, image_level=20.0, fraction=0.1)

    assert np.nanmin(additive) == 0.0
    assert np.nanmax(additive) == 2.0


@pytest.mark.parametrize(
    ("inner_gain", "outer_gain", "power", "expected_midpoint"),
    [(0.6, 1.2, 1, 0.9), (1.5, 1.1, 2, 1.4), (1.3, 0.7, 3, 1.225)],
)
def test_radial_gain_uses_configured_endpoints_and_power(
    inner_gain, outer_gain, power, expected_midpoint
):
    gain = radial_gain_mask(
        np.array([2.0, 5.0, 8.0]),
        inner_rsun=2.0,
        outer_rsun=8.0,
        inner_gain=inner_gain,
        outer_gain=outer_gain,
        power=power,
    )

    np.testing.assert_allclose(
        gain, [inner_gain, expected_midpoint, outer_gain], rtol=1e-6
    )


def test_fill_nearest_finite_completes_source_occulter_gap():
    source = np.array(
        [[1.0, 1.0, 2.0], [1.0, np.nan, 2.0], [3.0, 3.0, 4.0]],
        dtype=np.float32,
    )
    filled = fill_nearest_finite(source)

    assert np.isfinite(filled).all()
    assert filled[1, 1] in {1.0, 2.0, 3.0, 4.0}
    np.testing.assert_array_equal(
        filled[np.isfinite(source)], source[np.isfinite(source)]
    )


def test_fill_nearest_finite_rejects_empty_mask():
    with pytest.raises(ValueError, match="no finite pixels"):
        fill_nearest_finite(np.full((3, 3), np.nan, dtype=np.float32))


def test_degradation_preserves_image_grid_and_only_masks_fov(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    truth_dir = tmp_path / "truth"
    source_maps = {}
    for product in ("tb", "pb"):
        path = input_dir / product / f"L4_{product}050.fts"
        path.parent.mkdir(parents=True, exist_ok=True)
        source_maps[product] = _test_map(np.arange(25).reshape(5, 5) + 1)
        source_maps[product].save(path)

    config = degradation_config(
        synthetic="STEREO_A_COR2",
        inner_rsun=0.005,
        outer_rsun=0.025,
        mask_fraction=0.1,
        inner_gain=1.5,
        outer_gain=1.1,
        gain_power=2,
    )
    degrade_sequence(
        input_dir,
        output_dir,
        truth_dir,
        config,
        {"tb": _test_map(), "pb": _test_map()},
        truth_frame="050",
        overwrite=True,
    )

    valid = np.load(truth_dir / "valid_mask.npy").astype(bool)
    expected_fov = (radial_rsun(source_maps["tb"]) >= config["inner_rsun"]) & (
        radial_rsun(source_maps["tb"]) <= config["outer_rsun"]
    )
    np.testing.assert_array_equal(valid, expected_fov)
    for product in ("tb", "pb"):
        output_map = sunpy_map.Map(output_dir / product / f"L4_{product}050.fts")
        additive = np.load(truth_dir / f"{product}_additive.npy")
        multiplier = np.load(truth_dir / f"{product}_multiplier.npy")
        expected = (source_maps[product].data + additive) * multiplier

        assert output_map.data.shape == source_maps[product].data.shape
        assert output_map.meta["gainin"] == config["inner_gain"]
        assert output_map.meta["gainout"] == config["outer_gain"]
        assert output_map.meta["gainpow"] == config["gain_power"]
        np.testing.assert_allclose(
            output_map.wcs.pixel_scale_matrix,
            source_maps[product].wcs.pixel_scale_matrix,
        )
        np.testing.assert_array_equal(np.isfinite(output_map.data), valid)
        np.testing.assert_allclose(output_map.data[valid], expected[valid], rtol=1e-6)


def test_degradation_preview_includes_images_and_degradation_masks(tmp_path):
    clean_paths = {}
    degraded_paths = {}
    for product in ("tb", "pb"):
        clean_path = tmp_path / "clean" / f"L4_{product}050.fts"
        degraded_path = tmp_path / "degraded" / f"L4_{product}050.fts"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        degraded_path.parent.mkdir(parents=True, exist_ok=True)
        clean_map = _test_map()
        degraded_map = _test_map(clean_map.data * 1.2 + 0.5)
        degraded_map.meta["occmax"] = 0.02
        degraded_map.meta["occmin"] = 0.0
        degraded_map.meta["gainin"] = 1.2
        degraded_map.meta["gainout"] = 1.2
        degraded_map.meta["gainpow"] = 1.0
        clean_map.save(clean_path)
        degraded_map.save(degraded_path)
        clean_paths[product] = clean_path
        degraded_paths[product] = degraded_path

    output = plot_sample(clean_paths, degraded_paths, "L4", tmp_path / "plots")

    assert output.is_file()
    assert output.name == "L4_frame_050.png"


def test_degradation_without_fov_limits_keeps_the_field_of_view_of_the_data(tmp_path):
    from sunerf.data.psi.prepare_psi_clear import radial_rsun

    input_dir, output_dir, truth_dir = (tmp_path / name for name in ("input", "output", "truth"))
    clean = np.arange(25, dtype=np.float32).reshape(5, 5) + 1
    clean[0, 0] = np.nan
    for product in ("tb", "pb"):
        path = input_dir / product / f"L4_{product}050.fts"
        path.parent.mkdir(parents=True, exist_ok=True)
        _test_map(clean).save(path)

    config = degradation_config(synthetic="STEREO_A_COR2", inner_gain=1.5, outer_gain=1.1, gain_power=2)
    assert config["inner_rsun"] is None and config["outer_rsun"] is None
    degrade_sequence(input_dir, output_dir, truth_dir, config,
                     {"tb": _test_map(), "pb": _test_map()}, truth_frame="050", overwrite=True)

    degraded = sunpy_map.Map(output_dir / "tb" / "L4_tb050.fts")
    np.testing.assert_array_equal(np.isfinite(degraded.data), np.isfinite(clean))
    # The gain spans the radial extent of the valid pixels.
    radius = radial_rsun(degraded)[np.isfinite(clean)]
    assert degraded.meta["occmin"] == pytest.approx(radius.min())
    assert degraded.meta["occmax"] == pytest.approx(radius.max())

    with pytest.raises(ValueError, match="together"):
        degradation_config(synthetic="STEREO_A_COR2", inner_rsun=2.0)


def test_polar_smoothing_removes_a_radial_streamer_and_keeps_the_large_scale_pattern():
    from sunerf.data.coronagraph.compute_correction import smooth_polar

    size, center = 256, (127.5, 127.5)
    rows, columns = np.indices((size, size))
    radius = np.hypot(columns - center[0], rows - center[1])
    angle = np.arctan2(rows - center[1], columns - center[0])
    background = 1e4 / np.clip(radius, 20, None) ** 2 * (1 + 0.3 * np.cos(angle))
    # The streamer lies on the periodic seam of the position angle.
    streamer = 1 + np.exp(-0.5 * (np.angle(np.exp(1j * (angle - np.pi))) / np.deg2rad(8)) ** 2)
    mask = (background * streamer).astype(np.float32)
    mask[radius < 20] = np.nan

    smoothed = smooth_polar(mask, center, angle_deg=60.0, radius_pixels=0.0)
    quiet = background.astype(np.float32)
    quiet[radius < 20] = np.nan
    smoothed_quiet = smooth_polar(quiet, center, angle_deg=60.0, radius_pixels=0.0)

    np.testing.assert_array_equal(np.isnan(smoothed), np.isnan(mask))
    ring = np.abs(radius - 80) < 1
    # The streamer doubles the mask and leaves no trace in the smoothed one.
    assert np.nanmax((mask / quiet)[ring]) == pytest.approx(2.0, abs=0.05)
    np.testing.assert_allclose(smoothed[ring], smoothed_quiet[ring], rtol=0.05)
    # The dipole of the detector pattern survives, damped.
    assert np.nanmax(smoothed[ring]) / np.nanmin(smoothed[ring]) > 1.25
    # Without smoothing the mask passes through the polar grid unchanged.
    unchanged = smooth_polar(mask, center, angle_deg=0.0, radius_pixels=0.0)
    np.testing.assert_allclose(unchanged[radius > 25], mask[radius > 25], rtol=0.05)


def test_polar_percentile_smoothing_stays_below_the_corona_between_streamers():
    from sunerf.data.coronagraph.compute_correction import smooth_polar

    size, center = 256, (120.0, 135.0)
    rows, columns = np.indices((size, size))
    radius = np.hypot(columns - center[0], rows - center[1])
    angle = np.arctan2(rows - center[1], columns - center[0])
    mask = 1e4 / np.clip(radius, 20, None) ** 2.5 * (1 + 0.3 * np.cos(angle))
    for position, width in ((0.7, 8), (2.9, 12)):
        mask = mask * (1 + 1.5 * np.exp(-0.5 * (np.angle(np.exp(1j * (angle - position))) / np.deg2rad(width)) ** 2))
    mask = mask.astype(np.float32)
    mask[radius < 20] = np.nan
    valid = np.isfinite(mask) & (radius < 110)

    def oversubtracted(smoothed):
        return np.mean((mask - smoothed)[valid] < -0.02 * mask[valid])

    # The angular mean lies above the mask between the streamers; the running
    # low percentile follows its lower envelope.
    assert oversubtracted(smooth_polar(mask, center, 60.0, 0.0, percentile=None)) > 0.5
    assert oversubtracted(smooth_polar(mask, center, 60.0, 0.0, percentile=5.0)) < 0.02
