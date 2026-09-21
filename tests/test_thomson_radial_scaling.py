import numpy as np
import pytest
import torch
from datetime import datetime

import sunerf.train.callback as callback_module
from sunerf.data.loader.thomson_instrument import create_scaling_mask
from sunerf.model.thomson import ThomsonSuNeRFModule
from sunerf.train.callback import ThomsonImageCallback


def _signed_radial_test_data():
    radii = np.linspace(2.0, 30.0, 96, dtype=np.float32)
    projected_radius = np.repeat(radii[:, None], 256, axis=1)
    azimuthal_structure = np.sin(np.linspace(0, 8 * np.pi, 256, endpoint=False)).astype(np.float32)
    true_scale = (radii / radii[0]) ** -2.5
    tb = true_scale[:, None] * azimuthal_structure[None, :]
    pb = 0.4 * true_scale[:, None] * np.roll(azimuthal_structure, 11)[None, :]
    image_stack = np.stack([tb, pb], axis=-1)[None]
    return image_stack, projected_radius[None], true_scale


def test_radial_mad_mask_is_positive_and_tracks_signed_radial_spread():
    image_stack, projected_radius, true_scale = _signed_radial_test_data()

    mask = create_scaling_mask(
        projected_radius,
        {
            "type": "radial_mad",
            "n_bins": 24,
            "min_samples": 128,
            "trim_percentiles": None,
        },
        image_stack=image_stack,
    )

    assert mask.shape == image_stack.shape
    assert np.all(np.isfinite(mask))
    assert np.all(mask > 0)

    fitted_tb_scale = np.nanmedian(mask[0, ..., 0], axis=1)
    fitted_tb_scale /= fitted_tb_scale[0]
    np.testing.assert_allclose(fitted_tb_scale, true_scale, rtol=0.2, atol=0.02)

    adjusted = image_stack / mask
    assert np.nanmin(adjusted) < 0
    assert np.nanmax(adjusted) > 0
    adjusted_spread = np.nanmedian(
        np.abs(adjusted[0, ..., 0] - np.nanmedian(adjusted[0, ..., 0], axis=1, keepdims=True)),
        axis=1,
    )
    assert np.nanmax(adjusted_spread) / np.nanmin(adjusted_spread) < 1.5


def test_radial_mad_mask_is_fitted_from_temporal_mean_image():
    image_stack, projected_radius, _ = _signed_radial_test_data()
    config = {
        "type": "radial_mad",
        "n_bins": 24,
        "min_samples": 128,
        "trim_percentiles": None,
    }
    single_mask = create_scaling_mask(projected_radius, config, image_stack=image_stack)

    stacked_images = np.concatenate([image_stack, 2.0 * image_stack], axis=0)
    stacked_radius = np.repeat(projected_radius, 2, axis=0)
    stacked_mask = create_scaling_mask(stacked_radius, config, image_stack=stacked_images)

    np.testing.assert_allclose(stacked_mask[0], stacked_mask[1])
    np.testing.assert_allclose(stacked_mask[0], 1.5 * single_mask[0], rtol=2e-5, atol=1e-7)


def test_radial_mad_mask_uses_available_channel_for_missing_channel():
    image_stack, projected_radius, _ = _signed_radial_test_data()
    image_stack[..., 1] = np.nan

    mask = create_scaling_mask(
        projected_radius,
        {"type": "radial_mad", "n_bins": 16, "min_samples": 64},
        image_stack=image_stack,
    )

    np.testing.assert_allclose(mask[..., 1], mask[..., 0])


def test_radial_mad_cubic_fit_rejects_a_bright_radial_outlier():
    image_stack, projected_radius, true_scale = _signed_radial_test_data()
    image_stack = image_stack.copy()
    image_stack[:, 44:48] *= 50.0

    mask = create_scaling_mask(
        projected_radius,
        {
            "type": "radial_mad",
            "n_bins": 24,
            "min_samples": 128,
            "trim_percentiles": None,
        },
        image_stack=image_stack,
    )

    fitted_scale = np.nanmedian(mask[0, ..., 0], axis=1)
    fitted_scale /= fitted_scale[0]
    expected_scale = true_scale / true_scale[0]
    np.testing.assert_allclose(fitted_scale, expected_scale, rtol=0.3, atol=0.03)


def test_scaling_mask_normalization_preserves_negative_values():
    image = torch.tensor([[-2.0], [0.0], [4.0]])
    batch = {"scaling_mask": torch.tensor([[2.0], [2.0], [2.0]])}

    adjusted = ThomsonSuNeRFModule._normalize_with_scaling_mask(image, batch)

    torch.testing.assert_close(adjusted, torch.tensor([[-1.0], [0.0], [2.0]]))


def test_radial_mad_requires_image_stack():
    with pytest.raises(ValueError, match="require image_stack"):
        create_scaling_mask(np.ones((1, 4, 4)), {"type": "radial_mad"})


def test_thomson_callback_does_not_log_radial_scale_images(monkeypatch):
    image_shape = (8, 8)
    n_pixels = np.prod(image_shape)
    target = torch.linspace(-1.0, 1.0, n_pixels * 2).reshape(n_pixels, 2)
    outputs = {
        "target_image": target,
        "model_image": target * 0.9,
        "target_ratio": torch.full((n_pixels, 1), 0.25),
        "model_ratio": torch.full((n_pixels, 1), 0.2),
        "scaling_mask": torch.linspace(0.1, 1.0, n_pixels * 2).reshape(n_pixels, 2),
    }
    pl_module = type("Module", (), {"validation_outputs": {"test": outputs}})()
    logged = []
    monkeypatch.setattr(callback_module.wandb, "Image", lambda figure: figure)
    monkeypatch.setattr(callback_module.wandb, "log", lambda values: logged.append(values))

    ThomsonImageCallback("test", image_shape).on_validation_end(None, pl_module)

    logged_keys = {key for values in logged for key in values}
    assert "images.test" in logged_keys
    assert "images.test.radial_scale" not in logged_keys
    overview = next(values["images.test"] for values in logged if "images.test" in values)
    titles = {axis.get_title() for axis in overview.axes}
    assert "tB radial scale" not in titles
    assert "pB radial scale" not in titles


def test_data_overview_includes_radial_scaling_masks(monkeypatch):
    images = np.ones((1, 8, 8, 2), dtype=np.float32)
    scaling_masks = np.stack(
        [np.linspace(1.0, 0.1, 64).reshape(8, 8)] * 2,
        axis=-1,
    )[None]
    poses = np.eye(4, dtype=np.float32)[None]
    logged = []
    monkeypatch.setattr(callback_module.wandb, "Image", lambda figure: figure)
    monkeypatch.setattr(callback_module.wandb, "log", lambda values: logged.append(values))

    callback_module.log_overview(
        images,
        poses,
        np.array([0.0]),
        "gray",
        seconds_per_dt=86400,
        Rs_per_ds=15,
        ref_date=datetime(2021, 10, 28),
        ds_key="L4",
        brightness_mode="radially adjusted brightness",
        scaling_masks=scaling_masks,
    )

    overview = logged[0]["Overview.L4"]
    titles = {axis.get_title() for axis in overview.axes}
    assert "tB radial scaling mask" in titles
    assert "pB radial scaling mask" in titles
    scale_axes = [axis for axis in overview.axes if "radial scaling mask" in axis.get_title()]
    assert all(axis.images[0].get_cmap().name == "viridis" for axis in scale_axes)


def test_data_overview_plots_all_euv_channels_with_their_colormaps(monkeypatch):
    images = np.random.default_rng(0).random((1, 8, 8, 3)).astype(np.float32)
    images[0, 0, 0, 1] = np.nan
    poses = np.eye(4, dtype=np.float32)[None]
    logged = []
    monkeypatch.setattr(callback_module.wandb, "Image", lambda figure: figure)
    monkeypatch.setattr(callback_module.wandb, "log", lambda values: logged.append(values))

    cmaps = ["sdoaia171", "sdoaia193", "sdoaia304"]
    callback_module.log_overview(
        images,
        poses,
        np.array([0.0]),
        cmaps,
        seconds_per_dt=86400,
        Rs_per_ds=15,
        ref_date=datetime(2021, 10, 28),
        ds_key="EUVI-B-L5",
        channel_labels=["171", "195", "304"],
    )

    overview = logged[0]["Overview.EUVI-B-L5"]
    image_axes = [axis for axis in overview.axes if axis.images]
    assert [axis.get_title() for axis in image_axes] == ["171", "195", "304"]
    assert [axis.images[0].get_cmap().name for axis in image_axes] == cmaps
    assert not any("tB" in axis.get_title() or "pB" in axis.get_title() for axis in overview.axes)
