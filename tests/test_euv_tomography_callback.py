from types import SimpleNamespace

import numpy as np
import torch

from sunerf.train.euv_callback import EUVTomographyCallback


def _products():
    return {
        "channel_comparison": {
            "enabled": True,
            "rows": ["observation", "prediction", "residual", "relative_residual"],
            "intensity_percentile": 99.5,
            "residual_percentile": 99.0,
        },
        "plasma_diagnostics": {
            "enabled": True,
            "quantities": [
                "mean_log_temperature",
                "column_electron_density",
                "emission_measure",
                "emission_height",
                "absorption_fraction",
            ],
        },
        "thermal_distribution": {
            "enabled": True,
            "spatial_statistic": "median",
            "percentile_band": [16.0, 84.0],
        },
        "response_and_gains": {"enabled": True},
        "ray_sampling": {"enabled": True, "pixel_fraction": [0.5, 0.5]},
    }


def _module():
    renderer = SimpleNamespace(
        log_T=torch.tensor([5.0, 6.0, 7.0]),
        response_log_T=torch.tensor([5.0, 6.0, 7.0]),
        temperature_response=torch.tensor(
            [[[1.0, 0.2], [2.0, 1.0], [0.1, 3.0]]]
        ),
        log_density_axis=torch.empty(0),
        channels=("A94", "A171"),
        response_unit="cm5 DN / (pix s)",
        response_id="sha256:test",
        instrument_gain_delta_dex=torch.tensor([0.0, 0.05]),
        gain_limit_dex=0.15,
        absorption=True,
    )
    outputs = {
        "target_image": torch.tensor(
            [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        ),
        "pred_image": torch.tensor(
            [[1.1, 1.8], [2.1, 3.1], [2.8, 4.2], [4.1, 4.8]]
        ),
        "valid_mask": torch.ones((4, 2), dtype=torch.bool),
        "mean_T": torch.tensor([[5.8], [5.9], [6.0], [6.1]]),
        "column_electron_density_cm2": torch.tensor(
            [[1e17], [2e17], [3e17], [4e17]]
        ),
        "emission_measure_cm5": torch.tensor([[1e26], [2e26], [3e26], [4e26]]),
        "height_map": torch.tensor([1.05, 1.10, 1.15, 1.20]),
        "mean_absorption": torch.tensor([0.0, 0.1, 0.2, 0.3]),
        "differential_emission_measure_cm5_per_dex": torch.tensor(
            [[1e25, 3e25, 1e24], [2e25, 4e25, 2e24],
             [1e25, 5e25, 3e24], [3e25, 6e25, 4e24]]
        ),
        "z_vals_stratified": torch.tensor(
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        "z_vals_hierarchical": torch.tensor(
            [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0],
             [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]
        ),
        "distance": torch.tensor(
            [[1.4, 1.0, 1.4], [1.4, 1.0, 1.4],
             [1.4, 1.0, 1.4], [1.4, 1.0, 1.4]]
        ),
    }
    return SimpleNamespace(
        validation_outputs={"aia_valid": outputs},
        rendering=SimpleNamespace(rendering_modules={"AIA": renderer}),
    )


def test_callback_registers_only_enabled_plot_tensors():
    products = _products()
    products["thermal_distribution"]["enabled"] = False
    products["ray_sampling"]["enabled"] = False
    callback = EUVTomographyCallback(
        "aia_valid",
        "AIA",
        (2, 2),
        ({"id": "A94"}, {"id": "A171"}),
        ("171",),
        products,
    )

    assert callback.channel_indices == (1,)
    assert "target_image" in callback.validation_output_keys
    assert "column_electron_density_cm2" in callback.validation_output_keys
    assert "differential_emission_measure_cm5_per_dex" not in callback.validation_output_keys
    assert "z_vals_stratified" not in callback.validation_output_keys


def test_callback_cadence_and_all_plot_families(monkeypatch):
    logged = []
    monkeypatch.setattr(
        "sunerf.train.euv_callback.wandb.Image", lambda figure: figure
    )
    monkeypatch.setattr(
        "sunerf.train.euv_callback.wandb.log", lambda payload: logged.extend(payload)
    )
    callback = EUVTomographyCallback(
        "aia_valid",
        "AIA",
        (2, 2),
        ({"id": "A94", "cmap": "gray"}, {"id": "A171", "cmap": "gray"}),
        ("A94", "A171"),
        _products(),
        every_n_validations=2,
        figure_dpi=80,
    )

    callback.on_validation_end(None, _module())
    assert not any(key.startswith("euv_tomography.") for key in logged)
    assert "validation/psnr.aia_valid" in logged
    logged.clear()

    callback.on_validation_end(None, _module())

    assert {key for key in logged if key.startswith("euv_tomography.")} == {
        "euv_tomography.aia_valid.channel_comparison",
        "euv_tomography.aia_valid.plasma_diagnostics",
        "euv_tomography.aia_valid.thermal_distribution",
        "euv_tomography.aia_valid.response_and_gains",
        "euv_tomography.aia_valid.ray_sampling",
    }
    assert callback.state_dict() == {"validation_count": 2}

    restored = EUVTomographyCallback(
        "aia_valid",
        "AIA",
        (2, 2),
        ({"id": "A94"}, {"id": "A171"}),
        ("A94",),
        _products(),
        every_n_validations=2,
    )
    restored.load_state_dict(callback.state_dict())
    assert restored.state_dict() == {"validation_count": 2}


def _diagnostic_titles(callback, module):
    outputs = callback._reshape_outputs(module.validation_outputs["aia_valid"])
    renderer = module.rendering.rendering_modules["AIA"]
    figure = callback._plot_plasma_diagnostics(outputs, renderer)
    if figure is None:
        return []
    return [axis.get_title() for axis in figure.axes if axis.get_title()]


def test_plasma_diagnostics_track_modeled_physics():
    products = _products()
    products["plasma_diagnostics"]["quantities"] = [
        "mean_log_temperature",
        "absorption_fraction",
        "cool_hydrogen_column",
    ]
    callback = EUVTomographyCallback(
        "aia_valid", "AIA", (2, 2), ({"id": "A94"}, {"id": "A171"}), ("171",), products,
    )

    module = _module()
    assert _diagnostic_titles(callback, module) == [
        "Mean log$_{10}$(T / K)", "Mean absorbed fraction",
    ]

    module.validation_outputs["aia_valid"]["cool_hydrogen_column_cm2"] = torch.tensor(
        [1e18, 2e18, 3e18, 4e18]
    )
    assert _diagnostic_titles(callback, module) == [
        "Mean log$_{10}$(T / K)", "Mean absorbed fraction", "Cool H column [cm$^{-2}$]",
    ]

    module.rendering.rendering_modules["AIA"].absorption = False
    assert _diagnostic_titles(callback, module) == [
        "Mean log$_{10}$(T / K)", "Cool H column [cm$^{-2}$]",
    ]


def test_emission_weighted_diagnostics_are_masked_where_nothing_emits():
    from sunerf.train.euv_callback import _emission_measure_mask

    callback = EUVTomographyCallback(
        "aia_valid", "AIA", (2, 2), ({"id": "A94"}, {"id": "A171"}), ("171",), _products(),
    )
    # The emission measure is required whenever a masked diagnostic is plotted.
    assert "emission_measure_cm5" in callback.validation_output_keys

    module = _module()
    outputs = module.validation_outputs["aia_valid"]
    outputs["emission_measure_cm5"] = torch.tensor([1e26, 2e26, 3e26, 1e22])
    outputs = callback._reshape_outputs(outputs)
    image_mask = np.ones((2, 2), dtype=bool)
    mask = _emission_measure_mask(np.squeeze(outputs["emission_measure_cm5"]), image_mask)
    assert mask.tolist() == [[True, True], [True, False]]

    renderer = module.rendering.rendering_modules["AIA"]
    figure = callback._plot_plasma_diagnostics(outputs, renderer)
    by_title = {axis.get_title(): axis for axis in figure.axes if axis.get_title()}
    masked = by_title["Mean log$_{10}$(T / K)"].images[0].get_array()
    unmasked = by_title["Column $N_e$ [cm$^{-2}$]"].images[0].get_array()
    assert np.ma.getmaskarray(masked).tolist() == [[False, False], [False, True]]
    assert not np.ma.getmaskarray(unmasked).any()


def test_normalize_matches_prediction_moments_to_observation():
    from sunerf.train.euv_callback import _match_mean_and_std

    observed = np.array([[1.0, 2.0], [3.0, np.nan]])
    predicted = np.array([[10.0, 30.0], [50.0, 70.0]])
    matched = _match_mean_and_std(predicted, observed)
    finite = np.isfinite(observed)
    assert np.isclose(np.mean(matched[finite]), np.mean(observed[finite]))
    assert np.isclose(np.std(matched[finite]), np.std(observed[finite]))

    callback = EUVTomographyCallback(
        "eui_valid",
        "AIA",
        (2, 2),
        ({"id": "A94"}, {"id": "A171"}),
        ("A94",),
        _products(),
        normalize=True,
    )
    outputs = callback._reshape_outputs(_module().validation_outputs["aia_valid"])
    figure = callback._plot_channel_comparison(outputs)
    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
    assert "A94 — observation" in titles
    assert "A94 — prediction (normalized)" in titles


def test_validation_metrics_on_full_images(monkeypatch):
    logged = {}
    monkeypatch.setattr(
        "sunerf.train.euv_callback.wandb.log", lambda payload: logged.update(payload)
    )
    products = _products()
    for product in products.values():
        product["enabled"] = False
    products["response_and_gains"]["enabled"] = True
    callback = EUVTomographyCallback(
        "aia_valid", "AIA", (16, 16), ({"id": "A94"}, {"id": "A171"}), ("A94",), products,
        every_n_validations=10,
    )
    # Metrics need the images even when no image figure is enabled.
    assert {"target_image", "pred_image", "valid_mask"} <= set(callback.validation_output_keys)

    rng = np.random.default_rng(0)
    target = rng.uniform(0.2, 0.8, size=(256, 2)).astype(np.float32)
    prediction = target.copy()
    prediction[:, 1] += 0.1
    valid_mask = np.ones((256, 2), dtype=bool)
    valid_mask[:128, 1] = False
    # Masked pixels must not count, even if they are far off.
    prediction[:128, 1] = 5.0
    module = SimpleNamespace(validation_outputs={"aia_valid": {
        "target_image": torch.from_numpy(target),
        "pred_image": torch.from_numpy(prediction),
        "valid_mask": torch.from_numpy(valid_mask),
    }})

    callback.on_validation_end(None, module)

    assert logged["validation/loss.aia_valid.A94"] == 0.0
    assert logged["validation/psnr.aia_valid.A94"] == np.inf
    assert np.isclose(logged["validation/ssim.aia_valid.A94"], 1.0)
    assert np.isclose(logged["validation/loss.aia_valid.A171"], 0.01, rtol=1e-4)
    assert np.isclose(logged["validation/psnr.aia_valid.A171"], 20.0, rtol=1e-4)
    assert logged["validation/ssim.aia_valid.A171"] < 1.0
    assert np.isclose(logged["validation/loss.aia_valid"], 0.005, rtol=1e-4)
    assert np.isclose(logged["validation/psnr.aia_valid"], -10 * np.log10(0.005), rtol=1e-4)
    assert not any(key.startswith("euv_tomography.") for key in logged)
