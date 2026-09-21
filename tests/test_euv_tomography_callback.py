from types import SimpleNamespace

import torch

from sunerf.train.euv_callback import EUVTomographyCallback


def _products():
    return {
        "channel_comparison": {
            "enabled": True,
            "rows": ["observation", "prediction", "residual", "relative_residual"],
            "stretch": "log",
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
    assert logged == []

    callback.on_validation_end(None, _module())

    assert set(logged) == {
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
