import numpy as np
import pytest
import torch

from sunerf.model.model import PlasmaModel


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (PlasmaModel, {"backend": "mlp", "dim": 8, "n_layers": 1, "encoding": "none"}),
        (PlasmaModel, {"backend": "siren", "dim": 8, "n_layers": 2}),
    ],
)
def test_physical_density_offset_initializes_trainable_coronal_density(model_class, kwargs):
    model = model_class(
        log_T=np.array([5.0, 6.0, 7.0]),
        density_offset_log10_cm3=8.0,
        **kwargs,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()

    coordinates = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    output = model(coordinates)

    # One density and one temperature per point; no thermal-width outputs.
    assert set(output) == {"total_ne", "total_log_ne", "mean_log_T"}
    torch.testing.assert_close(output["total_log_ne"], torch.tensor([[8.0]]))
    torch.testing.assert_close(output["total_ne"], torch.tensor([[1.0e8]]), rtol=1e-6, atol=0)
    output["total_ne"].sum().backward()
    density_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert density_gradients
    assert any(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in density_gradients)
    assert model.state_dict()["density_offset_log10_cm3"].item() == 8.0


@pytest.mark.parametrize("model_class", [PlasmaModel])
def test_density_offset_must_be_explicit_and_finite(model_class):
    with pytest.raises(TypeError, match="density_offset_log10_cm3"):
        model_class(log_T=[5.0, 6.0])
    with pytest.raises(ValueError, match="must be finite"):
        model_class(log_T=[5.0, 6.0], density_offset_log10_cm3=float("nan"))


@pytest.mark.parametrize(
    "model_class, kwargs",
    [
        (PlasmaModel, {"backend": "mlp", "dim": 8, "n_layers": 1, "encoding": "none"}),
        (PlasmaModel, {"backend": "siren", "dim": 8, "n_layers": 2}),
    ],
)
def test_temperature_is_bounded_and_initialized_in_the_emitting_corona(model_class, kwargs):
    model = model_class(
        log_T=np.linspace(4.0, 7.5, 71), density_offset_log10_cm3=8.0, **kwargs
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    coordinates = torch.tensor([[1.2, 0.0, 0.0, 0.0]])

    # A zero network output starts at log T = 6.1, not at the 5.75 midpoint
    # where no supported channel has temperature response.
    torch.testing.assert_close(model(coordinates)["mean_log_T"], torch.tensor([[6.1]]))

    explicit = model_class(
        log_T=[4.0, 7.5], density_offset_log10_cm3=8.0, initial_log_T=5.0, **kwargs
    )
    with torch.no_grad():
        for parameter in explicit.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    torch.testing.assert_close(
        explicit(coordinates)["mean_log_T"], torch.tensor([[5.0]]), rtol=1e-5, atol=1e-5
    )
    random_output = model_class(
        log_T=[4.0, 7.5], density_offset_log10_cm3=8.0, **kwargs
    )(torch.randn(64, 4))["mean_log_T"]
    assert random_output.min() >= 4.0 and random_output.max() <= 7.5
    with pytest.raises(ValueError, match="initial_log_T"):
        model_class(log_T=[4.0, 7.5], density_offset_log10_cm3=8.0, initial_log_T=8.0, **kwargs)


def _zeroed(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    return model


def test_hydrostatic_density_baseline_replaces_the_wind_power_law():
    radius = torch.tensor([1.0, 1.2, 1.5])
    coordinates = torch.stack([radius, *(torch.zeros(3),) * 3], dim=-1)
    options = {"log_T": [5.4, 7.5], "density_offset_log10_cm3": 8.0, "dim": 8, "n_layers": 2}

    wind = _zeroed(PlasmaModel(**options))(coordinates)["total_log_ne"][:, 0]
    hydrostatic = _zeroed(PlasmaModel(
        **options, density_profile={"type": "hydrostatic", "scale_height_rsun": 0.1}
    ))(coordinates)["total_log_ne"][:, 0]

    # Default: n ~ r**-2. Hydrostatic: n ~ exp[-(1/H0)(1 - 1/r)], identical at r = 1.
    torch.testing.assert_close(wind, 8.0 - 2.0 * torch.log10(radius))
    torch.testing.assert_close(
        hydrostatic, 8.0 - (1.0 - 1.0 / radius) / (0.1 * np.log(10.0)), rtol=1e-6, atol=1e-5
    )
    # 1.45 dex between 1.0 and 1.5 R_sun instead of 0.35 dex.
    assert (hydrostatic[0] - hydrostatic[2]).item() == pytest.approx(1.448, abs=2e-3)
    assert (wind[0] - wind[2]).item() == pytest.approx(0.352, abs=2e-3)

    steeper = _zeroed(PlasmaModel(
        **options, density_profile={"type": "power_law", "exponent": 4.0}
    ))(coordinates)["total_log_ne"][:, 0]
    torch.testing.assert_close(steeper, 8.0 - 4.0 * torch.log10(radius))


@pytest.mark.parametrize("profile, match", [
    ({"type": "exponential"}, "must be one of"),
    ({"type": "hydrostatic"}, "scale_height_rsun"),
    ({"type": "hydrostatic", "scale_height_rsun": 0.0}, "scale_height_rsun"),
    ({"type": "hydrostatic", "scale_height_rsun": 0.1, "exponent": 2.0}, "unsupported fields"),
])
def test_density_profile_is_validated(profile, match):
    with pytest.raises(ValueError, match=match):
        PlasmaModel(log_T=[5.4, 7.5], density_offset_log10_cm3=8.0, density_profile=profile)


def test_generic_model_per_axis_positional_encoding_and_density_profile():
    options = dict(
        log_T=[5.4, 7.5], density_offset_log10_cm3=8.0, backend="mlp", dim=8, n_layers=1, num_frequencies=[64, 64, 64, 16], max_frequencies=[8, 8, 8, 5],
        density_profile={"type": "hydrostatic", "scale_height_rsun": 0.1},
    )
    model = PlasmaModel(**options)
    assert model.backend.d_in[0].d_output == 2 * (3 * 64 + 16)
    assert float(model.backend.d_in[0].frequencies[0][-1]) == pytest.approx(256 * torch.pi)
    assert float(model.backend.d_in[0].frequencies[3][-1]) == pytest.approx(32 * torch.pi)

    coordinates = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.5, 0.0, 0.0]])
    generic = _zeroed(model)(coordinates)["total_log_ne"]
    siren = _zeroed(PlasmaModel(
        log_T=[5.4, 7.5], density_offset_log10_cm3=8.0, dim=8, n_layers=2,
        density_profile=options["density_profile"],
    ))(coordinates)["total_log_ne"]
    assert torch.allclose(generic, siren)


def test_state_without_backend_prefix_is_migrated():
    options = dict(log_T=[5.4, 7.5], density_offset_log10_cm3=8.0, dim=8, n_layers=2)
    source, target = PlasmaModel(**options), PlasmaModel(**options)
    legacy = {key.removeprefix("backend."): value for key, value in source.state_dict().items()}
    target.load_state_dict(legacy, strict=True)
    coordinates = torch.rand(4, 4) + 1.0
    torch.testing.assert_close(target(coordinates)["total_log_ne"], source(coordinates)["total_log_ne"])
