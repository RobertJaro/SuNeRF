import numpy as np
import pytest
import torch

from sunerf.physics.euv import (
    numpy_trapezoid_node_weights,
    torch_trapezoid_node_weights,
)
from sunerf.rendering.base_tracing import _mask_rendered_output
from sunerf.rendering.plasma import PlasmaRadiativeTransfer
from sunerf.model.plasma import PlasmaSuNeRFModule
from sunerf.response import ResponseArtifact


def _write_response(
    path,
    *,
    log_temperature=(5.0, 6.0, 7.0),
    response=None,
    channels=("A",),
    log_density=None,
    convention="ne2",
):
    if response is None:
        response = np.ones((len(channels), len(log_temperature)))
    ResponseArtifact(
        channels=channels,
        log_temperature=np.asarray(log_temperature),
        log_density=None if log_density is None else np.asarray(log_density),
        response=np.asarray(response),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention=convention,
        provenance={"builder": "unit-test"},
    ).save(path)


def _ray_state(log_temperature, *, density=3.0, z_vals=(0.0, 1.0, 3.0)):
    n_samples = len(z_vals)
    n_temperature = len(log_temperature)
    z_vals = torch.as_tensor(np.asarray([z_vals]), dtype=torch.float32)
    query_points = torch.zeros((1, n_samples, 4), dtype=torch.float32)
    query_points[0, :, 0] = z_vals[0]
    return {
        "total_ne": torch.full((1, n_samples, 1), density),
        "mean_log_T": torch.full((1, n_samples, 1), float(log_temperature[n_temperature // 2])),
        "total_log_ne": torch.full((1, n_samples, 1), np.log10(density)),
        "z_vals": z_vals,
        "rays_d": torch.tensor([[1.0, 0.0, 0.0]]),
        "query_points": query_points,
    }


def test_nonuniform_node_weights_integrate_the_full_interval():
    nodes = np.array([0.0, 1.0, 3.0, 6.0])
    expected = np.array([0.5, 1.5, 2.5, 1.5])

    np.testing.assert_allclose(numpy_trapezoid_node_weights(nodes), expected)
    torch.testing.assert_close(
        torch_trapezoid_node_weights(torch.tensor(nodes)), torch.tensor(expected)
    )
    assert expected.sum() == nodes[-1] - nodes[0]


def test_renderer_uses_cm_path_quadrature_and_dl_weighted_diagnostics(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path, response=np.full((1, 3), 2.0))
    renderer = PlasmaRadiativeTransfer(
        {
            "artifact": str(path),
            "channels": ["A"],
            "model_length_unit_cm": 10.0,
        },
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()

    output = renderer(**_ray_state((5.0, 6.0, 7.0)))

    # K=2, n_e^2=9, and the path length is (3 model units)*10 cm.
    torch.testing.assert_close(output["image"], torch.tensor([[540.0]]))
    torch.testing.assert_close(output["column_electron_density_cm2"], torch.tensor([[90.0]]))
    torch.testing.assert_close(output["emission_measure_cm5"], torch.tensor([270.0]))
    torch.testing.assert_close(
        output["weights"], torch.tensor([[1 / 6, 1 / 2, 1 / 3]]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(output["mean_log_T"], torch.tensor([[6.0]]))
    torch.testing.assert_close(output["dem"].sum(-1), output["emission_measure_cm5"])


def test_invalid_shell_ray_uses_safe_quadrature_until_outer_mask(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path)
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 1.0},
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()
    state = _ray_state((5.0, 6.0, 7.0), z_vals=(0.0, 0.0, 0.0))
    ray_valid = torch.tensor([False])

    output = renderer(**state, ray_valid=ray_valid)
    masked = _mask_rendered_output(output, ray_valid)

    assert torch.isfinite(output["image"]).all()
    torch.testing.assert_close(masked["image"], torch.zeros_like(masked["image"]))
    assert not masked["ray_valid"].item()


def test_isothermal_slab_matches_bilinear_response_and_is_stable_under_refinement(tmp_path):
    path = tmp_path / "response.npz"
    response = np.array([[[1.0, 3.0, 5.0], [2.0, 6.0, 10.0]]])  # (channel, density, T)
    _write_response(
        path, log_temperature=(5.0, 6.0, 7.0), log_density=(8.0, 10.0), response=response
    )
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 2.0},
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()

    images = []
    for n_samples in (2, 9, 33):
        state = _ray_state((5.0, 6.0, 7.0), z_vals=np.linspace(0.0, 4.0, n_samples))
        state["mean_log_T"][:] = 6.5
        state["total_log_ne"][:] = 9.0
        state["total_ne"][:] = 1.0e9
        images.append(renderer(**state)["image"])

    # G(6.5, 9) = mean of the four corner values; I = G n_e^2 L with L = 8 cm.
    expected = 0.25 * (3.0 + 5.0 + 6.0 + 10.0) * 1.0e18 * 8.0
    for image in images:
        torch.testing.assert_close(image, torch.tensor([[expected]]), rtol=1e-5, atol=0)


def test_response_is_zero_outside_temperature_support_and_density_is_clamped(tmp_path):
    path = tmp_path / "response.npz"
    response = np.array([[[1.0, 1.0], [4.0, 4.0]]])
    _write_response(
        path, log_temperature=(5.0, 7.0), log_density=(8.0, 10.0), response=response
    )
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 1.0},
        np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    ).eval()
    log_temperature = torch.tensor([[[4.9], [6.0], [7.1]]])
    log_density = torch.tensor([[[6.0], [12.0], [9.0]]])

    value = renderer.response_at(log_temperature, log_density)

    torch.testing.assert_close(value[0, :, 0], torch.tensor([0.0, 4.0, 0.0]))


def test_temperature_cutoff_scales_emission_but_not_opacity(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path, log_temperature=(4.0, 5.0, 6.0, 7.0), response=np.ones((1, 4)))
    config = {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 1.0}
    grid = np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32)
    plain = PlasmaRadiativeTransfer(config, grid).eval()
    cut = PlasmaRadiativeTransfer(
        {**config, "temperature_cutoff": {"T_cut_K": 4.0e5, "delta_T_K": 5.0e4}}, grid
    ).eval()

    temperatures = torch.tensor([1.0e5, 4.0e5, 1.0e6])
    factor = cut.emission_cutoff(torch.log10(temperatures)[None, :, None])[0, :, 0]
    expected = 0.5 * (1.0 + torch.tanh((temperatures - 4.0e5) / 5.0e4))
    torch.testing.assert_close(factor, expected, rtol=1e-4, atol=1e-6)
    assert factor[0] < 1e-4 and factor[2] > 0.9999

    state = _ray_state((4.0, 5.0, 6.0, 7.0), z_vals=(0.0, 1.0))
    state["mean_log_T"][:] = float(np.log10(4.0e5))
    torch.testing.assert_close(
        cut(**state)["image"], 0.5 * plain(**state)["image"], rtol=1e-4, atol=0
    )
    with pytest.raises(ValueError, match="exactly T_cut_K and delta_T_K"):
        PlasmaRadiativeTransfer({**config, "temperature_cutoff": {"T_cut_K": 1.0}}, grid)


class _ColdSlabOpacity(torch.nn.Module):
    """Channel opacities that are non-zero only in cold plasma."""

    is_deterministic_physical = False

    def __init__(self, cross_section_cm2):
        super().__init__()
        self.register_buffer("cross_section", torch.as_tensor(cross_section_cm2))

    def opacity(self, *, total_ne, mean_log_T, **kwargs):
        cold = (mean_log_T < 5.0).to(total_ne.dtype)
        return {"alpha_cm_inverse": total_ne * cold * self.cross_section}


def test_slab_stack_attenuates_only_emission_behind_the_absorber(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(
        path, log_temperature=(5.5, 6.0, 6.5), channels=("A", "B"), response=np.ones((2, 3))
    )
    sigma = torch.tensor([1.0e-2, 3.0e-2])
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A", "B"], "model_length_unit_cm": 1.0},
        np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        absorption_model=_ColdSlabOpacity(sigma),
    ).eval()

    # Near emitter [0, 1], cold absorber [1, 2], far emitter [2, 3]. The thin
    # transition intervals keep the trapezoid sums exact to O(1e-3).
    eps = 1.0e-4
    z_vals = (0.0, 1.0 - eps, 1.0, 2.0, 2.0 + eps, 3.0)
    state = _ray_state((5.5, 6.0, 6.5), density=1.0, z_vals=z_vals)
    hot, cold_density = 1.0, 10.0
    density = torch.tensor([hot, hot, cold_density, cold_density, hot, hot])
    state["total_ne"][0, :, 0] = density
    state["total_log_ne"][0, :, 0] = torch.log10(density)
    state["mean_log_T"][0, :, 0] = torch.tensor([6.0, 6.0, 4.0, 4.0, 6.0, 6.0])

    output = renderer(**state, diagnostics=True)

    tau = sigma * cold_density * 1.0
    expected = 1.0 + torch.exp(-tau)  # near slab unattenuated, far slab exp(-tau)
    torch.testing.assert_close(output["image"][0], expected, rtol=2e-3, atol=0)
    torch.testing.assert_close(output["optical_depth"][0], tau, rtol=2e-3, atol=0)
    # The channel with the larger cross section loses more of the far emission.
    assert output["image"][0, 1] < output["image"][0, 0]
    torch.testing.assert_close(
        output["transmission"][0, 1], torch.ones(2), rtol=0, atol=1e-6
    )


def test_los_emission_measure_histogram_conserves_emission_measure(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path)
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 1.0},
        np.array([5.0, 5.5, 7.0], dtype=np.float32),
    ).eval()
    state = _ray_state((5.0, 6.0, 7.0), density=2.0, z_vals=(0.0, 2.0))
    state["mean_log_T"][:] = 5.25

    output = renderer(**state, diagnostics=True)

    torch.testing.assert_close(output["emission_measure_cm5"], torch.tensor([8.0]))
    torch.testing.assert_close(output["dem"], torch.tensor([[4.0, 4.0, 0.0]]))
    weights = torch_trapezoid_node_weights(torch.tensor([5.0, 5.5, 7.0]))
    torch.testing.assert_close(
        (output["differential_emission_measure_cm5_per_dex"] * weights).sum(-1),
        output["emission_measure_cm5"],
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"scaling": 0.2}, "temperature_response.scaling"),
        ({"learnable": True, "reference_channel": None}, "non-null"),
        (
            {
                "learnable": True,
                "reference_channel": "A",
                "gain_constraint": "zero_mean",
            },
            "mutually exclusive",
        ),
        ({"learnable": True, "gain_constraint": "unknown"}, "must be zero_mean"),
    ],
)
def test_renderer_rejects_ambiguous_or_unbounded_calibration_controls(
    tmp_path, override, match
):
    path = tmp_path / "response.npz"
    _write_response(path)

    with pytest.raises(ValueError, match=match):
        PlasmaRadiativeTransfer(
            {"artifact": str(path), "channels": ["A"], **override},
            np.array([5.0, 6.0, 7.0], dtype=np.float32),
        )


def test_density_interpolation_and_per_channel_gain_bounds(tmp_path):
    path = tmp_path / "response.npz"
    response = np.array(
        [
            [[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]],
            [[2.0, 2.0, 2.0], [6.0, 6.0, 6.0]],
        ]
    )
    _write_response(
        path,
        channels=("A", "B"),
        log_density=(8.0, 10.0),
        response=response,
    )
    renderer = PlasmaRadiativeTransfer(
        {
            "artifact": str(path),
            "channels": ["A", "B"],
            "model_length_unit_cm": 1.0,
            "learnable": True,
            "gain_limit_dex": 0.1,
        },
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()
    with torch.no_grad():
        renderer.instrument_scaling.copy_(torch.tensor([100.0, -100.0]))

    assert renderer.effective_instrument_scaling[0] <= 0.100001
    assert renderer.effective_instrument_scaling[1] >= -0.100001
    torch.testing.assert_close(
        renderer.instrument_gain_delta_dex.sum(), torch.tensor(0.0), atol=1e-7, rtol=0
    )

    state = _ray_state((5.0, 6.0, 7.0), density=1.0e9, z_vals=(0.0, 1.0))
    output = renderer(**state)
    # Midway between the density nodes: G = (2, 4), scaled by n_e^2 = 1e18.
    expected = 1.0e18 * torch.tensor([[2.0 * 10.0**0.1, 4.0 * 10.0**-0.1]])
    torch.testing.assert_close(output["image"], expected, rtol=1e-5, atol=0)


def test_reference_channel_has_exactly_fixed_gain(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(
        path,
        channels=("A", "B"),
        response=np.ones((2, 3)),
    )
    renderer = PlasmaRadiativeTransfer(
        {
            "artifact": str(path),
            "channels": ["A", "B"],
            "model_length_unit_cm": 1.0,
            "learnable": True,
            "gain_limit_dex": 0.2,
            "reference_channel": "B",
        },
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()
    with torch.no_grad():
        renderer.instrument_scaling.copy_(torch.tensor([100.0, 100.0]))

    assert renderer.gain_identifiability == "fixed_reference_channel"
    torch.testing.assert_close(
        renderer.instrument_gain_delta_dex, torch.tensor([0.2, 0.0])
    )
    torch.testing.assert_close(renderer.instrument_gain[1], torch.tensor(1.0))


def test_common_instrument_gain_is_uniform_and_global_reference_is_fixed(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path, channels=("A", "B"), response=np.ones((2, 3)))
    config = {
        "artifact": str(path),
        "channels": ["A", "B"],
        "model_length_unit_cm": 1.0,
        "learnable": True,
        "reference_channel": "A",
        "common_gain_limit_dex": 0.5,
    }
    renderer = PlasmaRadiativeTransfer(
        config,
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()
    with torch.no_grad():
        renderer.common_instrument_scaling.fill_(100.0)
    torch.testing.assert_close(
        renderer.instrument_gain,
        torch.full((2,), 10.0**0.5),
        rtol=1e-6,
        atol=1e-6,
    )

    reference = PlasmaRadiativeTransfer(
        {**config, "global_reference": True},
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).eval()
    assert not reference.common_instrument_scaling.requires_grad
    torch.testing.assert_close(reference.common_gain_delta_dex, torch.tensor(0.0))
    torch.testing.assert_close(reference.instrument_gain, torch.ones(2))


def test_instrument_scaling_divides_physical_images_by_fixed_channel_divisors():
    from sunerf.train.scaling import ImageLinearScaling

    physical_prediction = torch.tensor([[10.0, 40.0]])
    scaling = ImageLinearScaling(divisor=[10.0, 20.0])

    torch.testing.assert_close(scaling(physical_prediction), torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(physical_prediction, torch.tensor([[10.0, 40.0]]))
    # Rebuilt from the configuration, so existing checkpoints stay loadable.
    assert "divisor" not in scaling.state_dict()


def test_training_output_is_lean_unless_diagnostics_are_requested(tmp_path):
    path = tmp_path / "response.npz"
    _write_response(path)
    renderer = PlasmaRadiativeTransfer(
        {"artifact": str(path), "channels": ["A"], "model_length_unit_cm": 1.0},
        np.array([5.0, 6.0, 7.0], dtype=np.float32),
    ).train()
    state = _ray_state((5.0, 6.0, 7.0))

    lean = renderer(**state)
    diagnostic = renderer(**state, diagnostics=True)

    assert set(lean) == {
        "image",
        "weights",
        "mean_absorption",
        "em",
        "distance",
        "calibration_regularization",
        "instrument_gain_delta_dex",
        "common_gain_delta_dex",
    }
    assert "differential_emission_measure_cm5_per_dex" not in lean
    assert "transmission" not in lean
    assert "differential_emission_measure_cm5_per_dex" in diagnostic
    assert "transmission" in diagnostic

    renderer.eval()
    evaluation_image_only = renderer(**state, diagnostics=False)
    assert set(evaluation_image_only) == set(lean)
