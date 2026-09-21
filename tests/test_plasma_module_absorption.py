import numpy as np
import pytest
import torch

from sunerf.absorption import ABSORPTION_SPECIES, AbsorptionBundle
from sunerf.model.plasma import PlasmaSuNeRFModule
from sunerf.response import ResponseArtifact


ABUNDANCE = {"name": "test-coronal", "version": "1", "sha256": "a" * 64}


def _artifacts(tmp_path, bundle_abundance=ABUNDANCE):
    response = tmp_path / "response.npz"
    ResponseArtifact(
        channels=("171", "195"),
        log_temperature=np.array([5.0, 6.0, 7.0]),
        log_density=np.array([8.0, 10.0]),
        response=np.full((2, 2, 3), 1.0e-26),
        response_unit="cm5 DN s-1 pix-1",
        emission_measure_convention="ne2",
        provenance={"builder": "unit-test", "spectral_emissivity": {"abundance": ABUNDANCE}},
    ).save(response)
    bundle = tmp_path / "bundle.npz"
    AbsorptionBundle(
        species=ABSORPTION_SPECIES,
        log_temperature=np.array([4.0, 5.0, 7.0]),
        ion_fraction=np.array([[1.0, 1e-5, 1e-8], [1.0, 1e-4, 1e-12], [1e-9, 0.1, 1e-6]]),
        electron_per_hydrogen=np.array([2.0e-3, 1.15, 1.18]),
        abundance_per_hydrogen=np.array([1.0, 0.08, 0.08]),
        instrument_keys=("euvi_a", "euvi_a"),
        channels=("171", "195"),
        effective_cross_section_cm2=np.array([[5e-20, 4e-19, 7e-19], [7e-20, 6e-19, 1e-18]]),
        provenance={"provider": "unit-test", "abundance": {"model": bundle_abundance}},
    ).save(bundle)
    return response, bundle


def _module(response, bundle, absorption=True, cool_absorber=False, **module_options):
    return PlasmaSuNeRFModule(
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        instruments_config=[{
            "type": "plasma", "key": "EUVI-A",
            "temperature_response": {
                "artifact": str(response), "channels": ["171", "195"],
                "temperature_cutoff": {"T_cut_K": 4.0e5, "delta_T_K": 5.0e4},
            },
        }],
        absorption_config=(
            {"type": "photoionization", "artifact": str(bundle),
             "hydrogen_density_convention": "fully_ionized_proxy"}
            if absorption else {"type": None}
        ),
        model_config={
            "type": "siren", "density_offset_log10_cm3": 9.0, "dim": 16, "n_layers": 2,
            **({"cool_absorber": True, "cool_density_offset_log10_cm3": 7.0}
               if cool_absorber else {}),
            "temperature_grid": {"log10_K_min": 4.0, "log10_K_max": 7.5, "step_dex": 0.5},
        },
        sampling_config={"type": "spherical", "min_distance": 1.0, "max_distance": 1.5,
                         "n_samples": 8, "perturb": False},
        hierarchical_sampling_config={"type": "hierarchical", "n_samples": 8, "perturb": False},
        lambda_config={"image": 1.0, "regularization": 0.0},
        validation_dataset_mapping={},
        **module_options,
    )


def _batch():
    origins = torch.tensor([[215.0, 0.0, 0.0]]).repeat(4, 1)
    directions = torch.nn.functional.normalize(
        torch.tensor([[-1.0, 0.0, 0.0], [-1.0, 0.002, 0.0], [-1.0, 0.0, 0.0055], [-1.0, 0.5, 0.0]]),
        dim=-1,
    )
    return {"EUVI-A": {
        "rays": torch.stack([origins, directions], dim=1),
        "time": torch.zeros(4, 1), "instrument": "EUVI-A",
    }}


def test_module_renders_pointwise_plasma_with_packaged_style_absorption(tmp_path):
    response, bundle = _artifacts(tmp_path)
    torch.manual_seed(0)
    module = _module(response, bundle).train()

    output = module.rendering(_batch())
    image = output["model_out"]["EUVI-A"]["image"]

    assert image.shape == (4, 2)
    assert torch.isfinite(image).all()
    assert output["ray_valid"].tolist() == [True, True, True, False]
    assert torch.all(image[:3] > 0) and torch.all(image[3] == 0)
    image.sum().backward()
    gradients = [p.grad for p in module.rendering.model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)

    metadata = module.instrument_metadata["EUVI-A"]["absorption"]
    assert metadata["type"] == "photoionization" and metadata["species"] == list(ABSORPTION_SPECIES)
    renderer = module.rendering.rendering_modules["EUVI-A"]
    assert renderer.temperature_cutoff == {"T_cut_K": 4.0e5, "delta_T_K": 5.0e4}
    assert renderer.absorption_model.hydrogen_density_convention == "fully_ionized_proxy"


def test_cold_dense_plasma_attenuates_the_image_only_with_absorption(tmp_path):
    response, bundle = _artifacts(tmp_path)

    def render(absorption):
        torch.manual_seed(1)
        module = _module(response, bundle, absorption=absorption).eval()
        model = module.rendering.model

        def cold_front_hot_back(points):
            # Cold, dense plasma on the observer side (x > 0), hot corona behind it.
            front = (points[..., 0:1] > 0).to(points.dtype)
            log_ne = 9.0 + 2.5 * front
            return {"total_ne": 10.0**log_ne, "total_log_ne": log_ne,
                    "mean_log_T": 6.0 - 2.0 * front}

        model.forward = cold_front_hot_back
        with torch.no_grad():
            return module.rendering(_batch(), diagnostics=True)["model_out"]["EUVI-A"]

    plain, absorbed = render(False), render(True)
    # Limb ray (index 2) passes through both half spaces; the cold front does not
    # emit (cutoff) but attenuates the hot emission behind it.
    assert absorbed["image"][2, 0] < 0.9 * plain["image"][2, 0]
    assert torch.all(absorbed["optical_depth"][2] > 0.1)
    assert torch.all(plain["optical_depth"] == 0)


def test_module_rejects_absorption_built_with_other_abundances(tmp_path):
    response, bundle = _artifacts(
        tmp_path, bundle_abundance={"name": "photospheric", "version": "1", "sha256": "b" * 64}
    )
    with pytest.raises(ValueError, match="does not match absorption bundle abundance"):
        _module(response, bundle)


def test_light_travel_time_evaluates_samples_at_their_emission_time(tmp_path):
    response, bundle = _artifacts(tmp_path)
    delayed = _module(response, bundle, absorption=False, light_travel_time=True)
    instantaneous = _module(response, bundle, absorption=False)

    assert delayed.construction_spec["light_travel_time"] is True
    assert instantaneous.construction_spec["light_travel_time"] is False
    origin = torch.tensor([[215.0, 0.0, 0.0]])
    points = torch.tensor([[[1.2, 0.0, 0.0], [-1.2, 0.0, 0.0]]])
    times = torch.zeros(1, 1)

    sample_times = delayed.rendering.add_sample_times(points, origin, times)[..., 3]
    plain_times = instantaneous.rendering.add_sample_times(points, origin, times)[..., 3]

    # One solar radius is 2.3206 light seconds; model time is in days.
    seconds = -sample_times * 86400.0
    torch.testing.assert_close(
        seconds, torch.tensor([[213.8 * 2.32061, 216.2 * 2.32061]]), rtol=1e-3, atol=0
    )
    # The far side of the shell is seen 2.4 R_sun / c = 5.6 s earlier than the near side.
    assert (seconds[0, 1] - seconds[0, 0]).item() == pytest.approx(2.4 * 2.32061, rel=5e-2)
    torch.testing.assert_close(plain_times, torch.zeros(1, 2))


def test_separate_cool_absorber_is_rendered_regularized_and_recorded(tmp_path):
    response, bundle = _artifacts(tmp_path)
    torch.manual_seed(0)
    module = _module(response, bundle, cool_absorber=True).train()
    renderer = module.rendering.rendering_modules["EUVI-A"]

    lean = module.rendering(_batch())["model_out"]["EUVI-A"]
    # The training loop receives the LOS hydrogen column for the sparsity prior.
    assert lean["cool_hydrogen_column_cm2"].shape == (4,)
    assert torch.all(lean["cool_hydrogen_column_cm2"][:3] > 0)
    lean["cool_hydrogen_column_cm2"].sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().max() > 0
        for parameter in module.rendering.model.parameters()
    )

    metadata = module.instrument_metadata["EUVI-A"]["absorption"]
    assert metadata["cool_ion_fractions"] == {"H_I": 0.7, "He_I": 0.7, "He_II": 0.3}
    assert len(metadata["cool_cross_section_per_hydrogen_cm2"]) == 2
    assert module.construction_spec["cool_column_scale_cm2"] == 1.0e19
    assert module.lambda_cool_absorber == 1.0e-4

    # A dense cool layer on the observer side dims the corona behind it without
    # changing the emitting plasma, and the far-side rays are refined there.
    model = module.rendering.model.eval()
    module.eval()

    def state(points, cool_density):
        log_ne = torch.full_like(points[..., 0:1], 9.0)
        front = (points[..., 0:1] > 0).to(points.dtype)
        return {"total_ne": 10.0**log_ne, "total_log_ne": log_ne,
                "mean_log_T": torch.full_like(log_ne, 6.0),
                "cool_hydrogen_density": cool_density * front}

    with torch.no_grad():
        model.forward = lambda points: state(points, 0.0)
        plain = module.rendering(_batch(), diagnostics=True)["model_out"]["EUVI-A"]
        model.forward = lambda points: state(points, 2.0e9)  # optically thin enough
        # that the channels have not all saturated to the unabsorbed foreground
        dimmed = module.rendering(_batch(), diagnostics=True)["model_out"]["EUVI-A"]
    kappa = renderer.absorption_model.cool_cross_section_per_hydrogen_cm2
    assert torch.all(dimmed["image"][2] < 0.95 * plain["image"][2])
    torch.testing.assert_close(
        dimmed["cool_optical_depth"][2],
        dimmed["cool_hydrogen_column_cm2"][2] * kappa, rtol=1e-4, atol=0,
    )
    # The channel with the larger cross section is dimmed more.
    ratio = dimmed["image"][2] / plain["image"][2]
    assert (ratio[1] < ratio[0]) == bool(kappa[1] > kappa[0])


def test_cool_absorber_requires_photoionization_absorption(tmp_path):
    response, bundle = _artifacts(tmp_path)
    with pytest.raises(ValueError, match="cool_absorber requires absorption.type: photoionization"):
        _module(response, bundle, absorption=False, cool_absorber=True)
