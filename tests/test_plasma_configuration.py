import copy
from pathlib import Path

import pytest

from sunerf.configuration import PlasmaConfigError, validate_plasma_config
from sunerf.train.util import load_yaml_config


def _config():
    return {
        "schema_version": 2,
        "base_path": "/tmp/plasma",
        "instruments": [
            {
                "type": "plasma",
                "key": "AIA",
                "scaling": {"type": "asinh", "divisor": 1.0},
                "temperature_response": {
                    "artifact": "/responses/aia.sunerf.npz",
                    "channels": ["A94", "A171"],
                },
            }
        ],
        "data": {
            "Rs_per_ds": 1.0,
            "seconds_per_dt": 86400.0,
            "batch_size": 1024,
            "num_workers": 4,
            "holdout": {"strategy": "center", "count": 1},
            "train_datasets": [
                {
                    "type": "AIA",
                    "key": "aia_train",
                    "instrument_key": "AIA",
                    "data_path": "/data/aia/*.prepared.fits",
                    "wavelengths": [94, 171],
                    "strict_metadata": True,
                }
            ],
            "valid_datasets": [
                {
                    "type": "AIA",
                    "key": "aia_valid",
                    "instrument_key": "AIA",
                    "data_path": "/data/aia/*.prepared.fits",
                    "wavelengths": [94, 171],
                    "strict_metadata": True,
                }
            ],
        },
        "sampling": {"type": "spherical", "min_distance": 1.0, "max_distance": 1.5, "n_samples": 64},
        "model": {
            "density_offset_log10_cm3": 8.0,
            "temperature_grid": {
                "log10_K_min": 4.0,
                "log10_K_max": 9.0,
                "step_dex": 0.05,
            }
        },
        "module": {
            "regularization_density_scale_cm3": 1.0e8,
            "lr_config": {"start": 1.0e-3, "end": 1.0e-4, "iterations": 100_000},
        },
        "training": {"epochs": 10, "log_every_n_steps": 100},
        "lambda": {
            "image": 1.0,
            "regularization": 1.0e-3,
            "absorption": 0.0,
            "calibration": 1.0e-4,
        },
    }


def test_validate_plasma_config_returns_defensive_copy():
    config = _config()
    validated = validate_plasma_config(config)
    validated["data"]["batch_size"] = 1
    assert config["data"]["batch_size"] == 1024


def test_photoionization_absorption_requires_bundle_and_zero_regularizer():
    config = _config()
    config["absorption"] = {
        "type": "photoionization",
        "artifact": "/data/absorption/h-he.npz",
    }
    validated = validate_plasma_config(config)
    assert validated["absorption"]["artifact"].endswith("h-he.npz")

    config["lambda"]["absorption"] = 1.0e-4
    with pytest.raises(PlasmaConfigError, match="must be zero"):
        validate_plasma_config(config)

    config = _config()
    config["absorption"] = {"type": "photoionization"}
    with pytest.raises(PlasmaConfigError, match="artifact"):
        validate_plasma_config(config)


def test_absorption_rejects_unknown_provider_fields_and_types():
    config = _config()
    config["absorption"] = {
        "type": "photoionization",
        "artifact": "/data/absorption/h-he.npz",
        "learnable": True,
    }
    with pytest.raises(PlasmaConfigError, match="unsupported fields"):
        validate_plasma_config(config)

    config["absorption"] = {"type": "opaque-plugin"}
    with pytest.raises(PlasmaConfigError, match="absorption.type"):
        validate_plasma_config(config)


def test_euv_callback_defaults_cover_every_validation_dataset():
    validated = validate_plasma_config(_config())
    callback = validated["callbacks"]["euv_tomography"]

    assert callback["enabled"] is True
    assert callback["datasets"] == {
        "aia_valid": {"channels": ["94", "171"]}
    }
    assert callback["products"]["channel_comparison"]["enabled"] is True
    assert callback["products"]["thermal_distribution"]["enabled"] is True
    assert callback["products"]["ray_sampling"]["enabled"] is False


def test_euv_callback_resolves_configured_channel_aliases():
    config = _config()
    config["callbacks"] = {
        "euv_tomography": {
            "every_n_validations": 3,
            "datasets": {"aia_valid": {"channels": [171]}},
            "products": {
                "channel_comparison": {
                    "rows": ["prediction", "relative_residual"],
                    "stretch": "linear",
                },
                "ray_sampling": {"enabled": True, "pixel_fraction": [0.5, 0.75]},
            },
        }
    }

    callback = validate_plasma_config(config)["callbacks"]["euv_tomography"]

    assert callback["every_n_validations"] == 3
    assert callback["datasets"]["aia_valid"]["channels"] == ["171"]
    assert callback["products"]["channel_comparison"]["rows"] == [
        "prediction", "relative_residual"
    ]
    assert callback["products"]["ray_sampling"]["enabled"] is True


@pytest.mark.parametrize(
    ("callback_config", "match"),
    [
        ({"datasets": {"missing": {"channels": "all"}}}, "unknown validation"),
        ({"datasets": {"aia_valid": {"channels": [193]}}}, "match exactly one"),
        ({"products": {"unknown": {"enabled": True}}}, "unsupported products"),
        (
            {"products": {"plasma_diagnostics": {"quantities": ["density"]}}},
            "unsupported values",
        ),
        ({"every_n_validations": 0}, "integer >= 1"),
    ],
)
def test_euv_callback_config_rejects_invalid_plot_contracts(callback_config, match):
    config = _config()
    config["callbacks"] = {"euv_tomography": callback_config}

    with pytest.raises(PlasmaConfigError, match=match):
        validate_plasma_config(config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda cfg: cfg.update(schema_version=1), "schema_version"),
        (lambda cfg: cfg["data"].pop("seconds_per_dt"), "seconds_per_dt"),
        (lambda cfg: cfg["data"].update(Rs_per_ds=2.0), "Rs_per_ds == 1.0"),
        (lambda cfg: cfg["data"]["train_datasets"][0].update(wavelengths=[171, 94]), "channel order"),
        (lambda cfg: cfg["instruments"][0].update(type="emission"), "unsupported"),
    ],
)
def test_validate_plasma_config_rejects_invalid_contracts(mutation, match):
    config = _config()
    mutation(config)
    with pytest.raises(PlasmaConfigError, match=match):
        validate_plasma_config(config)


def test_schema_v2_rejects_legacy_response_inputs():
    config = _config()
    response = config["instruments"][0]["temperature_response"]
    response.pop("artifact")
    response.update(file="legacy.npz", format="legacy_npz", normalization=1.0)

    with pytest.raises(PlasmaConfigError, match="unsupported"):
        validate_plasma_config(config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda cfg: cfg["data"]["train_datasets"][0].pop("instrument_key"),
            "instrument_key",
        ),
        (
            lambda cfg: cfg["data"]["train_datasets"][0].update(type="BOGUS"),
            "type must be one of",
        ),
        (
            lambda cfg: cfg["instruments"][0]["temperature_response"].update(artifact=True),
            "artifact must be a non-empty path string",
        ),
        (
            lambda cfg: cfg["instruments"][0]["temperature_response"].update(
                model_length_unit_cm=1.0
            ),
            "cannot override model length",
        ),
    ],
)
def test_schema_v2_rejects_values_that_would_fail_after_side_effects(mutation, match):
    config = _config()
    mutation(config)
    with pytest.raises(PlasmaConfigError, match=match):
        validate_plasma_config(config)


def test_physical_response_cannot_be_renormalized_and_image_scaling_is_explicit():
    config = _config()
    config["instruments"][0]["temperature_response"]["normalization"] = 2.0
    with pytest.raises(PlasmaConfigError, match="renormalize"):
        validate_plasma_config(config)

    config = _config()
    config["instruments"][0]["scaling"].pop("divisor")
    with pytest.raises(PlasmaConfigError, match="scaling.divisor"):
        validate_plasma_config(config)

    config = _config()
    config["instruments"][0]["temperature_response"]["scaling"] = 0.2
    with pytest.raises(PlasmaConfigError, match="temperature_response.scaling"):
        validate_plasma_config(config)

    config = _config()
    divisor = {"A94": 10_000, "A171": 20_000}
    config["instruments"][0]["scaling"]["divisor"] = divisor
    validated = validate_plasma_config(config)
    assert validated["instruments"][0]["scaling"]["divisor"] == divisor

    config["instruments"][0]["scaling"]["divisor"]["A94"] = 0
    with pytest.raises(PlasmaConfigError, match="positive"):
        validate_plasma_config(config)

    config = _config()
    config["instruments"][0]["scaling"]["divisor"] = {
        94: 10_000, "A94": 20_000, 171: 30_000,
    }
    with pytest.raises(PlasmaConfigError, match="duplicate channel aliases"):
        validate_plasma_config(config)

    config = _config()
    config["instruments"][0]["scaling"]["divisor"] = [10_000]
    with pytest.raises(PlasmaConfigError, match="one divisor per channel"):
        validate_plasma_config(config)


def test_image_scaling_belongs_to_the_instrument_not_the_dataset():
    config = _config()
    config["instruments"][0]["scaling"]["divisor"] = "/data/aia/image_scaling.yaml"
    validate_plasma_config(config)

    for group in ("train_datasets", "valid_datasets"):
        config["data"][group][0]["scaling"] = 1.0
    with pytest.raises(PlasmaConfigError, match="scaling is unsupported"):
        validate_plasma_config(config)

    config = _config()
    config["instruments"][0]["scaling"]["strategy"] = "robust_percentile"
    with pytest.raises(PlasmaConfigError, match="unsupported fields"):
        validate_plasma_config(config)


def test_learnable_response_calibration_requires_one_global_reference():
    config = _config()
    response = config["instruments"][0]["temperature_response"]
    response.update(learnable=True, reference_channel="A171")
    with pytest.raises(PlasmaConfigError, match="exactly one"):
        validate_plasma_config(config)

    response["global_reference"] = True
    response["common_gain_limit_dex"] = 1.5
    response["common_gain_prior_sigma_dex"] = 0.5
    assert validate_plasma_config(config)["instruments"][0][
        "temperature_response"
    ]["global_reference"] is True

    response["gain_prior_sigma_dex"] = 0
    with pytest.raises(PlasmaConfigError, match="gain_prior_sigma_dex"):
        validate_plasma_config(config)


@pytest.mark.parametrize(
    ("relative_gauge", "match"),
    [
        ({"reference_channel": None}, "non-null"),
        (
            {"reference_channel": "A171", "gain_constraint": "zero_mean"},
            "mutually exclusive",
        ),
        ({"gain_constraint": "unknown"}, "gain_constraint: zero_mean"),
    ],
)
def test_relative_channel_gain_gauge_is_unambiguous(relative_gauge, match):
    config = _config()
    response = config["instruments"][0]["temperature_response"]
    response.update(learnable=True, global_reference=True, **relative_gauge)

    with pytest.raises(PlasmaConfigError, match=match):
        validate_plasma_config(config)


def test_global_reference_must_be_learnable_and_supervised():
    config = _config()
    config["instruments"][0]["temperature_response"]["global_reference"] = True
    with pytest.raises(PlasmaConfigError, match="requires learnable"):
        validate_plasma_config(config)

    config = _config()
    reference = config["instruments"][0]
    reference["temperature_response"].update(
        learnable=True,
        global_reference=True,
        reference_channel="A171",
    )
    supervised = copy.deepcopy(reference)
    supervised["key"] = "EUVI-A"
    supervised["temperature_response"]["global_reference"] = False
    config["instruments"].append(supervised)
    for group in ("train_datasets", "valid_datasets"):
        config["data"][group][0]["instrument_key"] = "EUVI-A"

    with pytest.raises(PlasmaConfigError, match="present in data.train_datasets"):
        validate_plasma_config(config)


def test_cross_instrument_calibration_disallows_extra_fixed_anchor():
    config = _config()
    learnable = copy.deepcopy(config["instruments"][0])
    learnable["key"] = "EUVI-A"
    learnable["temperature_response"].update(
        learnable=True,
        global_reference=True,
        reference_channel="A171",
    )
    config["instruments"].append(learnable)
    for group in ("train_datasets", "valid_datasets"):
        dataset = copy.deepcopy(config["data"][group][0])
        dataset["key"] = f"euvi_{group}"
        dataset["instrument_key"] = "EUVI-A"
        config["data"][group].append(dataset)

    with pytest.raises(PlasmaConfigError, match="every supervised instrument"):
        validate_plasma_config(config)


def test_schema_v2_requires_strict_prepared_fits_ingestion():
    config = _config()
    dataset = config["data"]["train_datasets"][0]
    dataset["manifest_path"] = dataset.pop("data_path")
    with pytest.raises(PlasmaConfigError, match="manifest_path is unsupported"):
        validate_plasma_config(config)

    config = _config()
    config["data"]["valid_datasets"][0]["strict_metadata"] = False
    with pytest.raises(PlasmaConfigError, match="must be true"):
        validate_plasma_config(config)


def test_shared_sources_require_a_matching_holdout_split():
    config = _config()
    config["data"]["valid_datasets"][0]["match_tolerance_minutes"] = 5
    with pytest.raises(PlasmaConfigError, match="same prepared observations"):
        validate_plasma_config(config)

    config = _config()
    for group in ("train_datasets", "valid_datasets"):
        config["data"][group][0]["holdout"] = None
    with pytest.raises(PlasmaConfigError, match="requires a holdout"):
        validate_plasma_config(config)


def test_validation_only_and_training_only_datasets_are_supported():
    config = _config()
    # Validation-only view of a configured instrument with its own files.
    config["data"]["valid_datasets"][0]["data_path"] = "/data/other/*.prepared.fits"
    # High-cadence event sequence without a validation counterpart.
    event = copy.deepcopy(config["data"]["train_datasets"][0])
    event.update(key="aia_event", data_path="/data/event/*.prepared.fits", holdout=None)
    config["data"]["train_datasets"].append(event)
    validate_plasma_config(config)

    # Validation-only instrument: fixed response next to learnable supervised ones.
    config["instruments"][0]["temperature_response"].update(
        learnable=True, global_reference=True, reference_channel="A171",
    )
    validation_only = copy.deepcopy(config["instruments"][0])
    validation_only["key"] = "EUVI-A"
    validation_only["temperature_response"].update(
        learnable=False, global_reference=False,
    )
    validation_only["temperature_response"].pop("reference_channel")
    config["instruments"].append(validation_only)
    with pytest.raises(PlasmaConfigError, match="unused instruments"):
        validate_plasma_config(config)
    config["data"]["valid_datasets"][0]["instrument_key"] = "EUVI-A"
    validate_plasma_config(config)


def test_normal_time_shuffle_is_expressed_in_seconds():
    config = _config()
    config["shuffle"] = {
        "type": "normal_time",
        "start_seconds": 43_200.0,
        "end_seconds": 60.0,
        "iterations": 100_000,
    }
    validate_plasma_config(config)

    config["shuffle"] = {"type": "normal_time", "start": 50, "end": 0.01, "iterations": 100_000}
    with pytest.raises(PlasmaConfigError, match="physical"):
        validate_plasma_config(config)


def test_canonical_2012_multi_instrument_config_uses_v2_contract():
    repository = Path(__file__).resolve().parents[1]
    config = load_yaml_config(repository / "config/plasma/all_2012_08.yaml")

    validated = validate_plasma_config(config)

    assert validated["schema_version"] == 2
    assert validated["data"]["holdout"] == {"strategy": "center", "count": 1}
    assert all(
        "artifact" in instrument["temperature_response"]
        for instrument in validated["instruments"]
    )
    assert all(
        "scaling" not in dataset for dataset in validated["data"]["train_datasets"]
    )
    assert all(
        isinstance(instrument["scaling"]["divisor"], str)
        for instrument in validated["instruments"]
    )


def test_physical_density_regularization_requires_an_explicit_scale():
    config = _config()
    config["module"].pop("regularization_density_scale_cm3")

    with pytest.raises(PlasmaConfigError, match="regularization_density_scale_cm3"):
        validate_plasma_config(config)


def test_pointwise_model_controls_and_packaged_artifacts_are_validated():
    config = _config()
    config["instruments"][0]["temperature_response"].update({
        "artifact": "builtin:aia",
        "temperature_cutoff": {"T_cut_K": 4.0e5, "delta_T_K": 5.0e4},
    })
    config["model"]["initial_log_T"] = 6.1
    config["absorption"] = {
        "type": "photoionization",
        "artifact": "builtin:h_he_photoionization",
        "hydrogen_density_convention": "cie_electrons_per_hydrogen",
        "minimum_electron_per_hydrogen": 0.1,
    }
    config.setdefault("lambda", {})["absorption"] = 0.0
    validate_plasma_config(config)

    for mutate, match in (
        (lambda c: c["absorption"].update(hydrogen_density_convention="ionized"),
         "hydrogen_density_convention"),
        (lambda c: c["absorption"].update(minimum_electron_per_hydrogen=2.0),
         "minimum_electron_per_hydrogen"),
        (lambda c: c["model"].update(initial_log_T=99.0), "initial_log_T"),
        (lambda c: c["instruments"][0]["temperature_response"].update(
            temperature_cutoff={"T_cut_K": 4.0e5}), "temperature_cutoff"),
        (lambda c: c["instruments"][0]["temperature_response"].update(
            temperature_cutoff={"T_cut_K": -1.0, "delta_T_K": 5.0e4}), "T_cut_K"),
    ):
        invalid = copy.deepcopy(config)
        mutate(invalid)
        with pytest.raises(PlasmaConfigError, match=match):
            validate_plasma_config(invalid)


def test_shipped_absorption_configuration_is_valid_and_portable():
    for name, absorption_artifact in (
        ("all_2012_08", None),
        ("all_2012_08_absorption", "builtin:h_he_photoionization"),
    ):
        config = load_yaml_config(
            str(Path(__file__).parents[1] / "config" / "plasma" / f"{name}.yaml")
        )
        validated = validate_plasma_config(config)

        artifacts = [
            instrument["temperature_response"]["artifact"]
            for instrument in validated["instruments"]
        ]
        assert artifacts == ["builtin:aia", "builtin:euvi_a", "builtin:euvi_b"]
        assert validated["absorption"].get("artifact") == absorption_artifact


def test_light_travel_time_is_a_validated_module_option():
    config = _config()
    config["module"]["light_travel_time"] = True
    validate_plasma_config(config)

    config["module"]["light_travel_time"] = "yes"
    with pytest.raises(PlasmaConfigError, match="module.light_travel_time must be a boolean"):
        validate_plasma_config(config)

    # Observations carry detector timestamps; the synthetic PSI frames are stamped
    # with their snapshot times (render.sh uses --no-light-travel-time).
    for name, expected in (("all_2012_08", True), ("psi_observers", False)):
        shipped = load_yaml_config(
            str(Path(__file__).parents[1] / "config" / "plasma" / f"{name}.yaml")
        )
        assert validate_plasma_config(shipped)["module"]["light_travel_time"] is expected


def test_cool_absorber_configuration_is_validated():
    config = _config()
    config["model"].update({"cool_absorber": True, "cool_density_offset_log10_cm3": 7.0})
    config["module"]["cool_column_scale_cm2"] = 1.0e19
    config["absorption"] = {
        "type": "photoionization", "artifact": "builtin:h_he_photoionization",
        "cool_ion_fractions": {"H_I": 0.7, "He_I": 0.7, "He_II": 0.3},
    }
    config.setdefault("lambda", {}).update({"absorption": 0.0, "cool_absorber": 1.0e-4})
    validate_plasma_config(config)

    for mutate, match in (
        (lambda c: c.update(absorption={"type": None}), "cool_absorber requires"),
        (lambda c: c["absorption"]["cool_ion_fractions"].update(He_II=0.6), "must not exceed one"),
        (lambda c: c["absorption"].update(cool_ion_fractions={"H_I": 0.5}), "exactly H_I"),
        (lambda c: c["model"].update(cool_absorber="yes"), "model.cool_absorber"),
        (lambda c: c["module"].update(cool_column_scale_cm2=0.0), "cool_column_scale_cm2"),
        (lambda c: c["lambda"].update(cool_absorber=-1.0), "lambda.cool_absorber"),
    ):
        invalid = copy.deepcopy(config)
        mutate(invalid)
        with pytest.raises(PlasmaConfigError, match=match):
            validate_plasma_config(invalid)


def test_density_profile_configuration_is_validated():
    config = _config()
    config["model"]["density_profile"] = {"type": "hydrostatic", "scale_height_rsun": 0.1}
    validate_plasma_config(config)
    config["model"]["density_profile"] = {"type": "power_law", "exponent": 2.0}
    validate_plasma_config(config)

    for profile, match in (
        ({"type": "exponential"}, "power_law' or 'hydrostatic"),
        ({"type": "hydrostatic"}, "scale_height_rsun"),
        ({"type": "hydrostatic", "scale_height_rsun": -0.1}, "scale_height_rsun"),
        ({"type": "hydrostatic", "scale_height_rsun": 0.1, "exponent": 2}, "unsupported fields"),
    ):
        invalid = copy.deepcopy(config)
        invalid["model"]["density_profile"] = profile
        with pytest.raises(PlasmaConfigError, match=match):
            validate_plasma_config(invalid)

    # Both shipped first-model configurations use the hydrostatic baseline and
    # run without the radial density penalty.
    for name in ("all_2012_08", "psi_observers"):
        shipped = validate_plasma_config(load_yaml_config(
            str(Path(__file__).parents[1] / "config" / "plasma" / f"{name}.yaml")
        ))
        assert shipped["model"]["density_profile"] == {
            "type": "hydrostatic", "scale_height_rsun": 0.1,
        }
        assert shipped["lambda"]["regularization"] == 0.0
