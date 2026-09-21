"""Validated configuration contracts for supported SuNeRF pipelines.

The historical runners accepted arbitrary YAML dictionaries and discovered
missing or inconsistent fields only after expensive data preparation.  This
module keeps validation dependency-free while making the plasma pipeline's
scientific units and channel identity explicit.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence


PLASMA_CONFIG_SCHEMA_VERSION = 2
SUPPORTED_EUV_DATASET_TYPES = frozenset({"AIA", "EUI", "EUVI"})
HYDROGEN_DENSITY_CONVENTIONS = frozenset({
    "fully_ionized_proxy", "cie_electrons_per_hydrogen",
})
EUV_CALLBACK_PRODUCTS = frozenset({
    "channel_comparison",
    "plasma_diagnostics",
    "thermal_distribution",
    "response_and_gains",
    "ray_sampling",
})

_EUV_CALLBACK_DEFAULTS = {
    "enabled": True,
    "every_n_validations": 1,
    "figure_dpi": 150,
    "datasets": "all",
    "products": {
        "channel_comparison": {
            "enabled": True,
            "rows": ["observation", "prediction", "residual"],
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
        "ray_sampling": {
            "enabled": False,
            "pixel_fraction": [0.25, 0.25],
        },
    },
}


class PlasmaConfigError(ValueError):
    """Raised when a plasma configuration violates the supported contract."""


def _require_mapping(value, path):
    if not isinstance(value, Mapping):
        raise PlasmaConfigError(f"{path} must be a mapping")
    return value


def _require_sequence(value, path):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlasmaConfigError(f"{path} must be a sequence")
    return value


def _positive_number(value, path):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise PlasmaConfigError(f"{path} must be a finite positive number") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise PlasmaConfigError(f"{path} must be a finite positive number")
    return numeric


def _nonnegative_number(value, path):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise PlasmaConfigError(f"{path} must be a finite non-negative number") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise PlasmaConfigError(f"{path} must be a finite non-negative number")
    return numeric


def _positive_integer(value, path, *, minimum=1):
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise PlasmaConfigError(f"{path} must be an integer >= {minimum}")
    return value


def canonical_channel_id(value):
    """Normalize common instrument-prefixed wavelength channel identifiers."""
    text = str(value).strip().upper()
    match = re.fullmatch(r"[A-Z_-]*([0-9]+(?:\.[0-9]+)?)", text)
    if match:
        return match.group(1).rstrip("0").rstrip(".") if "." in match.group(1) else match.group(1)
    return text


def canonical_channel_mapping(value):
    """Canonicalize channel-keyed data while rejecting ambiguous aliases."""
    canonical = {}
    source_keys = {}
    for key, item in value.items():
        channel = canonical_channel_id(key)
        if channel in canonical:
            raise ValueError(
                f"contains duplicate channel aliases {source_keys[channel]!r} and "
                f"{key!r} for canonical channel {channel!r}"
            )
        canonical[channel] = item
        source_keys[channel] = key
    return canonical


def _validate_unique_keys(entries, path):
    keys = []
    for index, entry in enumerate(entries):
        entry = _require_mapping(entry, f"{path}[{index}]")
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            raise PlasmaConfigError(f"{path}[{index}].key must be a non-empty string")
        keys.append(key)
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise PlasmaConfigError(f"{path} contains duplicate keys: {duplicates}")
    return keys


def _validate_response(instrument, path):
    response = _require_mapping(instrument.get("temperature_response"), f"{path}.temperature_response")
    channels = _require_sequence(response.get("channels"), f"{path}.temperature_response.channels")
    if not channels or any(not str(channel).strip() for channel in channels):
        raise PlasmaConfigError(f"{path}.temperature_response.channels must not be empty")
    canonical = [canonical_channel_id(channel) for channel in channels]
    if len(set(canonical)) != len(canonical):
        raise PlasmaConfigError(f"{path}.temperature_response.channels contains duplicates")

    artifact = response.get("artifact")
    if "file" in response:
        raise PlasmaConfigError(
            f"{path}.temperature_response.file is unsupported; provide a verified 'artifact'"
        )
    if not isinstance(artifact, str) or not artifact.strip():
        raise PlasmaConfigError(
            f"{path}.temperature_response.artifact must be a non-empty path string"
        )
    if "normalization" in response and float(response["normalization"]) != 1.0:
        raise PlasmaConfigError(
            f"{path}.temperature_response cannot renormalize a physical artifact; "
            "rebuild the response with correct units instead"
        )
    if "scaling" in response:
        raise PlasmaConfigError(
            f"{path}.temperature_response.scaling is unsupported; use the instrument's "
            "scaling.divisor for fixed image divisors and bounded learnable gains "
            "for calibration"
        )
    forbidden_length_overrides = {
        key for key in ('Rs_per_ds', 'model_length_unit_cm') if key in response
    }
    if forbidden_length_overrides:
        raise PlasmaConfigError(
            f"{path}.temperature_response cannot override model length with "
            f"{sorted(forbidden_length_overrides)!r}; schema-v2 fixes one model "
            "distance unit to one solar radius"
        )

    cutoff = response.get("temperature_cutoff")
    if cutoff is not None:
        cutoff = _require_mapping(cutoff, f"{path}.temperature_response.temperature_cutoff")
        if set(cutoff) != {"T_cut_K", "delta_T_K"}:
            raise PlasmaConfigError(
                f"{path}.temperature_response.temperature_cutoff must define exactly "
                "T_cut_K and delta_T_K"
            )
        for field in ("T_cut_K", "delta_T_K"):
            _positive_number(
                cutoff[field], f"{path}.temperature_response.temperature_cutoff.{field}"
            )

    learnable = response.get("learnable", False)
    if not isinstance(learnable, bool):
        raise PlasmaConfigError(f"{path}.temperature_response.learnable must be boolean")
    global_reference = response.get("global_reference", False)
    if not isinstance(global_reference, bool):
        raise PlasmaConfigError(
            f"{path}.temperature_response.global_reference must be boolean"
        )
    if global_reference and not learnable:
        raise PlasmaConfigError(
            f"{path}.temperature_response.global_reference requires learnable: true"
        )
    if learnable:
        if "reference_channel" in response and response["reference_channel"] is None:
            raise PlasmaConfigError(
                f"{path}.temperature_response.reference_channel must be non-null when supplied"
            )
        reference_channel = response.get("reference_channel")
        gain_constraint = response.get("gain_constraint")
        if reference_channel is not None and "gain_constraint" in response:
            raise PlasmaConfigError(
                f"{path}.temperature_response.reference_channel and gain_constraint "
                "are mutually exclusive relative-gain gauges"
            )
        if reference_channel is None and gain_constraint not in {None, "zero_mean"}:
            raise PlasmaConfigError(
                f"{path}.temperature_response learnable gains require reference_channel "
                "or gain_constraint: zero_mean"
            )
        if reference_channel is not None and canonical_channel_id(reference_channel) not in canonical:
            raise PlasmaConfigError(
                f"{path}.temperature_response.reference_channel is not in its ordered channels"
            )
        _positive_number(
            response.get("gain_limit_dex", 0.3),
            f"{path}.temperature_response.gain_limit_dex",
        )
        _positive_number(
            response.get(
                "gain_prior_sigma_dex",
                float(response.get("gain_limit_dex", 0.3)) / 2,
            ),
            f"{path}.temperature_response.gain_prior_sigma_dex",
        )
        _positive_number(
            response.get("common_gain_limit_dex", 1.0),
            f"{path}.temperature_response.common_gain_limit_dex",
        )
        _positive_number(
            response.get(
                "common_gain_prior_sigma_dex",
                float(response.get("common_gain_limit_dex", 1.0)) / 2,
            ),
            f"{path}.temperature_response.common_gain_prior_sigma_dex",
        )
    return canonical


IMAGE_SCALING_TYPES = {
    "asinh": frozenset({"type", "divisor", "a", "vmax"}),
    "linear": frozenset({"type", "divisor", "vmin", "vmax"}),
    "log": frozenset({"type", "divisor", "vmin", "vmax"}),
}


def _validate_image_divisor(value, channels, path):
    """Fixed per-channel divisor: number, ordered list, channel mapping, or table path."""
    if isinstance(value, str):
        # Path of a sunerf.image_scaling_table.v1 file written by
        # `python -m sunerf.data.euv.estimate_scaling`; resolved by the runner.
        if not value.strip():
            raise PlasmaConfigError(f"{path} must not be an empty table path")
    elif isinstance(value, Mapping):
        try:
            canonical_values = canonical_channel_mapping(value)
        except ValueError as error:
            raise PlasmaConfigError(f"{path} {error}") from error
        if set(canonical_values) != set(channels):
            raise PlasmaConfigError(
                f"{path} mapping keys must match ordered response channels {channels}"
            )
        for channel, divisor in canonical_values.items():
            _positive_number(divisor, f"{path}.{channel}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(channels):
            raise PlasmaConfigError(f"{path} must contain one divisor per channel")
        for index, divisor in enumerate(value):
            _positive_number(divisor, f"{path}[{index}]")
    else:
        _positive_number(value, path)


def _validate_instrument_scaling(instrument, channels, path):
    """The loss-space transform and its fixed divisor belong to the instrument.

    Every dataset mapped to the instrument shares them, whether it is trained on,
    validation-only, or one of several sequences with different cadence.
    """
    scaling = _require_mapping(instrument.get("scaling"), f"{path}.scaling")
    scaling_type = scaling.get("type", "asinh")
    if scaling_type not in IMAGE_SCALING_TYPES:
        raise PlasmaConfigError(
            f"{path}.scaling.type must be one of {sorted(IMAGE_SCALING_TYPES)}"
        )
    unknown = set(scaling).difference(IMAGE_SCALING_TYPES[scaling_type])
    if unknown:
        raise PlasmaConfigError(
            f"{path}.scaling contains unsupported fields: {sorted(unknown)}"
        )
    if "divisor" not in scaling:
        raise PlasmaConfigError(
            f"{path}.scaling.divisor must explicitly define the image divisor: a "
            "number, one value per channel, or the path of a table written by "
            "`python -m sunerf.data.euv.estimate_scaling`; use 1.0 for no rescaling"
        )
    _validate_image_divisor(scaling["divisor"], channels, f"{path}.scaling.divisor")


def _validate_dataset_group(entries, path, instruments):
    entries = _require_sequence(entries, path)
    _validate_unique_keys(entries, path)
    for index, dataset in enumerate(entries):
        dataset_path = f"{path}[{index}]"
        dataset_type = dataset.get("type")
        if dataset_type not in SUPPORTED_EUV_DATASET_TYPES:
            raise PlasmaConfigError(
                f"{dataset_path}.type must be one of {sorted(SUPPORTED_EUV_DATASET_TYPES)}"
            )
        instrument_key = dataset.get("instrument_key")
        if not isinstance(instrument_key, str) or not instrument_key:
            raise PlasmaConfigError(
                f"{dataset_path}.instrument_key must be an explicit non-empty string"
            )
        if instrument_key not in instruments:
            raise PlasmaConfigError(
                f"{dataset_path}.instrument_key={instrument_key!r} does not name a configured instrument"
            )
        channels = dataset.get("channels", dataset.get("wavelengths"))
        if channels is None:
            raise PlasmaConfigError(f"{dataset_path} must define ordered channels or wavelengths")
        canonical = [canonical_channel_id(channel) for channel in _require_sequence(channels, f"{dataset_path}.channels")]
        expected = instruments[instrument_key]
        if canonical != expected:
            raise PlasmaConfigError(
                f"{dataset_path} channel order {canonical} does not match "
                f"instrument {instrument_key!r} response order {expected}"
            )
        data_path = dataset.get("data_path")
        if "manifest_path" in dataset:
            raise PlasmaConfigError(
                f"{dataset_path}.manifest_path is unsupported; use a prepared FITS data_path"
            )
        if not isinstance(data_path, str) or not data_path:
            raise PlasmaConfigError(
                f"{dataset_path}.data_path must name prepared FITS files"
            )
        if "scaling" in dataset:
            raise PlasmaConfigError(
                f"{dataset_path}.scaling is unsupported; datasets stay in physical units "
                f"and share instruments[].scaling.divisor of {instrument_key!r}"
            )
        # Optional override of data.holdout, e.g. `holdout: null` for a training
        # sequence without a validation counterpart.
        if dataset.get("holdout") is not None:
            _validate_holdout(dataset["holdout"])
        if dataset.get("strict_metadata") is not True:
            raise PlasmaConfigError(
                f"{dataset_path}.strict_metadata must be true for schema-v2 plasma data"
            )
    return entries


def _freeze_config_value(value):
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_config_value(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _dataset_signature(entry):
    scientific_config = dict(entry)
    # Dataset keys route batches and may legitimately differ between the
    # train/validation loaders. Every field that selects or interprets the
    # observations must otherwise be identical for an index-based holdout.
    scientific_config.pop("key", None)
    return _freeze_config_value(scientific_config)


def _validate_holdout_separation(train_datasets, valid_datasets, data):
    """Validation must never see training observations.

    A validation dataset that reads the prepared files of a training dataset is
    separated from it by the index-based holdout, which requires an identical
    selection on both sides. Validation datasets with their own files (a view
    that is never trained on) and training datasets without a validation twin
    (e.g. a short high-cadence event sequence) need no counterpart.
    """
    train_by_path = {}
    for index, dataset in enumerate(train_datasets):
        train_by_path.setdefault(dataset["data_path"], []).append((index, dataset))
    for index, dataset in enumerate(valid_datasets):
        path = f"data.valid_datasets[{index}]"
        shared = train_by_path.get(dataset["data_path"], ())
        if not shared:
            continue
        if not any(
            _dataset_signature(dataset) == _dataset_signature(train_dataset)
            for _, train_dataset in shared
        ):
            raise PlasmaConfigError(
                f"{path} shares data_path with data.train_datasets"
                f"[{shared[0][0]}] and must select and interpret the same prepared "
                "observations; the holdout performs the split"
            )
        if dataset.get("holdout", data.get("holdout")) is None:
            raise PlasmaConfigError(
                f"{path} shares data_path with a training dataset and requires a "
                "holdout so validation observations are excluded from training"
            )


def _validate_shuffle(config):
    if config is None:
        return
    shuffle = _require_mapping(config, "shuffle")
    shuffle_type = shuffle.get("type")
    if shuffle_type == "normal_time":
        legacy = {"start", "end"}.intersection(shuffle)
        if legacy:
            raise PlasmaConfigError(
                "shuffle.normal_time uses physical start_seconds/end_seconds; "
                f"legacy normalized fields are not supported: {sorted(legacy)}"
            )
        start = _nonnegative_number(shuffle.get("start_seconds"), "shuffle.start_seconds")
        end = _nonnegative_number(shuffle.get("end_seconds"), "shuffle.end_seconds")
        if start < end:
            raise PlasmaConfigError("shuffle.start_seconds must be >= shuffle.end_seconds")
        _positive_integer(shuffle.get("iterations"), "shuffle.iterations")
    elif shuffle_type == "time":
        probability = float(shuffle.get("probability", 0.5))
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise PlasmaConfigError("shuffle.probability must lie in [0, 1]")
        _positive_integer(shuffle.get("iterations"), "shuffle.iterations")
    else:
        raise PlasmaConfigError(f"shuffle.type={shuffle_type!r} is unsupported")


def _validate_holdout(config):
    holdout = _require_mapping(config, "data.holdout")
    unknown = set(holdout).difference({"strategy", "count"})
    if unknown:
        raise PlasmaConfigError(f"data.holdout contains unsupported fields: {sorted(unknown)}")
    if holdout.get("strategy") != "center":
        raise PlasmaConfigError("data.holdout.strategy must currently be 'center'")
    _positive_integer(holdout.get("count"), "data.holdout.count")


def _validate_temperature_grid(model):
    grid = _require_mapping(model.get("temperature_grid"), "model.temperature_grid")
    log_min = float(grid.get("log10_K_min", math.nan))
    log_max = float(grid.get("log10_K_max", math.nan))
    step = _positive_number(grid.get("step_dex"), "model.temperature_grid.step_dex")
    if not math.isfinite(log_min) or not math.isfinite(log_max) or log_min >= log_max:
        raise PlasmaConfigError(
            "model.temperature_grid requires finite log10_K_min < log10_K_max"
        )
    n_steps = (log_max - log_min) / step
    if not math.isclose(n_steps, round(n_steps), rel_tol=0.0, abs_tol=1e-8):
        raise PlasmaConfigError(
            "model.temperature_grid range must be an integer multiple of step_dex"
        )
    initial_log_T = model.get("initial_log_T")
    if initial_log_T is not None and not (
        isinstance(initial_log_T, (int, float)) and log_min < float(initial_log_T) < log_max
    ):
        raise PlasmaConfigError(
            "model.initial_log_T must lie strictly inside model.temperature_grid"
        )
    if model.get("type", "siren") not in ("siren", "mlp", "generic"):
        raise PlasmaConfigError("model.type must be 'siren' or 'mlp'")
    if "density_profile" in model:
        profile = _require_mapping(model["density_profile"], "model.density_profile")
        profile_type = profile.get("type")
        fields = {"power_law": "exponent", "hydrostatic": "scale_height_rsun"}
        if profile_type not in fields:
            raise PlasmaConfigError(
                "model.density_profile.type must be 'power_law' or 'hydrostatic'"
            )
        unknown = set(profile) - {"type", fields[profile_type]}
        if unknown:
            raise PlasmaConfigError(
                f"model.density_profile contains unsupported fields: {sorted(unknown)}"
            )
        if profile_type == "hydrostatic" or fields[profile_type] in profile:
            _positive_number(
                profile.get(fields[profile_type]),
                f"model.density_profile.{fields[profile_type]}",
            )
    if "cool_absorber" in model:
        _boolean(model["cool_absorber"], "model.cool_absorber")
    if "cool_density_offset_log10_cm3" in model and not (
        isinstance(model["cool_density_offset_log10_cm3"], (int, float))
        and math.isfinite(float(model["cool_density_offset_log10_cm3"]))
    ):
        raise PlasmaConfigError("model.cool_density_offset_log10_cm3 must be a finite number")
    density_offset = model.get("density_offset_log10_cm3")
    if not isinstance(density_offset, (int, float)) or not math.isfinite(float(density_offset)):
        raise PlasmaConfigError(
            "model.density_offset_log10_cm3 must be an explicit finite number"
        )


def _validate_absorption(config):
    absorption = _require_mapping(config.get("absorption", {"type": None}), "absorption")
    absorption_type = absorption.get("type")
    if absorption_type is None:
        unknown = set(absorption).difference({"type"})
        if unknown:
            raise PlasmaConfigError(
                f"disabled absorption contains unsupported fields: {sorted(unknown)}"
            )
        return
    if absorption_type == "photoionization":
        unknown = set(absorption).difference({
            "type", "artifact", "hydrogen_density_convention",
            "minimum_electron_per_hydrogen", "cool_ion_fractions",
        })
        if "cool_ion_fractions" in absorption:
            fractions = _require_mapping(
                absorption["cool_ion_fractions"], "absorption.cool_ion_fractions"
            )
            if set(fractions) != {"H_I", "He_I", "He_II"}:
                raise PlasmaConfigError(
                    "absorption.cool_ion_fractions must define exactly H_I, He_I and He_II"
                )
            for species, value in fractions.items():
                if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    raise PlasmaConfigError(
                        f"absorption.cool_ion_fractions.{species} must lie in [0, 1]"
                    )
            if float(fractions["He_I"]) + float(fractions["He_II"]) > 1.0 + 1e-9:
                raise PlasmaConfigError(
                    "absorption.cool_ion_fractions He_I + He_II must not exceed one"
                )
        convention = absorption.get("hydrogen_density_convention", "fully_ionized_proxy")
        if convention not in HYDROGEN_DENSITY_CONVENTIONS:
            raise PlasmaConfigError(
                "absorption.hydrogen_density_convention must be one of "
                f"{sorted(HYDROGEN_DENSITY_CONVENTIONS)}"
            )
        if "minimum_electron_per_hydrogen" in absorption:
            minimum = _positive_number(
                absorption["minimum_electron_per_hydrogen"],
                "absorption.minimum_electron_per_hydrogen",
            )
            if minimum > 1:
                raise PlasmaConfigError(
                    "absorption.minimum_electron_per_hydrogen must not exceed one"
                )
        if unknown:
            raise PlasmaConfigError(
                "photoionization absorption contains unsupported fields: "
                f"{sorted(unknown)}"
            )
        artifact = absorption.get("artifact")
        if not isinstance(artifact, str) or not artifact.strip():
            raise PlasmaConfigError(
                "absorption.artifact must be a non-empty bundle path for photoionization"
            )
        lambda_absorption = _nonnegative_number(
            _require_mapping(config.get("lambda"), "lambda").get("absorption", 0.0),
            "lambda.absorption",
        )
        if lambda_absorption != 0:
            raise PlasmaConfigError(
                "lambda.absorption must be zero for deterministic photoionization opacity"
            )
        return
    if absorption_type not in {"learned", "constant"}:
        raise PlasmaConfigError(
            "absorption.type must be null, 'photoionization', 'learned', or 'constant'"
        )


def _validate_training_contract(config):
    lambdas = _require_mapping(config.get("lambda"), "lambda")
    lambda_values = {
        key: _nonnegative_number(lambdas.get(key, default), f"lambda.{key}")
        for key, default in (
            ("image", 1.0),
            ("regularization", 0.0),
            ("absorption", 0.0),
            ("calibration", 1.0e-4),
            ("cool_absorber", 1.0e-4),
        )
    }
    if lambda_values["image"] == 0:
        raise PlasmaConfigError("lambda.image must be strictly positive")

    module = _require_mapping(config.get("module", {}), "module")
    if lambda_values["regularization"] > 0:
        _positive_number(
            module.get("regularization_density_scale_cm3"),
            "module.regularization_density_scale_cm3",
        )

    if "cool_column_scale_cm2" in module:
        _positive_number(module["cool_column_scale_cm2"], "module.cool_column_scale_cm2")
    if "light_travel_time" in module:
        _boolean(module["light_travel_time"], "module.light_travel_time")

    lr = _require_mapping(module.get("lr_config"), "module.lr_config")
    _positive_number(lr.get("start"), "module.lr_config.start")
    _positive_number(lr.get("end"), "module.lr_config.end")
    _positive_integer(lr.get("iterations"), "module.lr_config.iterations")

    training = _require_mapping(config.get("training"), "training")
    _positive_integer(training.get("epochs"), "training.epochs")
    if "log_every_n_steps" in training:
        _positive_integer(training["log_every_n_steps"], "training.log_every_n_steps")
    _positive_integer(
        training.get("check_val_every_n_epoch", 1),
        "training.check_val_every_n_epoch",
    )


def _boolean(value, path):
    if not isinstance(value, bool):
        raise PlasmaConfigError(f"{path} must be a boolean")
    return value


def _percentage(value, path, *, lower=0.0, upper=100.0):
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise PlasmaConfigError(f"{path} must be a finite percentage") from error
    if not math.isfinite(numeric) or not lower <= numeric <= upper:
        raise PlasmaConfigError(f"{path} must lie in [{lower}, {upper}]")
    return numeric


def _unique_choice_sequence(value, path, choices):
    values = [str(item) for item in _require_sequence(value, path)]
    if not values:
        raise PlasmaConfigError(f"{path} must not be empty")
    if len(set(values)) != len(values):
        raise PlasmaConfigError(f"{path} must not contain duplicates")
    unknown = sorted(set(values).difference(choices))
    if unknown:
        raise PlasmaConfigError(f"{path} contains unsupported values: {unknown}")
    return values


def _resolve_callback_channels(requested, available, path):
    if requested == "all":
        return list(available)
    requested = _require_sequence(requested, path)
    resolved = []
    for requested_channel in requested:
        matches = [
            channel for channel in available
            if canonical_channel_id(channel) == canonical_channel_id(requested_channel)
        ]
        if len(matches) != 1:
            raise PlasmaConfigError(
                f"{path} channel {requested_channel!r} must match exactly one of {tuple(available)}"
            )
        if matches[0] in resolved:
            raise PlasmaConfigError(f"{path} selects channel {matches[0]!r} more than once")
        resolved.append(matches[0])
    if not resolved:
        raise PlasmaConfigError(f"{path} must select at least one channel")
    return resolved


def _validate_euv_callbacks(config, valid_datasets, channel_orders):
    callbacks = config.setdefault("callbacks", {})
    callbacks = _require_mapping(callbacks, "callbacks")
    unknown_callbacks = set(callbacks).difference({"euv_tomography"})
    if unknown_callbacks:
        raise PlasmaConfigError(
            f"callbacks contains unsupported entries: {sorted(unknown_callbacks)}"
        )

    supplied = callbacks.get("euv_tomography", {})
    supplied = _require_mapping(supplied, "callbacks.euv_tomography")
    unknown = set(supplied).difference(_EUV_CALLBACK_DEFAULTS)
    if unknown:
        raise PlasmaConfigError(
            "callbacks.euv_tomography contains unsupported fields: "
            f"{sorted(unknown)}"
        )

    normalized = copy.deepcopy(_EUV_CALLBACK_DEFAULTS)
    for key in ("enabled", "every_n_validations", "figure_dpi", "datasets"):
        if key in supplied:
            normalized[key] = copy.deepcopy(supplied[key])
    normalized["enabled"] = _boolean(
        normalized["enabled"], "callbacks.euv_tomography.enabled"
    )
    normalized["every_n_validations"] = _positive_integer(
        normalized["every_n_validations"],
        "callbacks.euv_tomography.every_n_validations",
    )
    normalized["figure_dpi"] = _positive_integer(
        normalized["figure_dpi"], "callbacks.euv_tomography.figure_dpi", minimum=72
    )

    dataset_instruments = {
        dataset["key"]: dataset["instrument_key"] for dataset in valid_datasets
    }
    requested_datasets = normalized["datasets"]
    if requested_datasets == "all":
        requested_datasets = {key: {"channels": "all"} for key in dataset_instruments}
    else:
        requested_datasets = _require_mapping(
            requested_datasets, "callbacks.euv_tomography.datasets"
        )
    unknown_datasets = set(requested_datasets).difference(dataset_instruments)
    if unknown_datasets:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.datasets contains unknown validation datasets: "
            f"{sorted(unknown_datasets)}"
        )
    normalized_datasets = {}
    for dataset_key, dataset_settings in requested_datasets.items():
        dataset_settings = _require_mapping(
            dataset_settings, f"callbacks.euv_tomography.datasets.{dataset_key}"
        )
        unsupported = set(dataset_settings).difference({"channels"})
        if unsupported:
            raise PlasmaConfigError(
                f"callbacks.euv_tomography.datasets.{dataset_key} contains unsupported "
                f"fields: {sorted(unsupported)}"
            )
        instrument_key = dataset_instruments[dataset_key]
        normalized_datasets[dataset_key] = {
            "channels": _resolve_callback_channels(
                dataset_settings.get("channels", "all"),
                channel_orders[instrument_key],
                f"callbacks.euv_tomography.datasets.{dataset_key}.channels",
            )
        }
    if normalized["enabled"] and not normalized_datasets:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.datasets must not be empty when enabled"
        )
    normalized["datasets"] = normalized_datasets

    supplied_products = supplied.get("products", {})
    supplied_products = _require_mapping(
        supplied_products, "callbacks.euv_tomography.products"
    )
    unknown_products = set(supplied_products).difference(EUV_CALLBACK_PRODUCTS)
    if unknown_products:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products contains unsupported products: "
            f"{sorted(unknown_products)}"
        )
    for product_name, defaults in normalized["products"].items():
        product = supplied_products.get(product_name, {})
        product = _require_mapping(
            product, f"callbacks.euv_tomography.products.{product_name}"
        )
        unknown_fields = set(product).difference(defaults)
        if unknown_fields:
            raise PlasmaConfigError(
                f"callbacks.euv_tomography.products.{product_name} contains unsupported "
                f"fields: {sorted(unknown_fields)}"
            )
        defaults.update(copy.deepcopy(product))
        defaults["enabled"] = _boolean(
            defaults["enabled"],
            f"callbacks.euv_tomography.products.{product_name}.enabled",
        )

    comparison = normalized["products"]["channel_comparison"]
    comparison["rows"] = _unique_choice_sequence(
        comparison["rows"],
        "callbacks.euv_tomography.products.channel_comparison.rows",
        {"observation", "prediction", "residual", "relative_residual"},
    )
    if comparison["stretch"] not in {"linear", "asinh", "log"}:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.channel_comparison.stretch must be "
            "'linear', 'asinh' or 'log'"
        )
    comparison["intensity_percentile"] = _percentage(
        comparison["intensity_percentile"],
        "callbacks.euv_tomography.products.channel_comparison.intensity_percentile",
        lower=50.0,
    )
    comparison["residual_percentile"] = _percentage(
        comparison["residual_percentile"],
        "callbacks.euv_tomography.products.channel_comparison.residual_percentile",
        lower=50.0,
    )

    diagnostics = normalized["products"]["plasma_diagnostics"]
    diagnostics["quantities"] = _unique_choice_sequence(
        diagnostics["quantities"],
        "callbacks.euv_tomography.products.plasma_diagnostics.quantities",
        {
            "mean_log_temperature",
            "column_electron_density",
            "emission_measure",
            "emission_height",
            "absorption_fraction",
        },
    )

    thermal = normalized["products"]["thermal_distribution"]
    if thermal["spatial_statistic"] not in {"median", "mean"}:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.thermal_distribution.spatial_statistic "
            "must be 'median' or 'mean'"
        )
    band = _require_sequence(
        thermal["percentile_band"],
        "callbacks.euv_tomography.products.thermal_distribution.percentile_band",
    )
    if len(band) != 2:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.thermal_distribution.percentile_band "
            "must contain two values"
        )
    band = [
        _percentage(
            value,
            "callbacks.euv_tomography.products.thermal_distribution.percentile_band",
        )
        for value in band
    ]
    if band[0] >= band[1]:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.thermal_distribution.percentile_band "
            "must be strictly increasing"
        )
    thermal["percentile_band"] = band

    ray = normalized["products"]["ray_sampling"]
    pixel = _require_sequence(
        ray["pixel_fraction"],
        "callbacks.euv_tomography.products.ray_sampling.pixel_fraction",
    )
    if len(pixel) != 2:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.ray_sampling.pixel_fraction must have two values"
        )
    try:
        pixel = [float(value) for value in pixel]
    except (TypeError, ValueError) as error:
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.ray_sampling.pixel_fraction "
            "must contain numeric values"
        ) from error
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in pixel):
        raise PlasmaConfigError(
            "callbacks.euv_tomography.products.ray_sampling.pixel_fraction values "
            "must lie in [0, 1]"
        )
    ray["pixel_fraction"] = pixel

    if normalized["enabled"] and not any(
        product["enabled"] for product in normalized["products"].values()
    ):
        raise PlasmaConfigError(
            "callbacks.euv_tomography must enable at least one plot product"
        )
    callbacks["euv_tomography"] = normalized


def _find_unresolved_placeholders(value, path="config"):
    unresolved = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            unresolved.extend(_find_unresolved_placeholders(item, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            unresolved.extend(_find_unresolved_placeholders(item, f"{path}[{index}]"))
    elif isinstance(value, str) and re.search(r"\{[^{}]+\}", value):
        unresolved.append(path)
    return unresolved


def validate_plasma_config(config):
    """Validate and return a defensive copy of a schema-v2 plasma config."""
    config = copy.deepcopy(_require_mapping(config, "config"))
    version = config.get("schema_version")
    if version != PLASMA_CONFIG_SCHEMA_VERSION:
        raise PlasmaConfigError(
            f"schema_version must be {PLASMA_CONFIG_SCHEMA_VERSION}; received {version!r}"
        )

    for key in ("base_path",):
        if not isinstance(config.get(key), str) or not config[key]:
            raise PlasmaConfigError(f"{key} must be a non-empty string")

    instruments = _require_sequence(config.get("instruments"), "instruments")
    instrument_keys = _validate_unique_keys(instruments, "instruments")
    if not instrument_keys:
        raise PlasmaConfigError("instruments must not be empty")
    channel_orders = {}
    for index, (key, instrument) in enumerate(zip(instrument_keys, instruments)):
        instrument_type = str(instrument.get("type", "")).lower()
        if instrument_type != "plasma":
            raise PlasmaConfigError(
                f"instruments[{index}].type={instrument_type!r} is unsupported; "
                "only the plasma instrument type is valid"
            )
        channel_orders[key] = _validate_response(instrument, f"instruments[{index}]")
        _validate_instrument_scaling(
            instrument, channel_orders[key], f"instruments[{index}]"
        )

    responses_by_instrument = {
        instrument["key"]: instrument["temperature_response"]
        for instrument in instruments
    }

    data = _require_mapping(config.get("data"), "data")
    rs_per_ds = _positive_number(data.get("Rs_per_ds"), "data.Rs_per_ds")
    if not math.isclose(rs_per_ds, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise PlasmaConfigError(
            "schema-v2 plasma reconstructions require data.Rs_per_ds == 1.0 so "
            "model-space radii, coronal priors, and evaluation coordinates all use "
            "solar-radius units"
        )
    _positive_number(data.get("seconds_per_dt"), "data.seconds_per_dt")
    _positive_integer(data.get("batch_size"), "data.batch_size")
    if "validation_batch_size" in data:
        _positive_integer(data["validation_batch_size"], "data.validation_batch_size")
    num_workers = data.get("num_workers", 0)
    if not isinstance(num_workers, int) or num_workers < 0:
        raise PlasmaConfigError("data.num_workers must be a non-negative integer")
    train_datasets = _validate_dataset_group(
        data.get("train_datasets"), "data.train_datasets", channel_orders
    )
    valid_datasets = _validate_dataset_group(
        data.get("valid_datasets"), "data.valid_datasets", channel_orders
    )
    _validate_holdout_separation(train_datasets, valid_datasets, data)
    supervised_instruments = {dataset["instrument_key"] for dataset in train_datasets}
    configured_references = {
        key
        for key, response in responses_by_instrument.items()
        if response.get("global_reference", False)
    }
    unused_references = configured_references - supervised_instruments
    if unused_references:
        raise PlasmaConfigError(
            "temperature_response.global_reference must name an instrument present "
            "in data.train_datasets; unsupervised references do not fix the "
            f"density/calibration gauge: {sorted(unused_references)}"
        )
    validated_instruments = {dataset["instrument_key"] for dataset in valid_datasets}
    unused_instruments = (
        set(responses_by_instrument) - supervised_instruments - validated_instruments
    )
    if unused_instruments:
        raise PlasmaConfigError(
            "every configured instrument must be represented in data.train_datasets "
            f"or data.valid_datasets; unused instruments: {sorted(unused_instruments)}"
        )
    # Validation-only instruments are never optimized: their gains stay nominal
    # and they do not anchor the calibration, so the gauge rules below concern
    # the supervised instruments only.
    configured_learnable = {
        key
        for key in supervised_instruments
        if responses_by_instrument[key].get("learnable", False)
    }
    configured_fixed = supervised_instruments - configured_learnable
    if configured_learnable and configured_fixed:
        raise PlasmaConfigError(
            "learnable cross-instrument calibration requires learnable: true for "
            "every supervised instrument; fixed responses would create additional "
            f"absolute anchors: {sorted(configured_fixed)}"
        )
    if configured_learnable and len(configured_references) != 1:
        raise PlasmaConfigError(
            "learnable response calibration requires exactly one supervised "
            "instruments[].temperature_response.global_reference=true; "
            f"found {sorted(configured_references)}"
        )
    if "holdout" not in data:
        raise PlasmaConfigError(
            "data.holdout is required so validation observations are excluded from training"
        )
    _validate_holdout(data["holdout"])

    sampling = _require_mapping(config.get("sampling", {}), "sampling")
    _positive_integer(sampling.get("n_samples", 64), "sampling.n_samples", minimum=2)
    min_distance = float(sampling.get("min_distance", 1.0))
    max_distance = _positive_number(sampling.get("max_distance"), "sampling.max_distance")
    if min_distance < 0 or min_distance >= max_distance:
        raise PlasmaConfigError("sampling requires 0 <= min_distance < max_distance")

    hierarchical = _require_mapping(
        config.get("hierarchical_sampling", {}), "hierarchical_sampling"
    )
    _positive_integer(
        hierarchical.get("n_samples", 128),
        "hierarchical_sampling.n_samples",
    )
    _validate_shuffle(config.get("shuffle"))
    _validate_temperature_grid(_require_mapping(config.get("model"), "model"))
    _validate_training_contract(config)
    _validate_absorption(config)
    if config["model"].get("cool_absorber", False) and (
        _require_mapping(config.get("absorption", {"type": None}), "absorption").get("type")
        != "photoionization"
    ):
        raise PlasmaConfigError(
            "model.cool_absorber requires absorption.type: photoionization"
        )
    _validate_euv_callbacks(config, valid_datasets, channel_orders)

    unresolved = _find_unresolved_placeholders(config)
    if unresolved:
        raise PlasmaConfigError(f"unresolved configuration placeholders at: {unresolved}")
    return config
