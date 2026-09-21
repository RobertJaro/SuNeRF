from inspect import signature
from pathlib import Path

import pytest
import yaml

from sunerf.model.thomson import ThomsonSuNeRFModule
from sunerf.train.util import load_yaml_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CME_CONFIG_DIR = PROJECT_ROOT / "config" / "cme"
SYNTHETIC_CME_DATASET_TYPES = {"hao", "psi_cme"}


def _cme_configs():
    return {
        path: yaml.safe_load(path.read_text())
        for path in sorted(CME_CONFIG_DIR.rglob("*.yaml"))
    }


def _dataset_types(config):
    data_config = config.get("data", {})
    datasets = [
        dataset
        for group in ("train_datasets", "valid_datasets")
        for dataset in data_config.get(group, [])
    ]
    return {
        str(dataset.get("type", "")).lower()
        for dataset in datasets
        if isinstance(dataset, dict)
    }


def test_load_yaml_config_replaces_declared_placeholder(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("base_path: '{root}/run'\nvalue: 3\n")
    config = load_yaml_config(path, ["--root", "/tmp/output"])
    assert config == {"base_path": "/tmp/output/run", "value": 3}


@pytest.mark.parametrize(
    "overrides",
    [
        ["--root"],
        ["root", "/tmp"],
        ["--missing", "/tmp"],
    ],
)
def test_load_yaml_config_rejects_malformed_or_unknown_overrides(tmp_path, overrides):
    path = tmp_path / "config.yaml"
    path.write_text("base_path: '{root}'\n")
    with pytest.raises(ValueError):
        load_yaml_config(path, overrides)


def test_load_yaml_config_requires_mapping_root(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- item\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml_config(path)


def test_synthetic_cme_configs_explicitly_disable_light_travel_time():
    configs = _cme_configs()
    synthetic_configs = {
        path: config
        for path, config in configs.items()
        if _dataset_types(config) & SYNTHETIC_CME_DATASET_TYPES
    }

    assert synthetic_configs, "No HAO or PSI synthetic CME configurations were found."
    missing_opt_out = [
        str(path.relative_to(PROJECT_ROOT))
        for path, config in synthetic_configs.items()
        if config.get("module", {}).get("light_travel_time") is not False
    ]
    assert not missing_opt_out, (
        "Instantaneous synthetic snapshots must disable retarded-time sampling: "
        + ", ".join(missing_opt_out)
    )


def test_observational_cme_configs_keep_light_travel_time_default():
    default = signature(ThomsonSuNeRFModule.__init__).parameters[
        "light_travel_time"
    ].default
    assert default is True

    unexpected_overrides = [
        str(path.relative_to(PROJECT_ROOT))
        for path, config in _cme_configs().items()
        if not (_dataset_types(config) & SYNTHETIC_CME_DATASET_TYPES)
        and "light_travel_time" in config.get("module", {})
    ]
    assert not unexpected_overrides, (
        "Observational CME configurations should retain detector-time retardation: "
        + ", ".join(unexpected_overrides)
    )
