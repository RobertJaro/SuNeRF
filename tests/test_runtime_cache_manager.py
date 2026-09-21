from pathlib import Path

from sunerf.train.runtime import DataModuleCache


class _Dataset:
    def __init__(self, cache_file):
        self.batches_file_paths = {"image": str(cache_file)}


class _DataModule:
    def __init__(self, cache_file):
        self.training_datasets = {"train": _Dataset(cache_file)}
        self.validation_datasets = {}
        self.cleared = False

    def clear(self):
        self.cleared = True


def test_data_module_cache_builds_reuses_and_reloads(tmp_path, monkeypatch):
    monkeypatch.delenv("SUNERF_CACHE_RELOAD_TOKEN", raising=False)
    source = tmp_path / "source.fits"
    source.write_bytes(b"source")
    config = {"data_path": str(source)}
    builds = []

    def builder(generation):
        builds.append(generation)
        cache_file = Path(generation) / "image.npy"
        cache_file.write_bytes(b"cache")
        return _DataModule(cache_file)

    manager = DataModuleCache(tmp_path, config)
    first = manager.build(builder)
    assert manager.wait().cache_fingerprint == first.cache_fingerprint
    assert len(builds) == 1

    reused = DataModuleCache(tmp_path, config).build(builder)
    assert reused.cache_generation_directory == first.cache_generation_directory
    assert len(builds) == 1

    # An existing data module is trusted as-is: changed sources neither
    # trigger a rebuild nor a fingerprint computation.
    source.write_bytes(b"changed source")
    trusting_manager = DataModuleCache(tmp_path, config)
    trusted = trusting_manager.build(builder)
    assert trusted.cache_generation_directory == first.cache_generation_directory
    assert trusting_manager.wait().cache_fingerprint == first.cache_fingerprint
    assert trusting_manager._fingerprint is None
    assert len(builds) == 1

    # A cache written by older code (format version) is rebuilt without reload.
    outdated = DataModuleCache(tmp_path, config, format_version=trusting_manager.format_version + 1)
    rebuilt = outdated.build(builder)
    assert rebuilt.cache_generation_directory != first.cache_generation_directory
    assert outdated.wait().cache_format_version == outdated.format_version
    assert len(builds) == 2
    first = rebuilt
    builds.pop()

    reloaded_manager = DataModuleCache(tmp_path, config, reload=True)
    reloaded = reloaded_manager.build(builder)
    assert reloaded.cache_generation_directory != first.cache_generation_directory
    assert len(builds) == 2
    assert not Path(first.cache_generation_directory).exists()


def _builder(builds):
    def builder(generation):
        builds.append(generation)
        cache_file = Path(generation) / "image.npy"
        cache_file.write_bytes(b"cache")
        return _DataModule(cache_file)

    return builder


def test_prepare_data_module_builds_on_rank_zero_and_applies_runtime_workers(tmp_path, monkeypatch):
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    from sunerf.train.runtime import prepare_data_module

    monkeypatch.delenv("SUNERF_CACHE_RELOAD_TOKEN", raising=False)
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
    config = {"num_workers": 3}
    builds = []
    data_module = prepare_data_module(tmp_path, config, _builder(builds))
    assert len(builds) == 1
    assert data_module.num_workers == 3

    # A non-zero rank never builds; it loads rank zero's published generation.
    monkeypatch.setattr(rank_zero_only, "rank", 1, raising=False)
    other = prepare_data_module(tmp_path, config, _builder(builds))
    assert len(builds) == 1
    assert other.cache_generation_directory == data_module.cache_generation_directory


def test_start_wandb_logger_initializes_run_before_data_loading(tmp_path, monkeypatch):
    import wandb
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    from sunerf.train.runtime import start_wandb_logger

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(rank_zero_only, "rank", 0, raising=False)
    start_wandb_logger({"project": "sunerf-test"}, str(tmp_path), hparams={"a": 1})
    try:
        assert wandb.run is not None
        wandb.log({"Overview.test": 1})  # must not raise "call wandb.init() first"
    finally:
        wandb.finish()
