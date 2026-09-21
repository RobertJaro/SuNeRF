"""Shared, pipeline-independent training runtime utilities.

Both Thomson and EUV reconstructions use the same cache provenance, atomic
publication, distributed cache coordination, and Lightning device selection.
Keeping these helpers outside either runner prevents one physics pipeline from
depending on the other's command module.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import time
import uuid
import warnings
from dataclasses import dataclass, field

import torch


# Bump whenever the cached arrays change meaning for an unchanged config (e.g.
# a new default in a dataset class). Existing data modules are otherwise
# reused without any provenance check.
# 6: observational datasets mask non-positive brightness (clip_negative).
DATA_CACHE_FORMAT_VERSION = 6


def _cache_json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _cache_relevant_config(value):
    """Drop runtime-only options which do not change cached array contents."""
    if isinstance(value, dict):
        return {
            key: _cache_relevant_config(item)
            for key, item in value.items()
            if key not in {"num_workers", "preprocess_workers"}
        }
    if isinstance(value, list):
        return [_cache_relevant_config(item) for item in value]
    return value


def _configured_source_files(value, key=""):
    """Collect input files referenced by path/file entries in a data config."""
    if isinstance(value, dict):
        files = []
        for child_key, child_value in value.items():
            files.extend(_configured_source_files(child_value, str(child_key)))
        return files
    if isinstance(value, (list, tuple)):
        files = []
        for child_value in value:
            files.extend(_configured_source_files(child_value, key))
        return files
    if not isinstance(value, str):
        return []

    expanded_value = os.path.expanduser(value)
    is_path_key = any(token in key.lower() for token in ("path", "file", "manifest", "artifact"))
    if not is_path_key and not glob.has_magic(expanded_value) and not os.path.isfile(expanded_value):
        return []
    matches = glob.glob(expanded_value)
    if not matches and os.path.isfile(expanded_value):
        matches = [expanded_value]
    return [os.path.abspath(path) for path in matches if os.path.isfile(path)]


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_cache_fingerprint(data_config, *, format_version=DATA_CACHE_FORMAT_VERSION):
    """Fingerprint cache semantics and exact content of every transitive input."""
    source_files = set(_configured_source_files(data_config))
    source_records = []
    for path in sorted(source_files):
        stat = os.stat(path)
        source_records.append(
            {
                "path": path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(path),
            }
        )
    payload = {
        "format_version": int(format_version),
        "config": _cache_relevant_config(data_config),
        "sources": source_records,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_cache_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest(), source_records


def atomic_torch_save(value, destination):
    """Publish a torch artifact atomically on the destination filesystem."""
    temporary_path = f"{destination}.tmp-{uuid.uuid4().hex}"
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


def atomic_json_save(value, destination):
    """Publish a durable canonical JSON sidecar atomically."""
    temporary_path = f"{destination}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            os.remove(temporary_path)
        except FileNotFoundError:
            pass


# Historical private names remain available while runners migrate.
_atomic_torch_save = atomic_torch_save
_atomic_json_save = atomic_json_save


def _is_cache_generation(path, cache_root):
    if not path:
        return False
    real_path = os.path.realpath(path)
    real_root = os.path.realpath(cache_root)
    return os.path.dirname(real_path) == real_root and os.path.basename(real_path).startswith(
        "generation-"
    )


def data_module_cache_files(data_module):
    files = set()
    datasets = (
        *getattr(data_module, "training_datasets", {}).values(),
        *getattr(data_module, "validation_datasets", {}).values(),
    )
    for dataset in datasets:
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        files.update(
            os.path.abspath(path)
            for path in getattr(dataset, "batches_file_paths", {}).values()
        )
    return sorted(files)


_data_module_cache_files = data_module_cache_files


def data_cache_is_usable(data_module, fingerprint=None, reload_token=None, format_version=None):
    """Accept an existing cache when every mmap file is present.

    ``fingerprint=None`` skips the provenance comparison, so an existing data
    module is trusted without hashing its source files. ``format_version`` is
    a free attribute comparison that rejects caches written by older code.
    """
    if data_module is None:
        return False
    if format_version is not None and getattr(data_module, "cache_format_version", None) != format_version:
        return False
    if fingerprint is not None and getattr(data_module, "cache_fingerprint", None) != fingerprint:
        return False
    if reload_token is not None and getattr(data_module, "cache_reload_token", None) != reload_token:
        return False
    cache_files = getattr(data_module, "cache_files", None)
    if cache_files is None:
        cache_files = data_module_cache_files(data_module)
    return all(os.path.isfile(path) for path in cache_files)


_data_cache_is_usable = data_cache_is_usable


def remove_cache_generation(path, cache_root):
    """Delete only a UUID generation directory created by this cache manager."""
    if _is_cache_generation(path, cache_root):
        shutil.rmtree(path, ignore_errors=True)


_remove_cache_generation = remove_cache_generation


def wait_for_data_module(path, fingerprint=None, reload_token=None, format_version=None):
    """Wait for rank zero's atomic cache publication and validate its manifest."""
    timeout = float(os.environ.get("SUNERF_CACHE_WAIT_SECONDS", 6 * 60 * 60))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            data_module = torch.load(path, weights_only=False)
            if data_cache_is_usable(
                data_module, fingerprint, reload_token=reload_token, format_version=format_version
            ):
                return data_module
            last_error = RuntimeError(
                "cache provenance does not match or one of its generated files is missing"
            )
        except (FileNotFoundError, EOFError, OSError, RuntimeError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for data cache {path}: {last_error}")


_wait_for_data_module = wait_for_data_module


def cache_reload_token(enabled):
    """Return one reload publication token shared by all ranks in this launch."""
    if not enabled:
        return None
    existing = os.environ.get("SUNERF_CACHE_RELOAD_TOKEN")
    if existing:
        return existing

    shared_launch_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if not shared_launch_id and ("RANK" in os.environ or "LOCAL_RANK" in os.environ):
        shared_launch_id = ":".join(
            [
                os.environ.get("SLURM_JOB_ID", os.environ.get("PBS_JOBID", "external")),
                os.environ.get("MASTER_ADDR", "localhost"),
                os.environ.get("MASTER_PORT", "unknown"),
                os.environ.get("WORLD_SIZE", "unknown"),
            ]
        )
    token = shared_launch_id or uuid.uuid4().hex
    os.environ["SUNERF_CACHE_RELOAD_TOKEN"] = token
    return token


def trainer_device_config(cuda_device_count):
    """Return a valid Lightning accelerator/device pair for GPU or CPU hosts."""
    cuda_device_count = int(cuda_device_count)
    if cuda_device_count > 0:
        return "gpu", cuda_device_count
    return "cpu", 1


def start_wandb_logger(logging_config, work_directory, hparams=None):
    """Create the W&B logger and start its run on rank zero.

    Data preparation logs overview figures through ``wandb.log``, so the run
    must exist before any data module is built. ``WandbLogger`` only calls
    ``wandb.init`` lazily, and only on rank zero; logging the hyperparameters
    here forces that initialization at a well-defined point for every runner.
    """
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    logger = WandbLogger(**logging_config, save_dir=work_directory)

    @rank_zero_only
    def _start_run():
        logger.experiment  # noqa: B018 - property access initializes the run
        if hparams is not None:
            logger.log_hyperparams(hparams)

    _start_run()
    return logger


def prepare_data_module(work_directory, data_config, builder, reload=False):
    """Build the cache generation on rank zero and load it on every rank.

    ``builder(generation_directory)`` constructs the pipeline-specific data
    module. Non-zero ranks never build; they wait for rank zero's atomic
    publication so that all DDP processes share identical mmap files.
    """
    from lightning.pytorch.utilities.rank_zero import rank_zero_only

    data_cache = DataModuleCache(work_directory, data_config, reload=reload)

    @rank_zero_only
    def _publish_data_module():
        warnings.filterwarnings("ignore")  # ignore warnings from sunpy
        data_cache.build(builder)

    _publish_data_module()
    data_module = data_cache.wait()
    # Worker count is runtime state and deliberately excluded from the expensive
    # array-cache fingerprint; it can change without rewriting array files.
    if "num_workers" in data_config:
        data_module.num_workers = data_config["num_workers"]
    return data_module


def build_trainer(*, logger, callbacks, max_epochs, val_check_interval=None,
                  check_val_every_n_epoch=1, gradient_clip_val=0.5, **kwargs):
    """Create the Lightning trainer shared by the Thomson and EUV runners.

    All visible GPUs are used; more than one GPU selects DDP. Both pipelines
    skip parameters for absent datasets or disabled losses, which requires
    ``find_unused_parameters``.
    """
    from lightning.pytorch import Trainer
    from lightning.pytorch.strategies import DDPStrategy

    torch.set_float32_matmul_precision("high")
    n_gpus = torch.cuda.device_count()
    accelerator, devices = trainer_device_config(n_gpus)
    return Trainer(
        max_epochs=max_epochs,
        logger=logger,
        devices=devices,
        accelerator=accelerator,
        strategy=DDPStrategy(find_unused_parameters=True) if n_gpus > 1 else "auto",
        num_sanity_val_steps=0,  # validate all points to check the first image
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        **kwargs,
    )


@dataclass
class DataModuleCache:
    """Manage one atomically published data-module generation.

    ``build`` must be called only on rank zero (see ``prepare_data_module``).
    All ranks then call ``wait`` to load the same trusted local pickle after
    publication. An existing data
    module is reused as-is; the source fingerprint is only computed when a new
    generation is built (first run or ``reload``).
    """

    work_directory: str
    data_config: dict
    reload: bool = False
    format_version: int = DATA_CACHE_FORMAT_VERSION
    filename: str = "data_module.pkl"
    reload_token: str | None = field(init=False)
    _fingerprint: str | None = field(init=False, default=None, repr=False)
    _source_records: list | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.work_directory = os.path.abspath(os.fspath(self.work_directory))
        self.cache_root = os.path.join(self.work_directory, ".sunerf_cache")
        self.module_path = os.path.join(self.work_directory, self.filename)
        self.manifest_path = os.path.join(self.work_directory, "data_cache_manifest.json")
        self.reload_token = cache_reload_token(self.reload)

    def _compute_fingerprint(self):
        if self._fingerprint is None:
            self._fingerprint, self._source_records = build_data_cache_fingerprint(
                self.data_config,
                format_version=self.format_version,
            )

    @property
    def fingerprint(self):
        self._compute_fingerprint()
        return self._fingerprint

    @property
    def source_records(self):
        self._compute_fingerprint()
        return self._source_records

    def _load_existing(self):
        try:
            return torch.load(self.module_path, weights_only=False)
        except FileNotFoundError:
            return None
        except (EOFError, OSError, RuntimeError) as error:
            warnings.warn(f"Ignoring unreadable data cache: {error}")
            return None

    def build(self, builder):
        """Build and publish a cache using ``builder(generation_directory)``."""
        os.makedirs(self.work_directory, exist_ok=True)
        old_data_module = self._load_existing()
        if not self.reload and data_cache_is_usable(old_data_module, format_version=self.format_version):
            return old_data_module
        if old_data_module is not None and not self.reload:
            print(
                "Rebuilding data module: cache format version "
                f"{getattr(old_data_module, 'cache_format_version', None)} != {self.format_version} "
                "or cached files are missing."
            )

        os.makedirs(self.cache_root, exist_ok=True)
        generation_directory = os.path.join(
            self.cache_root,
            f"generation-{uuid.uuid4().hex}",
        )
        os.makedirs(generation_directory)
        data_module = None
        try:
            data_module = builder(generation_directory)
            data_module.cache_fingerprint = self.fingerprint
            data_module.cache_format_version = self.format_version
            data_module.cache_generation_directory = generation_directory
            data_module.cache_files = data_module_cache_files(data_module)
            data_module.cache_reload_token = self.reload_token
            data_module.cache_source_records = self.source_records
            atomic_torch_save(data_module, self.module_path)
            try:
                atomic_json_save(
                    {
                        "format_version": self.format_version,
                        "fingerprint": self.fingerprint,
                        "generation_directory": generation_directory,
                        "cache_files": data_module.cache_files,
                        "reload_token": self.reload_token,
                        "sources": self.source_records,
                    },
                    self.manifest_path,
                )
            except OSError as error:
                warnings.warn(f"Could not write data-cache manifest: {error}")
        except Exception:
            if data_module is not None and hasattr(data_module, "clear"):
                data_module.clear()
            remove_cache_generation(generation_directory, self.cache_root)
            raise

        if old_data_module is not None:
            remove_cache_generation(
                getattr(old_data_module, "cache_generation_directory", None),
                self.cache_root,
            )
        return data_module

    def wait(self):
        """Load rank zero's cache publication on every rank."""
        return wait_for_data_module(
            self.module_path,
            reload_token=self.reload_token,
            format_version=self.format_version,
        )
