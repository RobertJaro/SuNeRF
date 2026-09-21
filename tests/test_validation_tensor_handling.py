import torch
from torch import nn

from sunerf.model.sunerf import BaseSuNeRFModule
from sunerf.model.thomson import ThomsonSuNeRFModule
from sunerf.data.loader.thomson_instrument import FixedViewpointSeriesDataset
from sunerf.train.callback import CorrectionImageCallback, ThomsonImageCallback


class _Rendering(nn.Module):
    def forward(self, batch):
        return batch

    def on_train_batch_end(self, *args, **kwargs):
        pass


def _base_module():
    return BaseSuNeRFModule(
        Rs_per_ds=1.0,
        seconds_per_dt=1.0,
        rendering=_Rendering(),
        validation_dataset_mapping={0: "image"},
    )


def test_callback_requirements_union_and_drop_unconsumed_tensors():
    module = _base_module()
    ThomsonImageCallback("image", (2, 2)).setup(None, module, "fit")
    CorrectionImageCallback("image", (2, 2)).setup(None, module, "fit")

    outputs = {
        "target_image": torch.ones(2, 2),
        "model_image": torch.ones(2, 2),
        "target_ratio": torch.ones(2, 1),
        "model_ratio": torch.ones(2, 1),
        "correction.tB_add": torch.ones(2, 1),
        "distance": torch.ones(2, 192),
        "density": torch.ones(2, 1),
    }
    batch = {"dataset_idx": torch.tensor([0]), "render_mode": torch.tensor(0)}
    module.on_validation_batch_end(outputs, batch, batch_idx=0, dataloader_idx=0)

    retained = module.validation_batches[0][0]
    assert set(retained) == {
        "target_image",
        "model_image",
        "target_ratio",
        "model_ratio",
        "correction.tB_add",
        "dataset_idx",
    }


def test_explicit_empty_callback_filter_retains_no_validation_tensors():
    module = _base_module()
    module.enable_validation_output_filter()

    module.on_validation_batch_end(
        {"large_diagnostic": torch.ones(2, 192)},
        {"dataset_idx": torch.tensor([0]), "render_mode": torch.tensor(0)},
        batch_idx=0,
        dataloader_idx=0,
    )

    assert module.validation_batches == {}


def test_validation_batch_merge_restores_order_and_drops_ddp_padding():
    batches = [
        {"dataset_idx": torch.tensor([2]), "value": torch.tensor([[20.0], [21.0]])},
        {"dataset_idx": torch.tensor([0]), "value": torch.tensor([[0.0]])},
        {"dataset_idx": torch.tensor([1]), "value": torch.tensor([[10.0], [11.0]])},
        # DistributedSampler can pad with a duplicate dataset item.
        {"dataset_idx": torch.tensor([0]), "value": torch.tensor([[999.0]])},
    ]

    merged = BaseSuNeRFModule._merge_validation_batches(batches)
    assert merged["value"].flatten().tolist() == [0.0, 10.0, 11.0, 20.0, 21.0]


def test_validation_end_releases_callback_tensors():
    module = _base_module()
    module.validation_outputs = {"image": {"value": torch.ones(4)}}
    module.validation_batches = {0: [{"value": torch.ones(4)}]}
    module.on_validation_end()
    assert module.validation_outputs == {}
    assert module.validation_batches == {}


def test_thomson_instrument_validation_does_not_return_los_diagnostics():
    class _FakeModule:
        scaling_modules = {"instrument": nn.Identity()}
        ratio_range = (0.0, 3.0)
        ratio_epsilon = 1.0e-6
        _polarization_ratio = ThomsonSuNeRFModule._polarization_ratio

        @staticmethod
        def _apply_alignment(batch):
            return batch

        @staticmethod
        def _apply_image_modules(batch, model_image):
            return model_image, {}

        @staticmethod
        def _normalize_with_scaling_mask(image, batch):
            return image

        @staticmethod
        def rendering(batch):
            n_rays = batch["image"]["image"].shape[0]
            return {
                "model_out": {
                    "image": {
                        "image": torch.ones(n_rays, 2),
                        "density": torch.ones(n_rays, 1),
                        "distance": torch.ones(n_rays, 192),
                        "distance_from_sun": torch.ones(n_rays, 192),
                        "distance_from_obs": torch.ones(n_rays, 192),
                    }
                }
            }

    batch = {
        "instrument": "instrument",
        "image": torch.ones(3, 2),
    }
    output = ThomsonSuNeRFModule._val_instrument(_FakeModule(), batch, "image")

    assert "density" in output
    assert "distance" not in output
    assert "distance_from_sun" not in output
    assert "distance_from_obs" not in output


def test_fixed_viewpoint_cache_preserves_pixel_order_and_full_hpc_coordinates(tmp_path):
    dataset = FixedViewpointSeriesDataset(
        instrument_key='PUNCH',
        n_times=2,
        time_range=[0.0, 1.0],
        resolution=(2, 3),
        Rs_per_ds=100.0,
        work_directory=tmp_path,
        batch_size=12,
    )

    batch = dataset[0]
    assert batch['hpc_coords'].shape == (12, 3)
    torch.testing.assert_close(batch['time'][:6], torch.zeros(6, 1))
    torch.testing.assert_close(batch['time'][6:], torch.ones(6, 1))
    dataset.clear()
