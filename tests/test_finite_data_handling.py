from datetime import datetime, timezone

import numpy as np

from sunerf.data.loader.insitu import INERTIAL_COORDS_KEY, InSituDataset
from sunerf.data.loader.thomson_instrument import valid_training_rows


def test_valid_rows_allow_one_observable_but_reject_infinite_geometry():
    tensors = {
        'image': np.array([
            [1.0, np.nan],
            [np.inf, 2.0],
            [3.0, 1.0],
        ], dtype=np.float32),
        'rays': np.ones((3, 2, 3), dtype=np.float32),
        'time': np.ones((3, 1), dtype=np.float32),
        'image_coords': np.ones((3, 2), dtype=np.float32),
        'hpc_coords': np.ones((3, 3), dtype=np.float32),
    }
    tensors['rays'][2, 0, 0] = np.inf

    np.testing.assert_array_equal(valid_training_rows(tensors), [True, True, False])


def test_valid_rows_require_scale_only_for_observed_channels():
    tensors = {
        'image': np.array([[1.0, np.nan], [1.0, 2.0]], dtype=np.float32),
        'rays': np.ones((2, 2, 3), dtype=np.float32),
        'time': np.ones((2, 1), dtype=np.float32),
        'image_coords': np.ones((2, 2), dtype=np.float32),
        'hpc_coords': np.ones((2, 3), dtype=np.float32),
        'scaling_mask': np.array([[1.0, np.nan], [1.0, np.inf]], dtype=np.float32),
    }

    np.testing.assert_array_equal(valid_training_rows(tensors), [True, False])


def test_valid_rows_reject_missing_brightness_even_with_finite_geometry():
    tensors = {
        'image': np.array([[1.0, np.nan], [np.nan, np.nan]], dtype=np.float32),
        'rays': np.ones((2, 2, 3), dtype=np.float32),
        'time': np.ones((2, 1), dtype=np.float32),
        'image_coords': np.ones((2, 2), dtype=np.float32),
        'hpc_coords': np.ones((2, 3), dtype=np.float32),
    }

    np.testing.assert_array_equal(valid_training_rows(tensors), [True, False])


def test_insitu_filter_preserves_rows_with_either_finite_target(tmp_path):
    data_path = tmp_path / 'insitu.npz'
    time_unix = np.array([1_600_000_000, 1_600_000_060, 1_600_000_120], dtype=np.float64)
    np.savez(
        data_path,
        time_unix=time_unix,
        density_cm3=np.array([5.0, np.nan, 7.0], dtype=np.float32),
        velocity_radial_kms=np.array([np.nan, 400.0, 450.0], dtype=np.float32),
        has_density=np.array([1.0, 0.0, 1.0], dtype=np.float32),
        has_velocity=np.array([0.0, 1.0, 1.0], dtype=np.float32),
        density_source=np.array([1.0, 0.0, 2.0], dtype=np.float32),
        **{
            INERTIAL_COORDS_KEY: np.array(
                [[10.0, 0.0, 0.0], [11.0, 0.1, 0.2], [12.0, -0.1, 0.4]],
                dtype=np.float32,
            )
        },
    )

    dataset = InSituDataset(
        data_path=data_path,
        Rs_per_ds=100.0,
        seconds_per_dt=86_400.0,
        ref_date=datetime.fromtimestamp(time_unix[0], timezone.utc),
        work_directory=tmp_path / 'cache',
        drho_cm3=1.0,
        batch_size=16,
        shuffle=False,
    )
    try:
        batch = dataset[0]
        np.testing.assert_array_equal(
            batch['has_density'][:, 0].numpy(), [1.0, 0.0, 1.0]
        )
        np.testing.assert_array_equal(
            batch['has_velocity'][:, 0].numpy(), [0.0, 1.0, 1.0]
        )
        assert batch['density'][1, 0].isnan().item()
        assert batch['velocity_radial'][0, 0].isnan().item()
    finally:
        dataset.clear()
