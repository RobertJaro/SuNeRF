import numpy as np
import torch

from sunerf.train.callback import BaseCallback, LongitudeSlicesCallback, RadialSlicesCallback


def test_validation_plot_key_omits_internal_dataset_name_by_default():
    callback = BaseCallback("valid_radial")

    assert callback.validation_plot_key("radial_slices", "density") == "radial_slices.density"


def test_validation_plot_key_keeps_explicit_name_for_multiple_instances():
    callback = BaseCallback("valid_radial", name="outer_corona")

    assert callback.validation_plot_key("radial_slices", "velocity") == (
        "radial_slices.velocity.outer_corona"
    )


def test_radial_slices_logs_density_and_velocity(monkeypatch):
    cube_shape = (2, 2, 3, 1)
    callback = RadialSlicesCallback(
        ds_key="valid_radial",
        cube_shape=cube_shape,
        radii=(3, 6),
        drho_cm3=10.0,
        Rs_per_ds=100.0,
        seconds_per_dt=86400.0,
    )

    radii = np.array([3.0, 6.0], dtype=np.float32)
    latitudes = np.array([-0.5, 0.5], dtype=np.float32)
    longitudes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    radius, latitude, longitude, _ = np.meshgrid(
        radii, latitudes, longitudes, np.array([0.0], dtype=np.float32), indexing="ij"
    )
    spherical_coords = np.stack((radius, latitude, longitude), axis=-1)
    n_points = int(np.prod(cube_shape))
    outputs = {
        "rho_pred": torch.linspace(1.0, 2.0, n_points).unsqueeze(-1),
        "v_pred": torch.linspace(1.0, 3.0, n_points * 3).reshape(n_points, 3),
        "spherical_coords": torch.from_numpy(spherical_coords.reshape(-1, 3)),
    }
    module = type("Module", (), {"validation_outputs": {"valid_radial": outputs}})()

    logged_keys = []
    monkeypatch.setattr("sunerf.train.callback.wandb.Image", lambda figure: figure)
    monkeypatch.setattr(
        "sunerf.train.callback.wandb.log",
        lambda values: logged_keys.extend(values),
    )

    callback.on_validation_end(None, module)

    assert logged_keys == ["radial_slices.density", "radial_slices.velocity"]


def test_longitude_slices_use_the_same_semantic_plot_names(monkeypatch):
    cube_shape = (2, 2, 2, 1)
    callback = LongitudeSlicesCallback(
        ds_key="valid_lon_slices",
        cube_shape=cube_shape,
        drho_cm3=10.0,
        Rs_per_ds=100.0,
        seconds_per_dt=86400.0,
        longitude_deg=(0, 90),
    )

    radii = np.array([3.0, 6.0], dtype=np.float32)
    latitudes = np.array([0.0, np.pi], dtype=np.float32)
    longitudes = np.array([0.0, np.pi / 2], dtype=np.float32)
    radius, latitude, longitude, _ = np.meshgrid(
        radii, latitudes, longitudes, np.array([0.0], dtype=np.float32), indexing="ij"
    )
    spherical_coords = np.stack((radius, latitude, longitude), axis=-1)
    n_points = int(np.prod(cube_shape))
    outputs = {
        "rho_pred": torch.linspace(1.0, 2.0, n_points).unsqueeze(-1),
        "v_pred": torch.linspace(1.0, 3.0, n_points * 3).reshape(n_points, 3),
        "spherical_coords": torch.from_numpy(spherical_coords.reshape(-1, 3)),
    }
    module = type("Module", (), {"validation_outputs": {"valid_lon_slices": outputs}})()

    logged_keys = []
    monkeypatch.setattr("sunerf.train.callback.wandb.Image", lambda figure: figure)
    monkeypatch.setattr(
        "sunerf.train.callback.wandb.log",
        lambda values: logged_keys.extend(values),
    )

    callback.on_validation_end(None, module)

    assert logged_keys == ["longitude_slices.density", "longitude_slices.velocity"]
