"""Learnable pointing alignment (roll + shift) of observational plasma instruments."""
import copy

import numpy as np
import pytest
import torch

from sunerf.configuration import PlasmaConfigError, validate_plasma_config
from sunerf.data.ray_sampling import get_rays
from sunerf.model.plasma import PlasmaSuNeRFModule, _validate_instrument_alignment_gauge
from sunerf.response import ResponseArtifact
from sunerf.train.coordinate_transformation import pose_spherical
from sunerf.train.correction import PointingAlignmentModule
from astropy import units as u


def _rays(lon_deg=30.0, lat_deg=10.0, distance=215.0, n=5):
    pose = pose_spherical(np.deg2rad(lon_deg), np.deg2rad(lat_deg), distance)
    tx = np.linspace(-900.0, 900.0, n) * u.arcsec
    ty = np.linspace(-600.0, 600.0, n) * u.arcsec
    tx, ty = np.meshgrid(tx, ty)
    rays_o, rays_d = get_rays(tx, ty, pose)
    rays = np.stack([rays_o, rays_d], axis=-2).reshape(-1, 2, 3)
    return torch.as_tensor(rays, dtype=torch.float64), tx.reshape(-1), ty.reshape(-1), pose


def test_alignment_starts_as_exact_identity():
    rays, *_ = _rays()
    module = PointingAlignmentModule().double()
    torch.testing.assert_close(module(rays), rays)
    assert module.roll_deg.item() == 0.0
    assert torch.count_nonzero(module.shift_arcsec) == 0


def test_shift_matches_a_helioprojective_pointing_offset():
    rays, tx, ty, pose = _rays()
    module = PointingAlignmentModule(shift_limit_arcsec=3600.0).double()
    shift = torch.tensor([20.0, -12.0], dtype=torch.float64)
    with torch.no_grad():
        module.raw_shift.copy_(torch.atanh(shift / 3600.0))
    aligned = module(rays)
    # Rays now point where a pixel with (Tx + dx, Ty + dy) would point.
    _, expected = get_rays(tx + shift[0].item() * u.arcsec, ty + shift[1].item() * u.arcsec, pose)
    torch.testing.assert_close(aligned[:, 0], rays[:, 0])
    torch.testing.assert_close(aligned[:, 1], torch.as_tensor(expected, dtype=torch.float64), atol=2e-6, rtol=0)


def test_roll_rotates_about_the_boresight():
    rays, tx, ty, pose = _rays()
    module = PointingAlignmentModule(roll_limit_deg=10.0).double()
    with torch.no_grad():
        module.raw_roll.fill_(torch.atanh(torch.tensor(0.5)).item())  # 5 degrees
    aligned = module(rays)[:, 1]
    forward = torch.as_tensor(-pose[:3, 3] / np.linalg.norm(pose[:3, 3]), dtype=torch.float64)
    # The angle to the boresight is preserved, rolled directions differ off-axis.
    torch.testing.assert_close(aligned @ forward, rays[:, 1] @ forward)
    right = torch.as_tensor(pose[:3, 0], dtype=torch.float64)
    up = torch.as_tensor(pose[:3, 1], dtype=torch.float64)
    angle = torch.atan2(aligned @ up, aligned @ right) - torch.atan2(rays[:, 1] @ up, rays[:, 1] @ right)
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))  # wrap to (-pi, pi]
    off_axis = (np.hypot(tx.to_value(u.arcsec), ty.to_value(u.arcsec)) > 1.0)
    torch.testing.assert_close(torch.rad2deg(angle)[off_axis],
                               torch.full((int(off_axis.sum()),), 5.0, dtype=torch.float64), atol=1e-3, rtol=0)


def test_alignment_is_bounded_and_trainable_switch_freezes_it():
    module = PointingAlignmentModule(roll_limit_deg=0.5, shift_limit_arcsec=10.0)
    with torch.no_grad():
        module.raw_roll.fill_(100.0)
        module.raw_shift.fill_(-100.0)
    assert module.roll_deg.item() == pytest.approx(0.5)
    assert module.shift_arcsec.tolist() == pytest.approx([-10.0, -10.0])
    frozen = PointingAlignmentModule(trainable=False)
    assert not any(p.requires_grad for p in frozen.parameters())
    with pytest.raises(ValueError, match='unsupported fields'):
        PointingAlignmentModule(angle_scale=1.0)


def _instruments(tmp_path):
    path = tmp_path / 'response.npz'
    ResponseArtifact(channels=('A94',), log_temperature=np.array([5., 6.]),
                     response=np.full((1, 2), 1e-26), response_unit='cm5 DN / (pix s)',
                     emission_measure_convention='ne2', provenance={'provider': 'fixture'}).save(path)
    response = {'artifact': str(path), 'channels': ['A94'], 'learnable': False}
    return [
        {'key': 'AIA', 'type': 'plasma', 'temperature_response': dict(response),
         'scaling': {'type': 'linear', 'divisor': 1.}},
        {'key': 'EUVI-A', 'type': 'plasma', 'temperature_response': dict(response),
         'scaling': {'type': 'linear', 'divisor': 1.}, 'alignment': True},
        {'key': 'EUI', 'type': 'plasma', 'temperature_response': dict(response),
         'scaling': {'type': 'linear', 'divisor': 1.}},
    ]


def _module(config, supervised=('AIA', 'EUVI-A')):
    return PlasmaSuNeRFModule(
        Rs_per_ds=1., seconds_per_dt=86400., instruments_config=config,
        supervised_instrument_keys=list(supervised), validation_dataset_mapping={},
        model_config={'type': 'siren', 'dim': 4, 'n_layers': 2, 'density_offset_log10_cm3': 8.,
                      'temperature_grid': {'log10_K_min': 5., 'log10_K_max': 6., 'step_dex': 1.}},
        sampling_config={'type': 'spherical', 'min_distance': 1., 'max_distance': 1.5, 'n_samples': 4, 'perturb': False},
        hierarchical_sampling_config={'type': 'hierarchical', 'n_samples': 4, 'perturb': False},
        absorption_config={'type': None}, lambda_config={'regularization': 0.}, regularization_density_scale_cm3=1e8,
    )


def test_alignment_is_allowed_on_every_supervised_view_but_not_held_out(tmp_path):
    config = _instruments(tmp_path)
    assert _validate_instrument_alignment_gauge(config, ['AIA', 'EUVI-A']) == ('EUVI-A',)
    config[0]['alignment'] = {'roll_limit_deg': 2.0}
    assert set(_module(config).alignment_modules) == {'AIA', 'EUVI-A'}
    config = _instruments(tmp_path)
    config[2]['alignment'] = True
    with pytest.raises(ValueError, match='held-out'):
        _module(config)


def test_module_trains_alignment_only_where_enabled_and_survives_reconstruction(tmp_path):
    module = _module(_instruments(tmp_path))
    assert set(module.alignment_modules) == {'EUVI-A'}
    assert module.instrument_metadata['EUVI-A']['alignment'] == {'roll_limit_deg': 1.0, 'shift_limit_arcsec': 30.0}
    assert 'alignment' not in module.instrument_metadata['AIA']

    rays, *_ = _rays(n=3)
    rays = rays.float()
    batch = {
        'EUVI-A': {'instrument': 'EUVI-A', 'rays': rays,
                   'time': torch.zeros(rays.shape[0], 1),
                   'image': torch.full((rays.shape[0], 1), 5.0),
                   'valid_mask': torch.ones(rays.shape[0], 1, dtype=torch.bool)},
    }
    module.train()
    loss = module.training_step(batch, 0)
    loss.backward()
    alignment = module.alignment_modules['EUVI-A']
    assert torch.isfinite(alignment.raw_shift.grad).all() and torch.isfinite(alignment.raw_roll.grad).all()
    assert torch.count_nonzero(alignment.raw_shift.grad) > 0
    metrics = module._instrument_scaling_metrics()
    assert {'alignment.EUVI-A.roll_deg', 'alignment.EUVI-A.shift_x_arcsec',
            'alignment.EUVI-A.shift_y_arcsec'} <= set(metrics)

    with torch.no_grad():
        alignment.raw_shift.fill_(0.3)
    restored = PlasmaSuNeRFModule(**module.construction_spec)
    restored.load_state_dict(module.state_dict())
    torch.testing.assert_close(restored.alignment_modules['EUVI-A'].shift_arcsec, alignment.shift_arcsec)
    torch.testing.assert_close(restored._apply_alignment(batch['EUVI-A'])['rays'],
                               module._apply_alignment(batch['EUVI-A'])['rays'])


def _config():
    from tests.test_plasma_configuration import _config as base
    config = base()
    second = copy.deepcopy(config['instruments'][0])
    second['key'] = 'EUVI-A'
    second['alignment'] = True
    config['instruments'].append(second)
    train = copy.deepcopy(config['data']['train_datasets'][0])
    train.update(key='euvi_train', instrument_key='EUVI-A', data_path='/data/euvi/*.prepared.fits')
    valid = copy.deepcopy(train)
    valid['key'] = 'euvi_valid'
    config['data']['train_datasets'].append(train)
    config['data']['valid_datasets'].append(valid)
    return config


def test_config_accepts_alignment_and_rejects_invalid_forms():
    validate_plasma_config(_config())
    config = _config()
    config['instruments'][1]['alignment'] = {'roll_limit_deg': 0.5, 'shift_limit_arcsec': 20.0, 'trainable': False}
    validate_plasma_config(config)

    config = _config()
    config['instruments'][0]['alignment'] = True
    validate_plasma_config(config)

    config = _config()
    config['data']['train_datasets'].pop()
    with pytest.raises(PlasmaConfigError, match='held-out'):
        validate_plasma_config(config)

    config = _config()
    config['instruments'][1]['alignment'] = {'angle_scale': 1e-3}
    with pytest.raises(PlasmaConfigError, match='unsupported fields'):
        validate_plasma_config(config)

    config = _config()
    config['instruments'][1]['alignment'] = {'shift_limit_arcsec': -1.0}
    with pytest.raises(PlasmaConfigError):
        validate_plasma_config(config)
