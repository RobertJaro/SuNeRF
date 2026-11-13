from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from sunpy.coordinates import frames
from sunpy.map import make_fitswcs_header

from sunerf.train.pole_callback import (
    NorthPoleViewCallback,
    latitude_circle_radius_pixels,
    north_pole_view_geometry,
)

TIME = datetime(2012, 8, 15, 12)
DISTANCE = (1 * u.AU).to(u.solRad)


def _validation_wcs(shape=(64, 64), scale=50.0):
    observer = SkyCoord(
        lon=0 * u.deg, lat=6 * u.deg, radius=DISTANCE, obstime=TIME,
        frame=frames.HeliographicStonyhurst,
    )
    reference = SkyCoord(
        0 * u.arcsec, 0 * u.arcsec,
        frame=frames.Helioprojective(observer=observer, obstime=TIME),
    )
    header = make_fitswcs_header(
        shape, reference, scale=[scale, scale] * u.arcsec / u.pix, projection_code='ARC'
    )
    return WCS(header)


def _geometry(**kwargs):
    observer = {'radius': DISTANCE, 'time': TIME}
    return north_pole_view_geometry(_validation_wcs(**kwargs), (64, 64), observer, 1.0)


def _closest_pixel(rays, point):
    origin, direction = rays[..., 0, :], rays[..., 1, :]
    offset = point - origin
    along = np.sum(offset * direction, axis=-1, keepdims=True)
    miss = np.linalg.norm(offset - along * direction, axis=-1)
    return np.unravel_index(np.argmin(miss), miss.shape)


def test_geometry_looks_down_on_the_north_pole():
    geometry = _geometry()
    rays = geometry['rays']
    assert geometry['image_shape'] == (64, 64)
    assert np.isclose(geometry['scale_arcsec'][0], 50.0)
    np.testing.assert_allclose(rays[..., 0, :].reshape(-1, 3) - [0.0, 0.0, DISTANCE.value], 0.0, atol=1e-3)
    np.testing.assert_allclose(rays[32, 32, 1], [0.0, 0.0, -1.0], atol=1e-3)

    # Carrington longitude 0 points right and longitude 90 up, at the radius
    # of the drawn latitude circle.
    latitude = np.radians(60.0)
    radius = latitude_circle_radius_pixels(60.0, geometry['distance_rsun'], 50.0)
    center = 31.5
    row, column = _closest_pixel(rays, np.array([np.cos(latitude), 0, np.sin(latitude)]))
    assert abs(row - center) <= 1 and abs(column - (center + radius)) <= 1
    row, column = _closest_pixel(rays, np.array([0, np.cos(latitude), np.sin(latitude)]))
    assert abs(column - center) <= 1 and abs(row - (center + radius)) <= 1


def test_geometry_keeps_field_of_view_at_coarser_resolution():
    observer = {'radius': DISTANCE, 'time': TIME}
    geometry = north_pole_view_geometry(
        _validation_wcs(shape=(512, 512), scale=6.0), (512, 512), observer, 1.0
    )
    assert geometry['image_shape'] == (256, 256)
    assert np.isclose(geometry['scale_arcsec'][0], 12.0)


class _FakeRendering(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, batch, shuffle=None, diagnostics=False):
        (key, item), = batch.items()
        self.calls.append((shuffle, diagnostics, self.training))
        rays = item['rays']
        n = rays.shape[0]
        # Brightness grows with the image x direction of the ray.
        x = rays[:, 1, 0:1]
        valid = torch.ones(n, dtype=torch.bool)
        valid[:1] = False
        return {'model_out': {key: {
            'image': torch.cat([1.0 + x, 2.0 + x], dim=-1),
            'mean_T': torch.full((n, 1), 6.1),
            'emission_measure_cm5': torch.full((n, 1), 1e26),
            'column_electron_density_cm2': torch.full((n,), 1e18),
            'ray_valid': valid,
        }}}


def test_callback_renders_and_logs_the_pole_view(monkeypatch):
    logged = {}
    monkeypatch.setattr('sunerf.train.pole_callback.wandb.Image', lambda figure: figure)
    monkeypatch.setattr('sunerf.train.pole_callback.wandb.log', logged.update)
    rendering = _FakeRendering().train()
    module = SimpleNamespace(
        rendering=rendering,
        image_scaling={'AIA': lambda image: image * 10.0},
        device=torch.device('cpu'),
    )
    callback = NorthPoleViewCallback(
        'AIA', 'AIA', ({'id': '94'}, {'id': '171'}), ('171',), _geometry(), 0.5,
        every_n_validations=2, batch_size=1000,
    )
    assert callback.channel_indices == (1,)

    callback.on_validation_end(None, module)
    assert not logged

    outputs = callback.render(module)
    assert outputs['image'].shape == (64, 64, 2)
    assert outputs['column_electron_density_cm2'].shape == (64, 64, 1)
    assert not outputs['ray_valid'][0, 0] and outputs['ray_valid'][0, 1]
    assert outputs['image'][32, 63, 1] > outputs['image'][32, 0, 1]
    np.testing.assert_allclose(outputs['image'][32, 32, 0], 10.0, atol=0.01)
    assert all(call == (False, True, False) for call in rendering.calls)
    assert len(rendering.calls) == 5
    assert rendering.training

    callback.on_validation_end(None, module)
    assert list(logged) == ['euv_tomography.north_pole.AIA']
    assert callback.state_dict() == {'validation_count': 2}


def test_callback_renders_through_the_plasma_module(tmp_path, monkeypatch):
    from sunerf.model.plasma import PlasmaSuNeRFModule
    from sunerf.response import ResponseArtifact

    monkeypatch.chdir(tmp_path)
    response = ResponseArtifact(
        channels=('A94', 'A171'),
        log_temperature=np.array([5.0, 6.0, 7.0]),
        response=np.ones((2, 3)),
        response_unit='cm5 DN / (pix s)',
        emission_measure_convention='ne2',
        provenance={'provider': 'test-fixture'},
    )
    response.save(tmp_path / 'response.npz')
    module = PlasmaSuNeRFModule(
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        instruments_config=[{
            'key': 'AIA',
            'type': 'plasma',
            'temperature_response': {
                'artifact': 'response.npz',
                'channels': ['A94', 'A171'],
                'response_id': response.response_id,
                'learnable': True,
                'global_reference': True,
                'common_gain_limit_dex': 1.0,
            },
            'scaling': {'type': 'log', 'divisor': [25.0, 50.0]},
        }],
        model_config={
            'type': 'siren', 'dim': 8, 'n_layers': 2,
            'density_offset_log10_cm3': 8.0,
            'temperature_grid': {'log10_K_min': 5.0, 'log10_K_max': 7.0, 'step_dex': 0.5},
        },
        sampling_config={
            'type': 'spherical', 'min_distance': 1.0, 'max_distance': 1.7,
            'n_samples': 8, 'perturb': True,
        },
        hierarchical_sampling_config={'type': 'hierarchical', 'n_samples': 8, 'perturb': True},
        absorption_config={'type': None},
        lambda_config={'regularization': 0.0},
        regularization_density_scale_cm3=1e8,
        validation_dataset_mapping={},
    )
    callback = NorthPoleViewCallback(
        'AIA', 'AIA', module.instrument_metadata['AIA']['channels'], ('94', '171'),
        _geometry(), 0.0, batch_size=1024,
    )
    module.train()
    outputs = callback.render(module)
    assert module.training

    on_disk = outputs['ray_valid']
    # Rays through the 1.7 R_sun shell are valid, the image corners miss it.
    assert on_disk[32, 32] and not on_disk[0, 0]
    for key in ('image', 'mean_T', 'emission_measure_cm5', 'column_electron_density_cm2'):
        assert np.all(np.isfinite(outputs[key][on_disk])), key
    assert np.any(outputs['emission_measure_cm5'][on_disk] > 0)
    # The deterministic validation path: two renders agree exactly.
    np.testing.assert_array_equal(callback.render(module)['image'], outputs['image'])
    figure = callback.plot(outputs)
    assert len(figure.axes) >= 5


def test_runner_builds_the_view_from_the_held_out_frame():
    from sunerf.run_plasma import _north_pole_view_callback

    data_module = SimpleNamespace(
        validation_dataset_mapping={0: 'AIA'},
        Rs_per_ds=1.0,
        seconds_per_dt=86400.0,
        ref_date=datetime(2012, 8, 1),
        config={'AIA': {
            'instrument_key': 'AIA',
            'wcs': _validation_wcs(),
            'image_shape': (64, 64),
            'observers': [{'radius': DISTANCE, 'time': TIME}],
            'times': [TIME],
            'cmaps': ['sdoaia94', 'sdoaia171'],
        }},
    )
    sunerf = SimpleNamespace(instrument_metadata={
        'AIA': {'channels': [{'id': '94'}, {'id': '171'}]}
    })
    callback = _north_pole_view_callback(
        {'enabled': True, 'dataset': 'AIA', 'channels': ['171']},
        {'every_n_validations': 3, 'figure_dpi': 80},
        sunerf,
        data_module,
        batch_size=256,
    )
    assert callback.normalized_time == 14.5
    assert callback.channel_indices == (1,)
    assert callback.channel_metadata[0]['cmap'] == 'sdoaia171'
    assert callback.every_n_validations == 3 and callback.batch_size == 256
    assert callback.rays.shape == (64 * 64, 2, 3)
