from datetime import datetime

import numpy as np
import pytest
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from sunerf.evaluation.loader import SuNeRFLoader, _carrington_pose_from_hci_observer
from sunerf.train.coordinate_transformation import pose_spherical


class _GlobalOutputRendering:
    def __call__(self, batch, diagnostics=False):
        del diagnostics
        instrument = next(iter(batch))
        n_rays = len(batch[instrument]['rays'])
        return {
            'model_out': {
                instrument: {
                    'image': torch.ones((n_rays, 2)),
                    'calibration_regularization': torch.tensor(0.25),
                    'instrument_gain_delta_dex': torch.tensor([0.1, -0.1]),
                    'common_gain_delta_dex': torch.tensor(0.2),
                }
            }
        }


def test_pose_spherical_returns_numpy_camera_matrix():
    pose = pose_spherical(0.1, -0.2, 215.0)

    assert isinstance(pose, np.ndarray)
    assert pose.shape == (4, 4)


def test_hci_evaluation_observer_is_transformed_to_training_carrington_frame():
    obstime = datetime(2025, 1, 1)
    observer = SkyCoord(
        lon=0 * u.deg,
        lat=0 * u.deg,
        distance=1 * u.AU,
        obstime=obstime,
        frame=frames.HeliocentricInertial,
    )
    carrington = observer.transform_to(
        frames.HeliographicCarrington(observer='self', obstime=obstime)
    )

    pose = _carrington_pose_from_hci_observer(observer, Rs_per_ds=1.0)
    expected = pose_spherical(
        carrington.lon.to_value(u.rad),
        carrington.lat.to_value(u.rad),
        carrington.radius.to_value(u.solRad),
    )
    incorrect_hci_pose = pose_spherical(
        observer.lon.to_value(u.rad),
        observer.lat.to_value(u.rad),
        observer.distance.to_value(u.solRad),
    )

    np.testing.assert_allclose(pose, expected)
    assert not np.allclose(pose, incorrect_hci_pose)


def test_load_pose_separates_ray_major_and_global_outputs():
    loader = SuNeRFLoader.__new__(SuNeRFLoader)
    loader.device = torch.device('cpu')
    loader.rendering = _GlobalOutputRendering()
    loader.instrument_keys = ['AIA']
    loader.seconds_per_dt = 86400.0
    loader.ref_date = datetime(2025, 1, 1)
    image_coordinates = np.zeros((2, 3, 2)) * u.arcsec

    output = loader.load_pose(
        image_coordinates,
        pose_spherical(0.0, 0.0, 215.0),
        loader.ref_date,
        batch_size=2,
        instrument_key='AIA',
        progress=False,
        model_outputs=None,
    )

    assert output['image'].shape == (2, 3, 2)
    assert output['calibration_regularization'].shape == ()
    np.testing.assert_allclose(output['instrument_gain_delta_dex'], [0.1, -0.1])
    np.testing.assert_allclose(output['common_gain_delta_dex'], 0.2)

    with pytest.raises(ValueError, match='did not produce requested outputs'):
        loader.load_pose(
            image_coordinates,
            pose_spherical(0.0, 0.0, 215.0),
            loader.ref_date,
            batch_size=2,
            instrument_key='AIA',
            progress=False,
            model_outputs=['missing'],
        )


def test_load_spherical_uses_artifact_distance_normalization():
    loader = SuNeRFLoader.__new__(SuNeRFLoader)
    loader.Rs_per_ds = 2.0
    loader.seconds_per_dt = 86400.0
    loader.ref_date = datetime(2025, 1, 1)
    captured = {}

    def capture(query_points, **kwargs):
        del kwargs
        captured['query_points'] = query_points
        return query_points

    loader.load_coords = capture
    loader.load_spherical(
        latitude_range=np.array([0.0]) * u.deg,
        longitude_range=np.array([0.0]) * u.deg,
        radius_range=np.array([2.0]) * u.solRad,
        time=loader.ref_date,
    )

    np.testing.assert_allclose(
        captured['query_points'][0, 0, 0, 0, :3], [1.0, 0.0, 0.0], atol=1e-7
    )


def test_exported_euv_maps_retain_units_and_response_identity():
    loader = SuNeRFLoader.__new__(SuNeRFLoader)
    loader.instrument_metadata = {
        'AIA': {
            'channels': [{
                'id': 'A171',
                'wavelength_angstrom': 171,
                'measurement_unit': 'DN / s',
                'response_id': 'sha256:response',
            }],
            'response': {
                'sha256': 'a' * 64,
                'provenance': {
                    'sensitivity_convention': 'reference_epoch',
                    'calibration_epoch': '2012-08-01T00:00:00Z',
                    'measurement_semantics': 'per_native_pixel',
                    'native_pixel_solid_angle_sr': 8.46e-12,
                    'native_pixel_solid_angle_relative_tolerance': 0.02,
                },
            },
            'calibration': {
                'schema': 'sunerf.response_calibration.v1',
                'mode': 'learned',
                'response_semantics': 'base_response_times_effective_gain',
                'gain_coordinate': 'base10_logarithm',
                'channel_ids': ['A171'],
                'global_reference': False,
                'density_gain_gauge': 'learned_relative_to_global_reference',
                'common_gain_delta_dex': 0.2,
                'relative_channel_gain_delta_dex': [-0.05],
                'effective_gain_delta_dex': [0.15],
                'effective_multiplicative_gain': [10.0**0.15],
            },
        }
    }
    obstime = datetime(2025, 1, 1)
    observer = SkyCoord(
        0 * u.deg,
        0 * u.deg,
        1 * u.AU,
        frame=frames.HeliographicStonyhurst,
        obstime=obstime,
    )
    reference = SkyCoord(
        Tx=0 * u.arcsec,
        Ty=0 * u.arcsec,
        observer=observer,
        obstime=obstime,
        frame=frames.Helioprojective,
    )

    output = loader.get_maps(
        np.ones((2, 3, 1)),
        reference,
        [2, 2] * u.arcsec / u.pix,
        'AIA',
    )['A171']

    assert output.meta['bunit'] == 'DN / s'
    assert output.meta['resp_id'] == 'sha256:response'
    assert output.meta['rsp_sha'] == 'a' * 64
    assert output.meta['senscon'] == 'reference_epoch'
    assert output.meta['cal_epoc'] == '2012-08-01T00:00:00Z'
    assert output.meta['radsem'] == 'per_native_pixel'
    assert output.meta['rsppxsr'] == 8.46e-12
    assert output.meta['calvers'] == 'sunerf.response_calibration.v1'
    assert output.meta['rspbase']
    assert not output.meta['calref']
    assert output.meta['calmode'] == 'learned'
    assert output.meta['comgdex'] == 0.2
    assert output.meta['relgdex'] == -0.05
    assert output.meta['effgdex'] == 0.15
    assert output.meta['effgain'] == pytest.approx(10.0**0.15)
